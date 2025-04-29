import os
from PIL import Image
import base64
import json
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem, ImageContentItem, ImageUrl

# Configuración
endpoint   = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT", "https://marta-m8snggnd-eastus2.services.ai.azure.com/models")
key        = os.getenv("AZURE_INFERENCE_SDK_KEY", "DCcpC9jDrzOQqWVmZvrPK52MgTx77QfE5sARL0UMTOUkJcy5KIz7JQQJ99BCACHYHv6XJ3w3AAAAACOGSDCI")
vision_mod = os.getenv("DEPLOYMENT_NAME", "Llama-3.2-11B-Vision-Instruct") #"Llama-3.2-11B-Vision-Instruct"#"Phi-3.5-vision-instruct"

# 1) Detectar objetos en la imagen usando la API de Vision-Instruct
client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
local_image_path = "istockphoto-1346064470-612x612.jpg"
image_format = "jpeg" if os.path.splitext(local_image_path)=="jpg" else "png"
with Image.open(local_image_path) as img:
    image_width, image_height = img.size

# Leer imagen y codificar en base64
with open(local_image_path, "rb") as f:
    image_bytes = f.read()

image_data = base64.b64encode(image_bytes).decode("utf-8")
data_url = f"data:image/{image_format};base64,{image_data}"

# Crear el mensaje para la API
system_prompt = (
    "You are an object detection assistant. "
    "Given an image, return a JSON array of objects. "
    "Return ONLY a valid JSON array. No text before or after. No bullet points. No explanations. "
    # f"The image has a resolution of {image_width} × {image_height} pixels. "
    # "All bbox values must be in absolute pixel coordinates. "
    "Each object must include: "
    "- class (string) "
    "- confidence (float between 0 and 1) "
    # "- bbox (object): with xmin, ymin, xmax, ymax. All in absolute pixel coordinates, not normalized. "
    # "Do not return any floating point values for coordinates. Use only pixel units. "
    # "- area (float): computed as (xmax - xmin) × (ymax - ymin), in pixel units of the bbox."
)
vision_msgs=[
        SystemMessage(content=system_prompt),
        UserMessage(content=[
            TextContentItem(text="Detect objects into the following image:"),
            ImageContentItem(image_url=ImageUrl(url=data_url))
        ]),
    ]

vision_resp = client.complete(
    model=vision_mod,
    messages=vision_msgs,
    # max_tokens=1000,
    temperature=0.0
)
raw = vision_resp.choices[0].message.content
detections = json.loads(raw)
print("Detections:", json.dumps(detections, indent=2))
