from abc import ABC, abstractmethod


AGENT_METADATA = {
    "function": "", # Replace by the agent's purpose. For example: Detect objects in an image.
    "required_inputs": "", # Replace by the agent's required inputs. Use the required input catalog, for example, image_path, task_definition, etc.
    "output": "", # Replace by the agent's output type. For example: list of detected objects, text, etc.
    "class": "", # Replace by the agent's class name, in this example, BaseAgent.
    "models": [ # Replace by the list of available agent's models
        {
            "name": '', # Replace by the name of one of the available agent's model, i.e., yolo11n.pt.
            "hyperparameters": {
                # Replace by the dictionary of the available hyperparameters
            }
        }
    ]
}

class BaseAgent(ABC):
    def __init__(self, crew_id: int, config: dict):
        """
        Initialize the agent with the provided configuration.
        The config dictionary must contain at least:
        - 'model': Name or identifier of the model to use.
        - 'hyperparameters': Dictionary containing the hyperparameters.
        """
        self.logger = None
        self.crew_id = crew_id
        self.config = config
        self.model_name = config.get("model")
        self.hyperparameters = config.get("hyperparameters", {})
        self.device = None

        self.model = None
        self.system_message = None
        self.human_message = None

        # Initialize the dispatcher for different model types
        self.dispatcher = {
            "gpt": {
                "init": self._init_openai,
                "run": self._run_openai
            },
            "o3": {
                "init": self._init_openai,
                "run": self._run_openai
            },
            "deepseek": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "nvidia": {
                "init": self._init_nvidia,
                "run": self._run_nvidia
            },
            "llama": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "phi": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "qwen": {
                "init": self._init_nvidia,
                "run": self._run_nvidia
            },
            "mistral": {
                "init": self._init_mistral,
                "run": self._run_mistral
            },
            "yolo": {
                "init": self._init_yolo,
                "run": self._run_yolo
            },
            "resnet": {
                "init": self._init_resnet,
                "run": self._run_resnet
            },
        }

        # Initialize the agent
        self._setup_agent(self.model_name, self.crew_id)

    # Dispatcher logic
    def _get_dispatch_entry(self):
        for key in self.dispatcher:
            if key in self.model_name.lower():
                return self.dispatcher[key]
        raise ValueError(f"No dispatcher found for model: {self.model_name}")

    @staticmethod
    def _read_file(input_path):
        """Read the content of a file."""
        import os
        ext = os.path.splitext(input_path)[-1].lower()

        if ext in [".txt", ".json", ".csv"]:
            import chardet
            with open(input_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(input_path, 'r', encoding=encoding) as f:
                return f.read()

        elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            import base64
            with open(input_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            image_format = "jpeg" if ext == ".jpg" else "png"
            return f"data:image/{image_format};base64,{image_data}"

        elif ext in [".mp3", ".wav"]:
            import base64
            with open(input_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:audio/{ext[1:]};base64,{audio_data}"

        elif ext in [".mp4", ".mov"]:
            import base64
            with open(input_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:video/{ext[1:]};base64,{video_data}"

        elif ext == ".docx":
            from docx import Document
            doc = Document(input_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _read_input_data(self, input_data):
        from .required_inputs_catalog import REQUIRED_INPUTS_CATALOG
        import os
        for key in REQUIRED_INPUTS_CATALOG.keys():
            if key in input_data and isinstance(input_data[key], str) and os.path.isfile(input_data[key]):
                input_data[key] = self._read_file(input_data[key])
        return input_data

    @staticmethod
    def get_input_key(input_dict):
        from .required_inputs_catalog import REQUIRED_INPUTS_CATALOG
        return next(
            (key for key in REQUIRED_INPUTS_CATALOG.keys() if key != "task_definition" and key in input_dict),
            None
        )

    # Init models' functions
    # # # # #
    def _init_openai(self):
        from langchain_openai import AzureChatOpenAI
        model_config = {
            "deployment_name": self.hyperparameters.get("deployment_name", self.model_name),
            "model_name": self.model_name,
            "api_version": self.hyperparameters.get("api_version", "2024-08-01-preview")
        }
        if "mini" not in self.model_name:
            model_config["temperature"] = self.hyperparameters.get("temperature", 0.2)
        else:
            self.logger.info(f"Model '{self.model_name}' does not allow 'temperature' configuration.")
        return AzureChatOpenAI(**model_config)

    @staticmethod
    def _init_foundry():
        import os
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        return ChatCompletionsClient(
            endpoint=os.getenv("AZURE_INFERENCE_SDK_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_SDK_KEY")),
        )

    @staticmethod
    def _init_nvidia():
        import os
        from openai import OpenAI
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
        )

    @staticmethod
    def _init_mistral():
        return "https://integrate.api.nvidia.com/v1/chat/completions"

    def _init_yolo(self):
        from ultralytics import YOLO
        return YOLO(f"{self.model_name}.pt").to(self.device)

    def _init_resnet(self):
        import torchvision.models as models
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        model.eval()
        return model

    # Run models' functions
    # # # # #
    def _run_openai(self, input_data):
        from langchain.schema.messages import SystemMessage, HumanMessage
        input_data = self._read_input_data(input_data)
        messages = [SystemMessage(content=self.system_message)]
        if input_data.get("image_path"):
            messages.append(
                HumanMessage(content=[
                    {"type": "text", "text": input_data.get("task_definition")},
                    {"type": "image_url", "image_url": {"url": input_data.get("image_path")}},
                ])
            )
        elif input_data.get("text"):
            messages.append(
                HumanMessage(content=[
                    {"type": "text", "text": self.human_message.format(
                        user_task=input_data.get("task_definition"),
                        input_data=input_data.get("text", '')
                    )},
                ])
            )
        return self.model.invoke(messages).content

    def _run_foundry(self, input_data):
        import re
        from azure.ai.inference.models import (
            SystemMessage, UserMessage, TextContentItem, ImageUrl, ImageContentItem
        )

        input_data = self._read_input_data(input_data)
        messages = [SystemMessage(content=self.system_message)]
        if input_data.get("image_path"):
            messages.append(
                UserMessage(content=[
                    TextContentItem(text=input_data["task_definition"]),
                    ImageContentItem(image_url=ImageUrl(url=input_data["image_path"]))
                ])
            )
        elif input_data.get("text"):
            messages.append(
                UserMessage(content=self.human_message.format(
                    user_task=input_data["task_definition"],
                    input_data=input_data["text"]
                ))
            )
        else:
            messages.append(
                UserMessage(content=self.human_message.format(
                    user_task=input_data["task_definition"],
                    input_data=""
                ))
            )
        # TODO: poner otros formatos de archivo

        response = self.model.complete(
            messages=messages,
            temperature=self.hyperparameters.get("temperature", 0.2),
            top_p=self.hyperparameters.get("top_p", 0.1),
            presence_penalty=self.hyperparameters.get("presence_penalty", 0.0),
            frequency_penalty=self.hyperparameters.get("frequency_penalty", 0.0),
            model=self.model_name
        )
        content = response.choices[0].message.content
        return re.sub(
            r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
        ) if "deepseek" in self.model_name.lower() else content

    def _run_nvidia(self, input_data):
        import re
        input_data = self._read_input_data(input_data)
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.human_message.format(
                    user_task=input_data.get("task_definition"),
                    input_data=input_data.get(self.get_input_key(input_data))
                )}
            ],
            temperature=self.hyperparameters.get("temperature", 0.2),
            top_p=self.hyperparameters.get("top_p", 0.1)
        )
        content = response.choices[0].message.content
        return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

    def _run_mistral(self, input_data, stream=False):
        import os
        import requests

        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC')}",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        if input_data.get("image_path"):
            input_data = self._read_input_data(input_data)
            input_data["text_input"] = '<img src="' + input_data["image_path"] + '" />'

        payload = {
            "model": 'mistralai/mistral-medium-3-instruct',
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user","content": self.human_message.format(
                    user_task=input_data["task_definition"], input_data=input_data.get("text_input", ''))}
            ],
            "temperature": self.hyperparameters.get("temperature", 0.2),
            "top_p": self.hyperparameters.get("top_p", 0.1),
            "presence_penalty": self.hyperparameters.get("presence_penalty", 0.0),
            "stream": stream
        }

        response = requests.post(self.model, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]

    def _run_yolo(self, input_data, save_tagged_img=False):
        import cv2

        def extract_objects(results):
            """
            Extracts detected objects from YOLO results.
            Args:
                results: YOLO prediction results.
            Returns:
                detected_objects: List of detected objects with their information.
            """
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    # Extract rectangle coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width, height = x2 - x1, y2 - y1
                    area = width * height

                    # Extract class and probability
                    class_id = int(box.cls[0])
                    predicted_obj = result.names[class_id]
                    confidence = float(box.conf[0])

                    detected_objects.append({
                        "class": predicted_obj,
                        "confidence": confidence,
                        "area": area,
                        "bbox": [x1, y1, x2, y2]
                    })
            return detected_objects

        def plot_bboxes(frame, detected_objects, rectangle_thickness=2, text_thickness=1):
            import cv2
            """
            Draws bounding boxes and labels on the image.
            Args:
                frame: The image on which results will be drawn.
                detected_objects: List of detected objects.
            Returns:
                frame: The image with drawn results.
            """
            for obj in detected_objects:
                if "bbox" in obj:  # Only draw if there is a bounding box (YOLO results)
                    x1, y1, x2, y2 = obj['bbox']
                    confidence = obj['confidence']
                    predicted_obj = obj['class']

                    # Draw rectangle and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
                    cv2.putText(
                        frame, f"{predicted_obj} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness
                    )
            return frame

        def save_results(frame, detected_objects, save_path):
            """Saves the image with results to the specified path."""
            tagged_img = plot_bboxes(frame, detected_objects)
            cv2.imwrite(save_path, tagged_img)
            print(f"Results saved in {save_path}")

        image_path = input_data.get("image_path")
        if not image_path:
            self.logger.error("No image path provided.")
            return None
        frame = cv2.imread(image_path)
        self.logger.info(f"Image loaded: {image_path}")
        self.logger.info(">>> Running YOLO object detection...")
        predict_args = {"source": frame, "conf": self.hyperparameters.get("confidence", 0.5)}
        classes = self.hyperparameters.get("classes")
        if classes:
            predict_args["classes"] = classes
        results = self.model.predict(**predict_args)
        objects = extract_objects(results)
        self.logger.info(f"Detected objects: {objects}")
        if save_tagged_img:
            save_results(frame, objects, "results.jpg")
        return objects

    def _run_resnet(self, input_data):
        import torch
        import cv2

        def _fetch_imagenet_classes():
            """Fetch ImageNet class labels from the defined URL."""
            import requests
            return requests.get(
                "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"  # Imagenet Classes URL
            ).text.splitlines()

        def _preprocess_image(frame):
            """Preprocess the input frame into a tensor suitable for the model."""
            from PIL import Image
            import cv2
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image_pil = Image.fromarray(image)  # Convert to PIL Image
            return transform(image_pil).unsqueeze(0).to(self.device)

        self.logger.info(">>> Running ResNet50 object classification...")

        image_path = input_data.get("image_path")
        if not image_path:
            self.logger.error("No image path provided.")
            return None
        frame = cv2.imread(image_path)

        # Load ImageNet class labels
        imagenet_classes = _fetch_imagenet_classes()

        # Preprocess the input frame
        image_tensor = _preprocess_image(frame)

        # Perform prediction
        with torch.no_grad():
            model_outputs = self.model(image_tensor)
            confidence, predicted_class_idx = torch.nn.functional.softmax(model_outputs, dim=1).max(1)

        predicted_label = imagenet_classes[predicted_class_idx.item()]
        objects = [{"class": predicted_label, "confidence": confidence.item()}]
        self.logger.info(f"Detected objects: {objects}")
        return objects

    # Abstract methods
    @abstractmethod
    def _setup_agent(self, model_name: str, crew_id: int):
        """
        Configures the initial settings of the agent (for example, model selection based on name or identifier).
        This method must be implemented by each specific agent, as its loading logic may vary.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Executes the agent's main logic.
        This method must be implemented by each specific agent.
        """
        pass
