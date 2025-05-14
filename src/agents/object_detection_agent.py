import torch
import cv2

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Detect objects in an image",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["image_path"]
    ],
    "output": "list of detected objects",
    "class": "ObjectDetectionAgent",
    "models": [
        {
            "name": "yolo11n",
            "hyperparameters": {
                "classes": []
            }
        },
        {
            "name": "yolov8n",
            "hyperparameters": {
                "classes": []
            }
        },
        {
            "name": "resnet50",
            "hyperparameters": {
                "weights": []
            }
        },
        {
            "name": "Llama-3.2-11B-Vision-Instruct",
            "hyperparameters": {
                "temperature": [0,1]
            }
        },
        {
            "name": "Phi-3.5-vision-instruct",
            "hyperparameters": {
                "temperature": [0,1]
            }
        }
    ]
}


class ObjectDetectionAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Object Detection - Crew {crew_id}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using Device: {self.device}')
        self.model_name = model_name

        # Dispatch map for model loaders
        model_loaders = {
            "yolo": self._load_yolo,
            "resnet50": self._load_resnet50,
            "llama-3.2-11b-vision-instruct": self._load_azure_vision_llm,
            "phi-3.5-vision-instruct": self._load_azure_vision_llm,
        }

        # Find and execute the loader
        model = self.model_name.lower()
        for key, loader in model_loaders.items():
            if key in model:
                self.model = loader()
                return

        raise ValueError(f"Unsupported model: {self.model_name}")

    def _load_yolo(self):
        from ultralytics import YOLO
        return YOLO(f"{self.model_name}.pt").to(self.device)

    def _load_resnet50(self):
        import torchvision.models as models
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        model.eval()
        return model

    @staticmethod
    def _load_azure_vision_llm():
        import os
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        return ChatCompletionsClient(
            endpoint=os.getenv("AZURE_INFERENCE_SDK_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_SDK_KEY"))
        )

    def run(self, input_data):
        """
        Execute the object detection agent logic:
        - Load the image.
        - Make the prediction.
        - Return the results.
        """
        # Getting image_path from input data
        image_path = input_data.get("image_path")
        if not image_path:
            self.logger.error("No image path provided.")
            return None
        self.logger.info(f"Running object detection agent on image: {image_path}")
        image = self.load_image(image_path)
        self.logger.info(f"Image loaded: {image_path}")
        results = self.predict(image, **self.hyperparameters)
        return results

    def load_image(self, path: str):
        """Load an image using OpenCV."""
        if self.model_name.lower() in ["llama-3.2-11b-vision-instruct", "phi-3.5-vision-instruct"]:
            import base64
            import os
            # For Llama and Phi vision models, load the image and codify in base64
            with open(path, "rb") as f:
                image_bytes = f.read()
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            image_format = "jpeg" if os.path.splitext(path) == "jpg" else "png"
            return f"data:image/{image_format};base64,{image_data}"
        return cv2.imread(path)

    def predict(self, frame, **kwargs):
        """
        Run the prediction on the frame.
        The logic here will depend on the model:
        - For YOLO, Llama and Phi vision models: run object detection.
        - For ResNet50: run classification
        """
        # Dispatch map definition
        dispatch_map = {
            "yolo": lambda: self._predict_yolo(
                frame,
                kwargs.get("classes", []),
                kwargs.get("conf", 0.5)
            ),
            "resnet50": lambda: self._predict_resnet50(frame),
            "llama-3.2-11b-vision-instruct": lambda: self._predict_vision_model(
                frame,
                kwargs.get("temperature", 0)
            ),
            "phi-3.5-vision-instruct": lambda: self._predict_vision_model(
                frame,
                kwargs.get("temperature", 0)
            )
        }

        # Match model name to prediction function
        model = self.model_name.lower()
        for key, func in dispatch_map.items():
            if key in model:
                return func()
        raise ValueError(f"Model '{self.model_name}' not supported for prediction.")

    def _predict_yolo(self, frame, classes, conf):
        """Runs YOLO object detection."""
        self.logger.info(">>> Running YOLO object detection...")
        predict_args = {"source": frame, "conf": conf}
        if classes:
            predict_args["classes"] = classes
        results = self.model.predict(**predict_args)
        objects = self.extract_objects(results)
        self.logger.info(f"Detected objects: {objects}")
        return objects

    def _predict_resnet50(self, frame):
        """Runs object classification using ResNet50."""
        self.logger.info(">>> Running ResNet50 object classification...")
        # Load ImageNet class labels
        imagenet_classes = self._fetch_imagenet_classes()

        # Preprocess the input frame
        image_tensor = self._preprocess_image(frame)

        # Perform prediction
        with torch.no_grad():
            model_outputs = self.model(image_tensor)
            confidence, predicted_class_idx = torch.nn.functional.softmax(model_outputs, dim=1).max(1)

        predicted_label = imagenet_classes[predicted_class_idx.item()]
        objects = [{"class": predicted_label, "confidence": confidence.item()}]
        self.logger.info(f"Detected objects: {objects}")
        return objects

    def _predict_vision_model(self, frame, temperature):
        """Runs object detection using Llama or Phi vision model."""
        import json
        from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem, ImageContentItem, ImageUrl

        self.logger.info(f">>> Running {self.model_name} vision model...")
        # Prepare the request to the Azure Vision model
        system_prompt = (
            "You are an object detection assistant. "
            "Given an image, return a JSON array of objects. "
            "Return ONLY a valid JSON array. No text before or after. No bullet points. No explanations. "
            "Each object must include: "
            "- class (string) "
            "- confidence (float between 0 and 1) "
        )
        vision_resp = self.model.complete(
            model=self.model_name,
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=[
                    TextContentItem(text="Detect objects into the following image:"),
                    ImageContentItem(image_url=ImageUrl(url=frame))
                ]),
            ],
            temperature=temperature
        )
        return json.loads(vision_resp.choices[0].message.content)

    @staticmethod
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

    @staticmethod
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

    # @staticmethod
    # def save_results(frame, save_path):
    #     """Saves the image with results to the specified path."""
    #     cv2.imwrite(save_path, frame)
    #     print(f"Results saved in {save_path}")

    @staticmethod
    def _fetch_imagenet_classes():
        """Fetch ImageNet class labels from the defined URL."""
        import requests
        return requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" # Imagenet Classes URL
        ).text.splitlines()

    def _preprocess_image(self, frame):
        """Preprocess the input frame into a tensor suitable for the model."""
        from PIL import Image
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image_pil = Image.fromarray(image)  # Convert to PIL Image
        return transform(image_pil).unsqueeze(0).to(self.device)


if __name__ == "__main__":
    config = {
        "model": "Llama-3.2-11B-Vision-Instruct",
        "hyperparameters": {
            "temperature": 0.5
        }
    }

    # config = {
    #     "model": "yolov8n",
    #     "hyperparameters": {
    #         "conf": 0.5,
    #         "classes": []
    #     }
    # }

    # config = {
    #     "model": "resnet50",
    #     "hyperparameters": {
    #         "classes": []
    #     }
    # }

    detector = ObjectDetectionAgent("crew_1", config)
    print(detector.run(input_data={"image_path": "pruebas/istockphoto-1346064470-612x612.jpg"}))
