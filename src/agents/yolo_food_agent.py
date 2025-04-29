import torch
import cv2

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Food recognition",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["image_path"]
    ],
    "output": "list of detected food items",
    "class": "FoodRecognitionAgent",
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
                "weights": [1.0, 5.0]
            }
        }
    ]
}


class FoodRecognitionAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Object Detection - Crew {crew_id}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using Device: {self.device}')
        self.model_name = model_name
        if "yolo" in model_name.lower():
            from ultralytics import YOLO
            self.model = YOLO(f"{model_name.lower()}.pt").to(self.device)
            # Train the model
            results = self.model.train(data="lvis.yaml", epochs=100, imgsz=640)

        elif model_name.lower() == "resnet50":
            import torchvision.models as models
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def run(self, input_data, task_definition: str = None):
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

    @staticmethod
    def load_image(path: str):
        """Load an image using OpenCV."""
        return cv2.imread(path)

    def predict(self, frame, **kwargs):
        """
        Run the prediction on the frame.
        The logic here will depend on the model:
        - For YOLO: run object detection.
        - For ResNet50: run classification
        """
        if "yolo" in self.model_name.lower():
            classes = kwargs.get("classes", [])
            conf = kwargs.get("conf", 0.5)
            return self._predict_yolo(frame, classes, conf)
        elif self.model_name.lower() == "resnet50":
            return self._predict_resnet50(frame)
        else:
            raise ValueError(f"Model {self.model_name} not supported for prediction.")
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
        "model": "yolov8n",
        "hyperparameters": {
            "conf": 0.5,
            "classes": []
        }
    }

    # config = {
    #     "model": "resnet50",
    #     "hyperparameters": {
    #         "classes": []
    #     }
    # }

    detector = ObjectDetectionAgent("crew_1", config)
    image_path = "pruebas/istockphoto-1346064470-612x612.jpg"
    detector.run(image_path)
