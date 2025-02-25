import torch
import cv2
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO
from PIL import Image
import requests


AGENT_METADATA = {
    "function": "Detect objects in an image",
    "input": "image",
    "output": "list of detected objects",
    "models": [
        {
            "name": "yolo11n",
            "hyperparameters": {
                "classes": [1,5]
            }
        },
        {
            "name": "yolov8n",
            "hyperparameters": {
                "classes": [5,8]
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


class ObjectDetection:
    def __init__(self, model_name):
        """Initializes the model based on the selected name."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using Device: {self.device}')
        self.model_name = model_name
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        """Loads the selected object detection or classification model."""
        if "yolo" in model_name.lower():
            return YOLO(f"{model_name.lower()}.pt").to(self.device)
        elif model_name.lower() == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
            model.eval()
            return model
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def load_image(image_path):
        """Read the image from path using OpenCV."""
        return cv2.imread(image_path)

    def predict(self, frame, classes=[], conf=0.5):
        """
        Runs object detection or classification based on the selected model.
        Args:
            frame: The image to process.
            classes: List of classes to filter (only for YOLO).
            conf: Minimum confidence threshold.
        Returns:
            List of detected objects.
        """
        if "yolo" in self.model_name.lower():
            return self._predict_yolo(frame, classes, conf)
        elif self.model_name.lower() == "resnet50":
            return self._predict_resnet50(frame)
        else:
            raise ValueError(f"Model {self.model_name} not supported for prediction.")

    def _predict_yolo(self, frame, classes, conf):
        """Runs YOLO object detection."""
        if classes:
            results = self.model.predict(frame, classes=classes, conf=conf)
        else:
            results = self.model.predict(frame, conf=conf)
        return self.extract_objects(results)

    def _predict_resnet50(self, frame):
        """Runs object classification using ResNet50."""
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        imagenet_classes = requests.get(url).text.splitlines()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image_pil = Image.fromarray(image)  # Convert to PIL Image
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted_class = outputs.max(1)

        predicted_label = imagenet_classes[predicted_class.item()]
        return [{"class": predicted_label, "confidence": torch.nn.functional.softmax(outputs, dim=1).max().item()}]

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

    @staticmethod
    def save_results(frame, save_path):
        """Saves the image with results to the specified path."""
        cv2.imwrite(save_path, frame)
        print(f"Results saved in {save_path}")


if __name__ == "__main__":

    # Choose model
    model_type = "resnet50"

    # Initialize the object detector
    detector = ObjectDetection(model_type)

    # Load image
    image_path = "pruebas/istockphoto-1346064470-612x612.jpg"
    image = detector.load_image(image_path)

    # Predict objects
    detected_objects = detector.predict(image, conf=0.5)

    # Display results in console
    if detected_objects:
        print("Detected objects:")
        [print(f"Class: {obj['class']}, Confidence: {obj['confidence']:.2f}") for obj in detected_objects]

    # If using YOLO, draw bounding boxes
    if "yolo" in model_type.lower():
        result_img = detector.plot_bboxes(image, detected_objects)
        cv2.imshow("Results", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        detector.save_results(result_img, "results.jpg")
