import torch
import cv2
from ultralytics import YOLO
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
import operator
from typing import Annotated, List
from typing_extensions import TypedDict

AGENT_METADATA = {
    "function": "Detect objects in an image",
    "input": "image",
    "output": "list of detected objects",
    "models": ["YOLOv11n", "YOLOv8n"]
}


class ObjectDetectionAgent:
    def __init__(self, model_name):
        """Initializes the YOLO model with the appropriate device."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using Device: {self.device}')
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        """Loads the specified YOLO model."""
        model = YOLO(model_name)
        return model

    def predict(self, frame, classes=[], conf=0.5):
        """
        Makes predictions on the given image.
        Args:
            frame: The image on which prediction will be made.
            classes: (Optional) List of classes to filter.
            conf: (Optional) Minimum confidence threshold.
        Returns:
            results: Prediction results.
        """
        if classes:
            results = self.model.predict(frame, classes=classes, conf=conf)
        else:
            results = self.model.predict(frame, conf=conf)
        return results

    def extract_objects(self, results):
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
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                width, height = x2 - x1, y2 - y1
                area = width * height

                # Extract class and probability
                class_id = int(box.cls[0])
                predicted_obj = result.names[class_id]
                confidence = float(box.conf[0])

                # Save object information
                detected_objects.append({
                    "class": predicted_obj,
                    "confidence": confidence,
                    "area": area,
                    "bbox": [x1, y1, x2, y2]
                })

        return detected_objects

    def plot_bboxes(self, frame, detected_objects, rectangle_thickness=2, text_thickness=1):
        """
        Draws bounding boxes and labels on the image.
        Args:
            frame: The image on which results will be drawn.
            detected_objects: List of detected objects.
            rectangle_thickness: Thickness of the rectangle.
            text_thickness: Thickness of the text.
        Returns:
            frame: The image with drawn results.
        """
        for obj in detected_objects:
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

    def save_results(self, frame, save_path):
        """Saves the image with results to the specified path."""
        cv2.imwrite(save_path, frame)
        print(f"Results saved in {save_path}")


class SystemAgentState(TypedDict):
    """
    The state of the Object Detection Agent.
    """
    image_path: str  # Path where the image to process is located
    detected_objects: Annotated[list, operator.add]  # List of detected objects
    user_feedback: str  # Human feedback
    finished_detection: bool  # Whether detection is finished


class SystemAgent:
    def __init__(self, llm, model_name):
        """
        Initializes the Object Detection Agent.
        Args:
            llm: The language model used by the agent.
            model_name: The YOLO model name to be used for object detection.
        """
        self.llm = llm
        self.detector = ObjectDetectionAgent(model_name)
        self.graph = self._build_agent()
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)

    def _build_agent(self) -> StateGraph:
        """
        Builds the state graph for the object detection agent.
        Returns:
            graph: The compiled state graph.
        """

        def object_detection(state: SystemAgentState):
            """Performs object detection on the given image."""
            image = cv2.imread(state["image_path"])
            results = self.detector.predict(image, conf=0.5)
            detected_objects = self.detector.extract_objects(results)
            state["detected_objects"] = detected_objects
            return state

        def human_feedback(state: SystemAgentState):
            """Asks the user for feedback on the detected objects."""
            print("Detected objects:")
            for obj in state["detected_objects"]:
                print(f"Class: {obj['class']}, Confidence: {obj['confidence']:.2f}, Area: {obj['area']}, BBox: {obj['bbox']}")
            feedback = input("Are you satisfied with the results? (yes/no): ")
            state["user_feedback"] = feedback
            state["finished_detection"] = feedback.lower() == "yes"
            return state

        def check_detection(state: SystemAgentState):
            """Checks if the detection process should be repeated based on user feedback."""
            if state["finished_detection"]:
                return state
            else:
                print("Repeating object detection with a different configuration...")
                state["detected_objects"] = []
                return state

        def finish_detection(state: SystemAgentState):
            """Finishes the detection process."""
            print("Object detection process finished.")
            state["finished_detection"] = True
            return state

        graph = StateGraph(SystemAgentState)
        graph.add_node("object_detection", object_detection)
        graph.add_node("human_feedback", human_feedback)
        graph.add_node("check_detection", check_detection)
        graph.add_node("finish_detection", finish_detection)

        graph.add_edge(START, "object_detection")
        graph.add_edge("object_detection", "human_feedback")
        graph.add_conditional_edges(
            "human_feedback",
            lambda s: "finish_detection" if s["finished_detection"] else "object_detection",
            ["finish_detection", "object_detection"]
        )
        graph.add_edge("finish_detection", END)

        return graph
