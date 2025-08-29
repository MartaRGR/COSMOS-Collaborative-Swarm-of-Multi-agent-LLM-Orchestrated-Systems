import torch

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Detects the number of people in an image",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["image_path"],
        REQUIRED_INPUTS_CATALOG["occupancy_time"]
    ],
    "output": "number of detected people",
    "class": "OccupancyDetectorAgent",
    "models": [
        MODELS_CATALOG["yolo11n"],
        MODELS_CATALOG["yolov8n"],
        MODELS_CATALOG["Llama-3.2-11B-Vision-Instruct"],
        MODELS_CATALOG["Phi-4-multimodal-instruct"],
        MODELS_CATALOG["gpt-4o"],
        MODELS_CATALOG["resnet50"],
        MODELS_CATALOG["Phi-3.5-vision-instruct"],
        MODELS_CATALOG["mistral-medium-3-instruct"]
    ]
}



class OccupancyDetectorAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Occupancy Detector Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using Device: {self.device}')
        self.model_name = model_name

        self.model = self._initialize()
        self.system_message = (
            "You are an assistant specialized in detecting the number of people in images. "
            "Your task is to analyze the provided image and return the exact number of people detected. "
            # "Respond with a valid INTEGER only. Do not include any additional text, explanations, or formatting. "
        )
        self.human_message = "Please, resolve this task {user_task} with the following input: {input_data}"

    def _initialize(self):
        return self._get_dispatch_entry()["init"]()

    def run(self, input_data):
        try:
            result = self._get_dispatch_entry()["run"](input_data)

            if "yolo" in self.model_name or "resnet" in self.model_name:
                # Validate that result["text"] is a list of detections
                detections = result.get("text", [])
                if not isinstance(detections, list):
                    raise ValueError("The model result does not contain a list of detections.")
                # Count people
                num_people = sum(1 for detection in detections if detection.get("class") == "person")
            else:
                # Extract number from text using regular expressions
                if not isinstance(result, str):
                    raise ValueError("The model result does not contain valid text.")
                import re
                match = re.search(r'\b\d+\b', result)
                if match:
                    num_people = int(match.group())
                else:
                    self.logger.error("No number was detected in the model response. Setting num_people to 0.")
                    num_people = 0

            # Build the final result
            result = {"occupancy": {"ds": input_data["occupancy_time"], "y": num_people}}
            self.logger.info(f">>> Task result:\n{result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to run {self.model_name}: {e}")
            raise

if __name__ == "__main__":
    config = {
        "model": "Llama-3.2-11B-Vision-Instruct",
        "hyperparameters": {
            "temperature": 1
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

    detector = OccupancyDetectorAgent("crew_1", config)
    print(detector.run(input_data={
        "task_definition": "Detects people into the image",
        "image_path": "pruebas/istockphoto-1346064470-612x612.jpg",
        "occupancy_time": "2024-01-01 12:00:00"
    }))
