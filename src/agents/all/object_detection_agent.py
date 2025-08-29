import torch

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
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
        MODELS_CATALOG["yolo11n"],
        MODELS_CATALOG["yolov8n"],
        MODELS_CATALOG["resnet50"],
        MODELS_CATALOG["Llama-3.2-11B-Vision-Instruct"],
        MODELS_CATALOG["Phi-3.5-vision-instruct"]
    ]
}


class ObjectDetectionAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Object Detection - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using Device: {self.device}')
        self.model_name = model_name

        self.model = self._initialize()
        # self.system_message = (
        #     "You are an object detection assistant. "
        #     "Given an image, return a JSON array of objects. "
        #     "Return ONLY a valid JSON array. No text before or after. No bullet points. No explanations. "
        #     # f"The image has a resolution of {{image_width}} × {{image_height}} pixels. "
        #     # "All bbox values must be in absolute pixel coordinates. "
        #     "Each object must include:\n"
        #     "- class (string)\n"
        #     "- confidence (float between 0 and 1)\n"
        #     # "- bbox (object): with xmin, ymin, xmax, ymax. All in absolute pixel coordinates, not normalized.\n"
        #     # "Do not return any floating point values for coordinates. Use only pixel units. "
        #     # "If you are not sure about the coordinates, use an empty list '[]'.\n"
        #     # "- area (float): computed as (xmax - xmin) × (ymax - ymin), in pixel units of the bbox."
        # )

        self.system_message = (
            "You are an object detection assistant. "
            "Given an image, return a JSON array of objects. "
            "Return ONLY a valid JSON array. No text before or after. No bullet points. No explanations. "
            f"The image has a resolution of {{image_width}} × {{image_height}} pixels. "
            "All bbox values must be in absolute pixel coordinates. "
            "Each object must include:\n"
            "- class (string)\n"
            "- confidence (float between 0 and 1)\n"
            "- bbox (object): with xmin, ymin, xmax, ymax. All in absolute pixel coordinates, not normalized.\n"
            "Do not return any floating point values for coordinates. Use only pixel units. "
            "If you are not sure about the coordinates, use an empty list '[]'.\n"
            "- area (float): computed as (xmax - xmin) × (ymax - ymin), in pixel units of the bbox."
        )
        self.human_message = "Please, resolve this task {user_task} with the following input: {input_data}"

    def _initialize(self):
        return self._get_dispatch_entry()["init"]()

    def run(self, input_data):
        try:
            # from PIL import Image
            # image_width, image_height = Image.open(input_data.get("image_path")).size
            # self.system_message = self.system_message.format(image_width=image_width, image_height=image_height)
            result = self._get_dispatch_entry()["run"](input_data)
            self.logger.info(f">>> Task result:\n{result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    # config = {
    #     "model": "Llama-3.2-11B-Vision-Instruct",
    #     "hyperparameters": {
    #         "temperature": 0.5
    #     }
    # }

    # config = {
    #     "model": "yolov8n",
    #     "hyperparameters": {
    #         "conf": 0.5,
    #         "classes": []
    #     }
    # }

    config = {
        "model": "resnet50",
        "hyperparameters": {
            "classes": []
        }
    }

    detector = ObjectDetectionAgent("crew_1", config)
    print(detector.run(input_data={"task_definition": "Detects objects into the image", "image_path": "test_images/istockphoto-1346064470-612x612.jpg"}))
