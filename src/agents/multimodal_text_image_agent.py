from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Interpret a combination of images and textual prompts",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["image_path"]
    ],
    "output": "textual interpretation of image + prompt",
    "class": "MultimodalTaskAgent",
    "models": [
        MODELS_CATALOG["Phi-3.5-vision-instruct"],
        MODELS_CATALOG["Llama-3.2-11B-Vision-Instruct"],
        MODELS_CATALOG["gpt-4o"],
        MODELS_CATALOG["Phi-4-multimodal-instruct"],
        MODELS_CATALOG["mistral-medium-3-instruct"]
    ]
}


class MultimodalTaskAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Multimodal Task Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.model_name = model_name

        self.model = self._initialize()
        self.system_message = (
            "You are a multimodal AI assistant that specializes in interpreting images in conjunction with textual prompts.\n"
            "Your role is to provide reflective, symbolic, critical, or contextual analysis of images when guided by a specific instruction or task.\n"
            "Use your understanding of visual storytelling, cultural cues, and language to generate thoughtful interpretations."
        )
        self.human_message = "Please, resolve this task {user_task} with the following input: {input_data}"


    def _initialize(self):
        return self._get_dispatch_entry()["init"]()


    def run(self, input_data):
        try:
            result = self._get_dispatch_entry()["run"](input_data)
            self.logger.info(f">>> Task result:\n{result}")
            return result
        except Exception as e:
            self.logger(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    config = {
        "model": "mistral-medium-3-instruct",
        "hyperparameters": {
            "temperature": 0.5
        }
    }

    # config = {
    #     "model": "gpt-4o",
    #     "hyperparameters": {
    #         "temperature": 0.5
    #     }
    # }

    detector = MultimodalTaskAgent("crew_1", config)
    print(detector.run({
        "task_definition": "describe la siguiente imagen",
        "image_path": "pruebas/istockphoto-1346064470-612x612.jpg"
    }))
