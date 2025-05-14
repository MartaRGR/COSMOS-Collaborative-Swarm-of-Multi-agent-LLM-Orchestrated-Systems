import os

import torch
import cv2

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Interpret a combination of images and textual prompts",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["image_path"],
        REQUIRED_INPUTS_CATALOG["text_input"]
    ],
    "output": "textual interpretation of image + prompt",
    "class": "MultimodalTaskAgent",
    "models": [
        {
            "name": "Phi-3.5-vision-instruct",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "top_p": [0.1, 1.0],
                "presence_penalty": [-2.0, 2.0],
                "frequency_penalty": [-1.0, 1.0]
            }
        },
        {
            "name": "Llama-3.2-11B-Vision-Instruct",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "top_p": [0.1, 1.0],
                "presence_penalty": [-2.0, 2.0],
                "frequency_penalty": [-1.0, 1.0]
            }
        },
        {
            "name": "gpt-4o",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "api_version": "2023-03-15-preview", # TODO: revisar
                "deployment_name": "gpt-4o"
            }
        },
        {
            "name": "Phi-4-multimodal-instruct",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "top_p": [0.1, 1.0],
                "presence_penalty": [-2.0, 2.0],
                "frequency_penalty": [-1.0, 1.0]
            }
        },
        {
            "name": "mistral-medium-3-instruct",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "top_p": [0.1, 1.0]
            }
        }
    ]
}


class MultimodalTaskAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Object Detection - Crew {crew_id}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using Device: {self.device}')
        self.model_name = model_name

        # Dispatch map for model loaders
        self.dispatcher = {
            "phi": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "llama": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "gpt": {
                "init": self._init_openai,
                "run": self._run_openai
            },
            "mistral": {
                "init": self._init_mistral,
                "run": self._run_mistral
            },
        }

        # Find and execute the loader
        model = self.model_name.lower()
        for key, loader in model_loaders.items():
            if key in model:
                self.model = loader()
                return

        raise ValueError(f"Unsupported model: {self.model_name}")



if __name__ == "__main__":
    config = {
        "model": "Llama-3.2-11B-Vision-Instruct",
        "hyperparameters": {
            "temperature": 0.5
        }
    }

    # detector = ObjectDetectionAgent("crew_1", config)
    # print(detector.run(input_data={"image_path": "pruebas/istockphoto-1346064470-612x612.jpg"}))
