import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Perform structured reasoning on tabular or numerical inputs",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["structured_data"]
    ],
    "output": "reasoned textual answer or numerical prediction",
    "class": "NumericReasoningAgent",
    "models": [
        {
            "name": "gpt-4o-mini",
            "hyperparameters": {
                "temperature": [0.0, 0.7]
            }
        },
        {
            "name": "DeepSeek-R1",
            "hyperparameters": {
                "temperature": [0.0, 0.7]
            }
        }
    ]
}



if __name__ == "__main__":
    config = {
        "model": "Llama-3.2-11B-Vision-Instruct",
        "hyperparameters": {
            "temperature": 0.5
        }
    }

    # detector = ObjectDetectionAgent("crew_1", config)
    # print(detector.run(input_data={"image_path": "pruebas/istockphoto-1346064470-612x612.jpg"}))
