import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Interpret and resolve language-based tasks (e.g., summarization, classification, Q&A)",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["task_definition"],
        REQUIRED_INPUTS_CATALOG["text_input"]
    ],
    "output": "textual response (summary, classification, etc.)",
    "class": "LanguageTaskAgent",
    "models": [
        {
            "name": "gpt-4o-mini",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "api_version": "2023-03-15-preview",
                "deployment_name": "gpt-4o-mini"
            }
        },
        {
            "name": "Phi-4-mini-instruct",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "top_p": [0.1, 1.0],
                "presence_penalty": [-2.0, 2.0],
                "frequency_penalty": [-1.0, 1.0]
            }
        },
        {
            "name": "Llama-3.3-70B-Instruct",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "top_p": [0.1, 1.0],
                "presence_penalty": [-2.0, 2.0],
                "frequency_penalty": [-1.0, 1.0]
            }
        },
        {
            "name": "DeepSeek-R1",
            "hyperparameters": {}
        },
        {
            "name": "qwen/qwen2.5-coder-32b-instruct",
            "hyperparameters": {
                "temperature": [0.0, 1.0],
                "top_p": [0.1, 1.0]
            }
        }
    ]
}


class LanguageTaskAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Default LLM - Crew {crew_id}")
        self.model_name = model_name

        self.llm = self._initialize()
        self.system_message = "You are an AI assistant that can solve tasks and use external tools when necessary."
        self.human_message = "Please, resolve this task {user_task} with the following input: {input_data}"

    def _initialize(self):
        return self._get_dispatch_entry()["init"]()

    @staticmethod
    def _read_file(text_input):
        """Read the content of a file."""
        ext = os.path.splitext(text_input)[-1].lower()
        if ext in [".txt", ".json", ".csv"]:
            import chardet
            with open(text_input, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(text_input, 'r', encoding=encoding) as f:
                return f.read()
        elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            import base64
            with open(text_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif ext == ".docx":
            from docx import Document
            doc = Document(text_input)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def run(self, input_data):
        try:
            task_definition = input_data.get("task_definition", None)
            text_input = input_data.get("text_input", None)
            if isinstance(text_input, str) and os.path.isfile(text_input):
                text_input = self._read_file(text_input)
            result = self._get_dispatch_entry()["run"](text_input, task_definition)
            self.logger.info(f">>> Task result:\n{result}")
            return result
        except Exception as e:
            self.logger(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    # config = {
    #     "model": "qwen/qwen2.5-coder-32b-instruct",
    #     "hyperparameters": {}
    # }

    config = {
        "model": "Phi-4-mini-instruct",
        "hyperparameters": {
            "temperature": 1,
            "top_p": 0.1,
            "presence_penalty": 2,
            "frequency_penalty": 1
        }
    }

    # config = {
    #     "model": "Phi-4-mini-instruct",
    #     "hyperparameters": {
    #         "temperature": 0.6,
    #         "top_p": 0.19,
    #         "presence_penalty": -1.02,
    #         "frequency_penalty": -0.38
    #     }
    # }

    # config = {
    #     "model": "gpt-4o-mini",
    #     "hyperparameters": {
    #         "temperature": 0.7,
    #         "api_version": "2024-08-01-preview",
    #         "deployment_name": "gpt-4o-mini"
    #     }
    # }

    # Initialize the object detector
    agent = LanguageTaskAgent(1, config)
    agent.run({
        "task_definition": "resumen el siguiente archivo",
        "text_input": "Blog_La_importancia_de_la_anonimización_de_datos.txt"
    })
