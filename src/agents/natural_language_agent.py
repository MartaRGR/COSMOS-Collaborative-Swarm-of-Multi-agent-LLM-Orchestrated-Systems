from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Interpret and resolve language-based tasks (e.g., summarization, classification, Q&A)",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["task_definition"],
        REQUIRED_INPUTS_CATALOG["text"]
    ],
    "output": "textual response (summary, classification, etc.)",
    "class": "LanguageTaskAgent",
    "models": [
        MODELS_CATALOG["gpt-4o-mini"],
        MODELS_CATALOG["Phi-4-mini-instruct"],
        MODELS_CATALOG["Llama-3.3-70B-Instruct"],
        MODELS_CATALOG["DeepSeek-R1"],
        MODELS_CATALOG["qwen/qwen2.5-coder-32b-instruct"]
    ]
}


class LanguageTaskAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Language Task Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.model_name = model_name

        self.model = self._initialize()
        self.system_message = (
            "You are an advanced AI assistant specialized in language-based tasks.\n"
            "Your core capabilities include comprehension, synthesis, inference, and text transformation.\n"
            "You can perform complex tasks such as analyzing contradictory opinions, generating divergent and unified summaries, "
            "and comparing model-generated responses across different architectures and sizes."
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
    # config = {
    #     "model": "qwen/qwen2.5-coder-32b-instruct",
    #     "hyperparameters": {
    #         "temperature": 0.87,
    #         "top_p": 0.12
    #     }
    # }

    # config = {
    #     "model": "Phi-4-mini-instruct",
    #     "hyperparameters": {
    #         "temperature": 1,
    #         "top_p": 0.1,
    #         "presence_penalty": 2,
    #         "frequency_penalty": 1
    #     }
    # }

    # config = {
    #     "model": "Phi-4-mini-instruct",
    #     "hyperparameters": {
    #         "temperature": 0.6,
    #         "top_p": 0.19,
    #         "presence_penalty": -1.02,
    #         "frequency_penalty": -0.38
    #     }
    # }

    config = {
        "model": "gpt-4o-mini",
        "hyperparameters": {
            "temperature": 0.7,
            "api_version": "2024-08-01-preview",
            "deployment_name": "gpt-4o-mini"
        }
    }

    # Initialize the object detector
    agent = LanguageTaskAgent(1, config)
    agent.run({
        "task_definition": "resume el siguiente archivo",
        "text": "Blog_La_importancia_de_la_anonimización_de_datos.txt"
    })
