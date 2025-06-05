from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Default LLM agent",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["task_definition"]
    ],
    "output": "text result of the agent's action",
    "class": "DefaultLlmAgent",
    "models": [
        MODELS_CATALOG["gpt-4o-mini"],
        MODELS_CATALOG["Phi-4-mini-instruct"],
        MODELS_CATALOG["Llama-3.3-70B-Instruct"],
        MODELS_CATALOG["DeepSeek-R1"],
        MODELS_CATALOG["qwen/qwen2.5-coder-32b-instruct"]
    ]
}


class DefaultLlmAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Language Task Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.model_name = model_name

        self.model = self._initialize()
        self.system_message = (
            "You are a highly capable AI assistant designed to interpret and solve any given task across a wide range of domains. "
            "You can understand, reason, analyze, and generate text based on user-provided instructions. " 
            "Always provide clear, structured, and relevant outputs tailored to the task. "
            "If the task requires multiple steps, outline them before presenting your solution. If the task is ambiguous, explain your interpretation before proceeding. "
            "Respond only to the user instruction and avoid unnecessary elaboration or assumptions."
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
        "model": "DeepSeek-R1",
        "hyperparameters": {}
    }

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
    #     "model": "gpt-4o-mini",
    #     "hyperparameters": {
    #         "temperature": 0.7,
    #         "api_version": "2024-08-01-preview",
    #         "deployment_name": "gpt-4o-mini"
    #     }
    # }

    # Initialize the object detector
    agent = DefaultLlmAgent(1, config)
    agent.run(input_data={"task_definition": "tell me the weather in Madrid"})

