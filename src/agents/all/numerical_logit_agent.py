from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Perform structured and numerical reasoning by solving STEM, mathematical or arithmetic multiple-choice question problems, interpreting data and providing justifications for logical decisions or predictions.",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["text"]
    ],
    "output": "Explanation and solution steps for the given quantitative or logical problem",
    "class": "NumericReasoningAgent",
    "models": [
        MODELS_CATALOG["gpt-4o-mini"],
        MODELS_CATALOG["o3-mini"],
        MODELS_CATALOG["DeepSeek-R1"],
        MODELS_CATALOG["Phi-4-mini-instruct"],
        MODELS_CATALOG["Llama-3.3-70B-Instruct"]
    ]
}


class NumericReasoningAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Numeric Reasoning Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.model_name = model_name

        self.model = self._initialize()
        self.system_message = (
            "You are an advanced AI assistant specialized in numerical reasoning, quantitative analysis, and structured data interpretation.\n"
            "You excel at solving mathematical and physical problems, interpreting tables, charts, and graphs, and performing logical inference over numeric content.\n"
            "You can:\n"
            "- Solve STEM problems with step-by-step reasoning, using multiple approaches if needed.\n"
            "- Justify choices between alternatives through logical inference.\n"
            "- Analyze small datasets (tables or charts), perform statistical reasoning, and predict outcomes based on trends or models.\n"
            "Always explain your reasoning clearly and choose the most rigorous method available."
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
            self.logger.error(f"Failed to run {self.model_name}: {e}")
            raise


if __name__ == "__main__":
    # config = {
    #     "model": "gpt-4o-mini",
    #     "hyperparameters": {
    #         "temperature": 0.5,
    #         "api_version": "2024-12-01-preview",
    #         "deployment_name": "gpt-4o"
    #     }
    # }

    config = {
        "model": "o3-mini",
        "hyperparameters": {
            "api_version": "2024-12-01-preview",
            "deployment_name": "o3-mini"
        }
    }

    # config = {
    #     "model": "nvidia/llama-3.1-nemotron-nano-8b-v1",
    #     "hyperparameters": {
    #         "temperature": 0.5
    #     }
    # }

    # config = {
    #     "model": "DeepSeek-R1",
    #     "hyperparameters": {
    #     }
    # }

    agent = NumericReasoningAgent("crew_1", config)
    agent.run({
        "task_definition": "Resuelve el siguiente problema matemático y explica tu razonamiento paso a paso.",
        "text": "Un colega le pregunta al Profesor Otto las edades de sus tres hijas y este responde que el producto de sus edades es igual a 36 y que la suma es igual al número del portal de enfrente. El colega mira el portal en cuestión y, tras pensar un momento, dice que le falta un dato. Entonces el profesor Otto asiente y dice: 'La mayor toca el piano'. ¿Qué edades tienen las tres hijas del Profesor Otto?"
    })
