from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.models_catalog import MODELS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "When asked a question about specific text or file (not a generic one), this agent is used to locate the answer before passing it on to a natural language agent.",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["query"],
        REQUIRED_INPUTS_CATALOG["text"]
    ],
    "output": "A dictionary containing the list of text parts that answer the question with their scores.",
    "class": "EmbeddingAgent",
    "models": [
        MODELS_CATALOG["text-embedding-ada-002"],
        MODELS_CATALOG["text-embedding-3-large"],
        MODELS_CATALOG["ibm-granite/granite-embedding-107m-multilingual"],
        MODELS_CATALOG["sentence-transformers/all-MiniLM-L6-v2"],
        MODELS_CATALOG["intfloat/multilingual-e5-small"]
    ]
}
# TODO: aumentar plan azure (s0 no vale)


class EmbeddingAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Embedding Agent - Crew {crew_id} - Model {model_name} - Hyperparameters {self.hyperparameters}")
        self.model_name = model_name
        self.model = self._initialize()

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
    config = {
        "model": "text-embedding-3-large"
    }
    # config = {
    #     "model": "intfloat/multilingual-e5-small"
    # }
    # config = {
    #     "model": "ibm-granite/granite-embedding-107m-multilingual"
    # }
    # config = {
    #     "model": "sentence-transformers/all-MiniLM-L6-v2"
    # }

    agent = EmbeddingAgent("crew_1", config)
    agent.run({
        "task_definition": ".",
        "query": "What is the purpose of this Regulation?",
        "text": r"C:\Users\rt01306\Downloads\OJ_L_202401689_EN_TXT.pdf"
    })
