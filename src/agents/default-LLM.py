from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Default LLM agent.",
    "required_inputs": [
         {
            "variable": "task_definition",
            "description": "The description of the task to be solved by the LLM agent."
         }
    ],
    "output": "text result of the agent's action.",
    "class": "DefaultLlmAgent",
    "models": [
        {
            "name": "gpt-4o-mini",
            "hyperparameters": {
                "temperature": [0,1],
                "api_version": "2023-03-15-preview",
                "deployment_name": "gpt-4o-mini"
            }
        }
    ]
}


class DefaultLlmAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Default LLM - Crew {crew_id}")
        self.model_name = model_name

    def run(self, user_input, task_definition: str = None):
        """Execute the default LLM agent logic"""
        llm_model = self._initialize_llm(**self.config.get("hyperparameters", {}))
        result = self._run_llm(user_input, task_definition, llm_model)
        self.logger.info(f">>> Task result:\n{result.content}")
        return result.content

    def _initialize_llm(self, **kwargs):
        temperature = kwargs.get("temperature", 0.2)
        deployment_name = kwargs.get("deployment_name", self.model_name)
        api_version = kwargs.get("api_version", "2024-08-01-preview")
        if not (0.0 <= temperature <= 1.0):
            temperature = 0.2
        try:
            return AzureChatOpenAI(
                deployment_name=deployment_name,
                model_name=self.model_name,
                api_version=api_version,
                temperature=temperature)
        except Exception as e:
            error_message = f"Failed to initialize AzureChatOpenAI: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)

    def _run_llm(self, input_data, task_definition, llm):
        try:
            messages = [
                ("system", "You are an AI assistant that can solve tasks and use external tools when necessary.."),
                ("human", "Please, resolve this task {user_task} with the following input: {input_data}"),]
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | llm
            return chain.invoke({
                "user_task": task_definition,
                "input_data": input_data
            })
        except Exception as e:
            error_message = f"Failed to run AzureChatOpenAI: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)


if __name__ == "__main__":

    config = {
        "model": "gpt-4o-mini",
        "hyperparameters": {
            "temperature": 0.7,
            "api_version": "2024-08-01-preview",
            "deployment_name": "gpt-4o-mini"
        }
    }

    # Initialize the object detector
    agent = DefaultLlmAgent(1, config)
    agent.run("tell me the weather in Madrid")

