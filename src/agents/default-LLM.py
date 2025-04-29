from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.utils.required_inputs_catalog import REQUIRED_INPUTS_CATALOG
from src.utils.setup_logger import get_agent_logger
from src.utils.base_agent import BaseAgent


AGENT_METADATA = {
    "function": "Default LLM agent.",
    "required_inputs": [
        REQUIRED_INPUTS_CATALOG["task_definition"]
    ],
    "output": "text result of the agent's action.",
    "class": "DefaultLlmAgent",
    "models": [
        {
            "name": "gpt-4o-mini",
            "hyperparameters": {
                "temperature": [0, 1],
                "api_version": "2023-03-15-preview",
                "deployment_name": "gpt-4o-mini"
            }
        },
        {
            "name": "Phi-4-mini-instruct",
            "hyperparameters": {
                "temperature": [0, 1],
                "top_p": [0.1, 1],
                "presence_penalty": [-2, 2],
                "frequency_penalty": [-1, 1]
            }
        },
        {
            "name": "DeepSeek-R1",
            "hyperparameters": {}
        }
    ]
}


class DefaultLlmAgent(BaseAgent):
    def _setup_agent(self, model_name: str, crew_id: int):
        self.logger = get_agent_logger(f"Default LLM - Crew {crew_id}")
        self.model_name = model_name

    def run(self, user_input, task_definition: str = None):
        """Execute the default LLM agent logic"""
        llm_model = self._initialize_llm()
        result = self._run_llm(user_input, task_definition, llm_model)
        self.logger.info(f">>> Task result:\n{result}")
        return result

    def _initialize_llm(self):
        try:
            if "gpt" in self.model_name.lower():
                from langchain_openai import AzureChatOpenAI
                return AzureChatOpenAI(
                    deployment_name=self.hyperparameters.get("deployment_name", self.model_name),
                    model_name=self.model_name,
                    api_version=self.hyperparameters.get("api_version", "2024-08-01-preview"),
                    temperature=self.hyperparameters.get("temperature", 0.2))
            else:
                import os
                from azure.ai.inference import ChatCompletionsClient
                from azure.core.credentials import AzureKeyCredential
                return ChatCompletionsClient(
                    endpoint=os.getenv("AZURE_INFERENCE_SDK_ENDPOINT"),
                    credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_SDK_KEY"))
                )

        except Exception as e:
            error_message = f"Failed to initialize {self.model_name}: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)


    def _run_llm(self, input_data, task_definition, llm):
        try:
            system_message = "You are an AI assistant that can solve tasks and use external tools when necessary."
            human_message = "Please, resolve this task {user_task} with the following input: {input_data}"
            if "gpt" in self.model_name.lower():
                from langchain.prompts import ChatPromptTemplate
                messages = [("system", system_message), ("human", human_message),]
                prompt = ChatPromptTemplate.from_messages(messages)
                chain = prompt | llm
                return chain.invoke({"user_task": task_definition, "input_data": input_data}).content
            else:
                from azure.ai.inference.models import SystemMessage, UserMessage
                from azure.core.credentials import AzureKeyCredential
                response = llm.complete(
                    messages=[
                        SystemMessage(content=system_message),
                        UserMessage(content=human_message.format(user_task=task_definition, input_data=input_data)),
                    ],
                    temperature=self.hyperparameters.get("temperature", 0.2),
                    top_p=self.hyperparameters.get("top_p", 0.1),
                    presence_penalty=self.hyperparameters.get("presence_penalty", 0.0),
                    frequency_penalty=self.hyperparameters.get("frequency_penalty", 0.0),
                    model=self.model_name
                )

                if "deepseek" in self.model_name.lower():
                    import re
                    return re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL)
                return response.choices[0].message.content

        except Exception as e:
            error_message = f"Failed to run {self.model_name}: {e}"
            self.logger.error(error_message)
            raise RuntimeError(error_message)


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
    agent.run("tell me the weather in Madrid")

