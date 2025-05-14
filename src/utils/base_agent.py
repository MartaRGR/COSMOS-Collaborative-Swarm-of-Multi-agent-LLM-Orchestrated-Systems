from abc import ABC, abstractmethod


AGENT_METADATA = {
    "function": "", # Replace by the agent's purpose. For example: Detect objects in an image.
    "required_inputs": "", # Replace by the agent's required inputs. Use the required input catalog, for example, image_path, task_definition, etc.
    "output": "", # Replace by the agent's output type. For example: list of detected objects, text, etc.
    "class": "", # Replace by the agent's class name, in this example, BaseAgent.
    "models": [ # Replace by the list of available agent's models
        {
            "name": '', # Replace by the name of one of the available agent's model, i.e., yolo11n.pt.
            "hyperparameters": {
                # Replace by the dictionary of the available hyperparameters
            }
        }
    ]
}

class BaseAgent(ABC):
    def __init__(self, crew_id: int, config: dict):
        """
        Initialize the agent with the provided configuration.
        The config dictionary must contain at least:
        - 'model': Name or identifier of the model to use.
        - 'hyperparameters': Dictionary containing the hyperparameters.
        """
        self.logger = None
        self.crew_id = crew_id
        self.config = config
        self.model_name = config.get("model")
        self.hyperparameters = config.get("hyperparameters", {})
        self.model = None
        self.device = None

        self.llm = None
        self.system_message = None
        self.human_message = None

        # Initialize the dispatcher for different model types
        self.dispatcher = {
            "gpt": {
                "init": self._init_openai,
                "run": self._run_openai
            },
            "deepseek": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "llama": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "phi": {
                "init": self._init_foundry,
                "run": self._run_foundry
            },
            "qwen": {
                "init": self._init_nvidia,
                "run": self._run_nvidia
            },
            "mistral": {
                "init": self._init_mistral,
                "run": self._run_mistral
            }
        }

        # Initialize the agent
        self._setup_agent(self.model_name, self.crew_id)

    # Dispatcher logic
    def _get_dispatch_entry(self):
        for key in self.dispatcher:
            if key in self.model_name.lower():
                return self.dispatcher[key]
        raise ValueError(f"No dispatcher found for model: {self.model_name}")

    # Init models' functions
    # # # # #
    def _init_openai(self):
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            deployment_name=self.hyperparameters.get("deployment_name", self.model_name),
            model_name=self.model_name,
            api_version=self.hyperparameters.get("api_version", "2024-08-01-preview"),
            temperature=self.hyperparameters.get("temperature", 0.2),
        )

    @staticmethod
    def _init_foundry():
        import os
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        return ChatCompletionsClient(
            endpoint=os.getenv("AZURE_INFERENCE_SDK_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_SDK_KEY")),
        )

    @staticmethod
    def _init_nvidia():
        import os
        from openai import OpenAI
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
        )

    @staticmethod
    def _init_mistral():
        return "https://integrate.api.nvidia.com/v1/chat/completions"

    # Run models' functions
    # # # # #
    def _run_openai(self, text_input, task_definition):
        from langchain.prompts import ChatPromptTemplate
        messages = [("system", self.system_message), ("human", self.human_message)]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        return chain.invoke({"user_task": task_definition, "input_data": text_input}).content

    def _run_foundry(self, text_input, task_definition):
        from azure.ai.inference.models import SystemMessage, UserMessage
        import re
        response = self.llm.complete(
            messages=[
                SystemMessage(content=self.system_message),
                UserMessage(content=self.human_message.format(user_task=task_definition, input_data=text_input)),
            ],
            temperature=self.hyperparameters.get("temperature", 0.2),
            top_p=self.hyperparameters.get("top_p", 0.1),
            presence_penalty=self.hyperparameters.get("presence_penalty", 0.0),
            frequency_penalty=self.hyperparameters.get("frequency_penalty", 0.0),
            model=self.model_name
        )
        content = response.choices[0].message.content
        return re.sub(
            r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
        ) if "deepseek" in self.model_name.lower() else content

    def _run_nvidia(self, text_input, task_definition):
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.human_message.format(user_task=task_definition, input_data=text_input)}
            ],
            temperature=self.hyperparameters.get("temperature", 0.2),
            top_p=self.hyperparameters.get("top_p", 0.1)
        )
        return response.choices[0].message.content

    def _run_mistral(self, text_input, task_definition, stream=False):
        import os
        import requests

        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC')}",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
            "model": 'mistralai/mistral-medium-3-instruct',
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user","content": self.human_message.format(user_task=task_definition, input_data=text_input)}
            ],
            "temperature": self.hyperparameters.get("temperature", 0.2),
            "top_p": self.hyperparameters.get("top_p", 0.1),
            "presence_penalty": self.hyperparameters.get("presence_penalty", 0.0),
            "stream": stream
        }

        response = requests.post(self.llm, headers=headers, json=payload)
        return response.json()

    # Abstract methods
    @abstractmethod
    def _setup_agent(self, model_name: str, crew_id: int):
        """
        Configures the initial settings of the agent (for example, model selection based on name or identifier).
        This method must be implemented by each specific agent, as its loading logic may vary.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Executes the agent's main logic.
        This method must be implemented by each specific agent.
        """
        pass
