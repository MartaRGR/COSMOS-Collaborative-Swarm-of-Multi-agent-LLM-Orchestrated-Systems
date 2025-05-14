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
        }

        self.llm = self._initialize()
        self.system_message = "You are an AI assistant that can solve tasks and use external tools when necessary."
        self.human_message = "Please, resolve this task {user_task} with the following input: {input_data}"

    def _get_dispatch_entry(self):
        for key in self.dispatcher:
            if key in self.model_name.lower():
                return self.dispatcher[key]
        raise ValueError(f"No dispatcher found for model: {self.model_name}")

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
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        return ChatCompletionsClient(
            endpoint=os.getenv("AZURE_INFERENCE_SDK_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_SDK_KEY")),
        )

    @staticmethod
    def _init_nvidia():
        from openai import OpenAI
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
        )

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
        return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL) if "deepseek" in self.model_name.lower() else content

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
