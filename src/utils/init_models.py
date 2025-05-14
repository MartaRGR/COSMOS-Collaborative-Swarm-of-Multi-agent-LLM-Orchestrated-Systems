# # # # This file contains the functions to call and run different generative models # # # #

# Init models
# # # #
def _init_openai(hyperparameters, model_name):
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        deployment_name=hyperparameters.get("deployment_name", model_name),
        model_name=model_name,
        api_version=hyperparameters.get("api_version", "2024-08-01-preview"),
        temperature=hyperparameters.get("temperature", 0.2),
    )

def _init_foundry():
    import os
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential
    return ChatCompletionsClient(
        endpoint=os.getenv("AZURE_INFERENCE_SDK_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_SDK_KEY")),
    )

def _init_nvidia():
    import os
    from openai import OpenAI
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
    )

# Run models
# # # # #
def _run_openai(self, text_input, task_definition):
    from langchain.prompts import ChatPromptTemplate
    messages = [("system", self.system_message), ("human", self.human_message)]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | self.llm
    return chain.invoke({"user_task": task_definition, "input_data": text_input}).content

def _run_foundry(llm, model_name, system_message, human_message, text_input, task_definition, hyperparameters):
    from azure.ai.inference.models import SystemMessage, UserMessage
    import re
    response = llm.complete(
        messages=[
            SystemMessage(content=system_message),
            UserMessage(content=human_message.format(user_task=task_definition, input_data=text_input)),
        ],
        temperature=hyperparameters.get("temperature", 0.2),
        top_p=hyperparameters.get("top_p", 0.1),
        presence_penalty=hyperparameters.get("presence_penalty", 0.0),
        frequency_penalty=hyperparameters.get("frequency_penalty", 0.0),
        model=model_name
    )
    content = response.choices[0].message.content
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL) if "deepseek" in self.model_name.lower() else content

def _run_nvidia(llm, model_name, system_message, human_message, text_input, task_definition, hyperparameters):
    response = llm.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message.format(user_task=task_definition, input_data=text_input)}
        ],
        temperature=hyperparameters.get("temperature", 0.2),
        top_p=hyperparameters.get("top_p", 0.1)
    )
    return response.choices[0].message.content

# Dispacher
# # # # #
def _get_dispatch_entry(self):
    for key in dispatcher:
        if key in model_name.lower():
            return dispatcher[key]
    raise ValueError(f"No dispatcher found for model: {model_name}")