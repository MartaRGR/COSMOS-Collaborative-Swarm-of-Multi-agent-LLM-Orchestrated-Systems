import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Read LLM configuration from environment variables
_ = load_dotenv(find_dotenv())  # read local .env file

llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    model_name="gpt-4o-mini",
    temperature=0
)

llm.invoke("tell me a joke")