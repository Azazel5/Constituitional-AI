from openai import OpenAI
from config import OPENROUTER_DEEPSEEK
from huggingface_hub import InferenceClient

def query_llm(query):
    client = InferenceClient()


    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=query
    )

    return completion.choices[0].message.content

def query_llm_openrouter(query):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_DEEPSEEK,
    )

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324",
        messages=query
    )

    return completion.choices[0].message.content