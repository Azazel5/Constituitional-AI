from huggingface_hub import InferenceClient


def query_llm(query):
    client = InferenceClient()


    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=query
    )

    return completion.choices[0].message.content
