import os
from dotenv import load_dotenv
from openai import OpenAI
from datasetconfig import DatasetConfig
from system_prompts.prompt_refiner import refiner_system_prompt

load_dotenv()

api_key = os.getenv("openrouter_api_key")

system = refiner_system_prompt()


def model(user_question):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    stream_response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick:free",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_question},
        ],
    )
    refined_prompt = stream_response.choices[0].message.content
    return refined_prompt
