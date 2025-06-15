import os
from dotenv import load_dotenv
from openai import OpenAI
from refine import model

load_dotenv()
gemma_api_key = os.getenv('gemma_api_key')

if not gemma_api_key:
    raise ValueError("API key 'gemma_api_key' not found in environment variables")

refined_prompt = model()

if not refined_prompt:
    raise ValueError("refined_prompt is empty or None")

system = "Generate the dataset based on the following prompt. The dataset can have maximum 1000 rows and minimum 50 rows. Return the data in CSV format with headers."

def generate_dataset():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=gemma_api_key, 
    )
    
    try:
        completion = client.chat.completions.create(
            model="google/gemma-3n-e4b-it:free",
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": refined_prompt
                }
            ]
        )
        output = completion.choices[0].message.content
        print(output)
        return output
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        return None


if __name__ == "__main__":
    generate_dataset()