import os
import csv
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from refine import model

load_dotenv()
deep_seek_api = os.getenv('deep_seek_api')

if not deep_seek_api:
    raise ValueError("API key 'deep_seek_api' not found in environment variables")

refined_prompt = model()

if not refined_prompt:
    raise ValueError("refined_prompt is empty or None")

# Updated system prompt to be more explicit about 1000 rows
system = """Generate a complete CSV dataset based on the following prompt.

CRITICAL REQUIREMENTS:
- Generate EXACTLY 1000 rows of data (not including header)
- Maximum number of columns: 8
- Minimum number of columns: 1
- Return ONLY the CSV data with headers
- Do NOT include any explanatory text, notes, or truncation messages
- Do NOT use placeholders like "... [Additional rows]..."
- Generate realistic, varied data for all 1000 rows

Output Format: Pure CSV data starting with headers, followed by exactly 1000 data rows."""

def generate_multiple_batches():
    """Generate dataset in multiple smaller batches and combine"""
    all_rows = []
    headers = None
    batch_size = 200
    total_batches = 5 
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=deep_seek_api,
    )
    
    for batch_num in range(total_batches):
        print(f"Generating batch {batch_num + 1}/{total_batches}...")
        
        batch_system = f"""Generate exactly {batch_size} rows of CSV data based on the prompt.
{'Include headers in the first row.' if batch_num == 0 else 'Do NOT include headers, only data rows.'}
Return pure CSV format with no explanatory text."""
        
        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[
                    {
                        "role": "system",
                        "content": batch_system
                    },
                    {
                        "role": "user",
                        "content": f"{refined_prompt}\n\nGenerate batch {batch_num + 1} with {batch_size} rows."
                    }
                ],
                max_tokens=4000,
                temperature=0.8 
            )
            
            batch_output = completion.choices[0].message.content.strip()
            batch_lines = [line.strip() for line in batch_output.split('\n') if line.strip() and ',' in line]
            
            if batch_num == 0 and batch_lines:
                headers = batch_lines[0]
                all_rows.extend(batch_lines)
            elif batch_lines:
                all_rows.extend(batch_lines)
                
        except Exception as e:
            print(f"Error generating batch {batch_num + 1}: {e}")
            continue
    
    if all_rows:
        final_csv = '\n'.join(all_rows)
        print(f"\nFinal dataset generated with {len(all_rows)-1} rows")
        print(final_csv)
        with open('generated_dataset_batched.csv', 'w', newline='') as f:
            f.write(final_csv)
        print("\nDataset saved to 'generated_dataset_batched.csv'")
        
        return final_csv
    
    return None

if __name__ == "__main__":
    print("\n Generating data set \n")
    result = generate_multiple_batches()
