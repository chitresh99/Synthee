import os
import csv
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()

try:
    from refine import model
    refine_available = True
except ImportError:
    refine_available = False

try:
    from system_prompts.dataset_config import dataset_config_prompt
    config_available = True
except ImportError:
    config_available = False

try:
    from logger import logger
    logger_available = True
except ImportError:
    logger_available = False

def generate_multiple_batches(user_prompt):
    """Generate dataset in multiple smaller batches and combine"""
    deep_seek_api = os.getenv('deep_seek_api')
    if not deep_seek_api:
        st.error("API key 'deep_seek_api' not found in environment variables")
        return None
    
    try:
        if refine_available:
            refined_prompt = model(user_prompt)
            if not refined_prompt:
                st.error("refined_prompt is empty or None")
                return None
        else:
            refined_prompt = user_prompt
            
        if logger_available:
            logger.info(f"Using refined prompt: {refined_prompt}")
            
    except Exception as e:
        st.error(f"Error generating refined prompt: {e}")
        return None
    
    all_rows = []
    headers = None
    batch_size = 200
    total_batches = 5 
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=deep_seek_api,
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_num in range(total_batches):
        status_text.text(f"Generating batch {batch_num + 1}/{total_batches}...")
        progress_bar.progress((batch_num + 1) / total_batches)
        
        batch_system = f"""Generate exactly {batch_size} rows of CSV data based on the prompt.
        {'Include headers in the first row.' 
        if batch_num == 0 
        else 'Do NOT include headers, only data rows.'}
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
            st.error(f"Error generating batch {batch_num + 1}: {e}")
            continue
    
    status_text.text("Dataset generation complete!")
    
    if all_rows:
        final_csv = '\n'.join(all_rows)
        st.success(f"Final dataset generated with {len(all_rows)-1} rows")
        return final_csv
    
    return None