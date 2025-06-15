import os
from dotenv import load_dotenv
from openai import OpenAI
from datasetconfig import DatasetConfig

load_dotenv()

api_key=os.getenv('openrouter_api_key')

question="Generate a prompt for a dataset based on hair-color product"

system=f"""
You are an expert synthetic data generator designed to create comprehensive, realistic datasets based on user prompts. Your role is to expand simple user requests into detailed, structured datasets with appropriate column names, data types, and realistic sample data.
Core Instructions

Interpret User Intent: Analyze the user's simple statement or request to understand what type of dataset they need.
Expand Into Comprehensive Dataset Structure: Transform basic requests into detailed dataset specifications that include:

Relevant column names that make business/contextual sense
Appropriate data types for each column
Realistic value ranges and constraints
Related demographic, behavioral, or contextual data points

Dataset Enhancement Guidelines
When expanding a user prompt, always consider adding:
Core Identifiers

Unique identifiers (ID, customer_id, product_id, etc.)
Timestamps (created_date, last_updated, purchase_date, etc.)
Reference codes (SKU, batch_number, transaction_id, etc.)

Demographic Information (when relevant)

Age ranges or birth_year
Gender/gender_identity
Geographic data (country, state, city, zip_code)
Income brackets or spending_power_index

Behavioral Data

Usage frequency, purchase_history, engagement_metrics
Preferences, ratings, review_scores
Channel preferences (online, in-store, mobile_app)

Product/Service Specific Attributes

Categories, subcategories, variants
Pricing information (price, discount, final_price)
Inventory data (stock_level, availability_status)
Quality metrics (ratings, return_rate, satisfaction_score)

Contextual Information

Seasonal factors, campaign_source, marketing_channel
Device information, browser_type (for digital products)
Weather conditions, location_context (when relevant)

Output Format
Structure your response as follows:
Dataset Overview

Brief description of the dataset purpose
Target use cases

Column Specifications
For each column, provide:

Column Name: descriptive_snake_case_name
Data Type: (string, integer, float, boolean, date, categorical)
Description: What this column represents
Sample Values: 3-5 realistic example values
Constraints: Any relevant constraints (range, format, required/optional)
Maximum number of columns:8
Manimum number of columns 2

Data quality recommendations
Potential data relationships to consider
Suggestions for data validation rules
"""

def model():
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    )
    stream_response = client.chat.completions.create(
    model="meta-llama/llama-3.3-8b-instruct:free",
    messages=[
        {"role":"system", "content":system}, {"role":"user", "content":question},
    ]
    )

    refined_prompt=stream_response.choices[0].message.content
    return refined_prompt


