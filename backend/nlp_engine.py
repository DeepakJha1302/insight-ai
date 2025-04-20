import openai
import os
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_insight(df_result: pd.DataFrame):
    prompt = f"""
    Analyze this data comparison of Actual vs Predicted outcomes:
    {df_result.head(10).to_string()}
    Provide a natural language insight about how well the model performed.
    """
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content
