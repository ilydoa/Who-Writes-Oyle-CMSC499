import os
import pandas as pd
import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

#Collect text posts from dataset
DATASET = "dataset_name"
df = pd.read_csv(DATASET)
tokens = df["token"].to_list()

load_dotenv()
api_key = os.getenv("API_KEY")
if api_key is None:
    print("Error: API key not found in environment variables.")
else:
    print("API key loaded successfully.")

#Set up client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

#*** GEMMA LID ***#
completions = []
for token in tqdm(tokens, desc="Processing"):
    completion = client.chat.completions.create(
        extra_body={},
        model="google/gemma-3-12b-it",
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f"""
                    You will receive a word token. Identify the language of the token as either:
                    Turkish
                    English
                    Mixed (Turkish and English)
                    Other (a different language)
                    Named entity (representing a or part of a named entity)
                    Ambiguous

                    Reply with either "TR" for Turkish, "EN" for English, "MIXED" for mixed Turkish and English,
                    "OTHER" for other, "AMBIGUOUS" for ambiguous, or "NE" for named entity.
                    Output the response and nothing else.

                    Here is the token:
        
                {token}
            """
            },

            ]
        }
        ]
    )
    completions.append(completion.choices[0].message.content)

df["gemma_langid"] = completions

#Save to CSV
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'gemma_lid_{timestamp}.csv'
df.to_csv(name)