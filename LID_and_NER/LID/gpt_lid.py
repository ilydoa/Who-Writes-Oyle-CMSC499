import os
import pandas as pd
import datetime
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

#Collect text posts 
DATASET = "../data/input/fixed_annotations.csv"
df = pd.read_csv(DATASET)
tokens = df["token"].to_list()

#Set up client
client=OpenAI(api_key="key")

#*** LID: GPT4o ***#
completions = []
for token in tqdm(tokens, desc="Processing LID"):
    prompt = f"""
                You will receive a word token. Identify the language of the token as either:
                Turkish
                English
                Mixed (Turkish and English)
                Other (a different language)
                Named entity (representing a or part of a named entity)
                Ambiguous

                Reply with either "TR" for Turkish, "EN" for English, "MIXED" for mixed,
                "OTHER" for other, "AMBIGUOUS" for ambiguous, or "NE" for named entity.
                Output the response and nothing else.

                Here is the token:
    
            {token}
            """
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    completions.append(response.output_text)

df["gpt_langid"] = completions


#Save to CSV
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'gpt_lid_{timestamp}.csv'
df.to_csv(name)