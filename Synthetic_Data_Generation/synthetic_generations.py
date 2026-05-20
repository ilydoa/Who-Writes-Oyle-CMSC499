import os
import pandas as pd
import datetime
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

client=OpenAI("key")

topics_list = ["business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships",
               "business", "gaming", "music", "tv", "sports", "cooking", "vacation", 
               "shopping", "university", "work", "health", "finance", "family", "school", 
               "environment", "online", "home", "news", "love", "relationships"]

completions = []

for topic in tqdm(topics_list, desc = "Processing topics:"):
    prompt = f"""Code-mixing refers to a phenomenon of combining 
    two or more languages in increasing sentences. 
    Can you generate a code-mixed Turkish-English sentence about {topic}?

    Return only the generated code-mixed sentence.
                """
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    completions.append(response.output_text)

df = pd.DataFrame({"topic": topics_list, "synthetic_code_mixed": completions})
output_dir = Path("data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'albanian_gpt_synthetic_data_{timestamp}.csv'
df.to_csv(name)