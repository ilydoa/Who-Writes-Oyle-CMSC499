import os
import pandas as pd
import datetime
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import re

#Collect text posts from selected dataset
DATASET = "dataset_name"
df = pd.read_csv(DATASET)

client=OpenAI(api_key="key")

prompt_text = """Analyze the following text and identify named entities based on the 
    predefined categories and definitions below. You will be using BIO tags
    (B- for beginning token of an entity, I- for tokens within the same entity,
    O for tokens that do not belong to any entity)

    1. B-PER / I-PER: Person. Includes real or fictional. 
    2. B-ORG / I-ORG: Organization. Companies, institutions, corporations, 
    etc. Context should support that token refers to an organization (e.g., Facebook 
    as an organization vs. Facebook as the website application).
    3. B-LOC / I-LOC: Location.
    4. B-GROUP / I-GROUP: Group. Sports teams, music bands, etc.
    5. B-PROD / I-PROD: Product. Devices, medicine, websites, etc.
    6. B-TITLE / I-TITLE: Title. Titles of movies, books, TV shows, songs, etc.
    7. B-EVENT / I-EVENT: Event. Does not include holidays.
    8. B-TIME / I-TIME: Time. Months, days of the week, seasons, holidays, etc.
    9. B-OTHER / I-OTHER: Other.
    10. O: Non-entity token.

    You must split the text into words exactly by spaces. 
    Return a JSON array of objects, where each object has "token" and "label".
    Output the response and nothing else.

    Text: "George lived in Washington."
    Output: 
    [
    {"token": "George", "label": "B-PER"},
    {"token": "lived", "label": "O"},
    {"token": "in", "label": "O"},
    {"token": "Washington.", "label": "B-LOC"}
    ]
    """

#*** NER: GPT4o ***#
def get_ner_labels(post):
    prompt = f"""
    {prompt_text}
    {post}
    """
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    return response.output_text

#*** Parse output ***#
def parse_llm_json_response(raw_response_string):
    #Clean the string
    cleaned = raw_response_string.strip()
    
    #Extract only the text inside the ```json ... ``` block
    match = re.search(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
    
    if match:
        json_content = match.group(1)
    else:
        #Fallback if the LLM occasionally omits the markdown block
        json_content = cleaned

    try:
        #Parse into a list of dictionaries
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON string. Error: {e}")
        return []

#***COLLECTING LABELS***#
all_labels = []
total_posts = df['doc_id'].nunique()

for doc_id, group in tqdm(df.groupby('doc_id'), total=total_posts, desc="Processing posts"):
    reconstructed_post = ' '.join(group['token'].tolist())
    
    result = get_ner_labels(reconstructed_post)
    llm_output = parse_llm_json_response(result)

    #Check to ensure exact same number of items as the group is returned
    if len(llm_output) == len(group):
        labels = [item['label'] for item in llm_output]
    else:
        #Fallback if any tokens missed/hallucinated
        print(f"Warning: Token count mismatch in post {doc_id} (Expected {len(group)}, got {len(llm_output)}). Using fallback alignment.")
        labels = ['O'] * len(group)
        
    all_labels.extend(labels)

df['gpt_ner'] = all_labels


#Save to CSV
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'gpt_ner_{timestamp}.csv'
df.to_csv(name)