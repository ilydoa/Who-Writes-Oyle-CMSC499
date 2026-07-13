import os
import pandas as pd
import datetime
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import re

with open("prompts/ner_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

#Collect text posts from selected dataset
DATASET = "../data_input/cleaned_annotated_dataset.csv"
df = pd.read_csv(DATASET)

client = OpenAI(api_key="key")

#*** NER: GPT4o ***#
def get_ner_labels(post):
    response = client.responses.create(
        model="gpt-4o",
        instructions=system_prompt,
        input=post,
        temperature=0
    )
    return response.output_text

def parse_ner_output(result):
    """
    Parse model output of the form:

        1|B-ORG
        2|I-ORG
        3|O

    Returns a dictionary mapping index -> label.
    """

    labels = {}

    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*(B-PER|I-PER|B-ORG|I-ORG|B-LOC|I-LOC|B-GROUP|I-GROUP|B-PROD|I-PROD|B-TITLE|I-TITLE|B-EVENT|I-EVENT|B-TIME|I-TIME|B-OTHER|I-OTHER|O)\s*$"
    )

    for line in result.splitlines():
        line = line.strip()

        match = pattern.match(line)

        if match:
            idx = int(match.group(1))
            tag = match.group(2)
            labels[idx] = tag

    return labels

#***COLLECTING LABELS***#
all_labels = []
total_posts = df['doc_id'].nunique()

for doc_id, group in tqdm(df.groupby('doc_id'), total=total_posts, desc="Processing posts"):

    numbered_tokens = "\n".join(
        f"{i}: {token}"
        for i, token in enumerate(group["token"].tolist(), start=1)
    )    

    result = get_ner_labels(numbered_tokens)
    label_dict = parse_ner_output(result)

    expected_indices = list(range(1, len(group) + 1))
    returned_indices = sorted(label_dict.keys())

    #Check to ensure exact same number of items as the group is returned
    if returned_indices == expected_indices:
        ordered_labels = [
            label_dict[i]
            for i in expected_indices
        ]
        all_labels.extend(ordered_labels)

    else:
        #Fallback if any tokens missed/hallucinated

        print("\n" + "=" * 80)
        print(f"WARNING: Index mismatch in post {doc_id}")

        print(f"Expected number of labels: {len(expected_indices)}")
        print(f"Received number of labels: {len(returned_indices)}")

        missing = sorted(set(expected_indices) - set(returned_indices))
        extra = sorted(set(returned_indices) - set(expected_indices))

        if missing:
            print(f"Missing indices: {missing}")

        if extra:
            print(f"Unexpected indices: {extra}")

        print("\nINPUT:")
        print(numbered_tokens)

        print("\nRAW MODEL OUTPUT:")
        print(result)

        print("=" * 80 + "\n")

        # Preserve dataframe alignment
        all_labels.extend(["UNK"] * len(group))
        
df['gpt_ner'] = all_labels


#Save to CSV
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'gpt_ner_{timestamp}.csv'
df.to_csv(name)