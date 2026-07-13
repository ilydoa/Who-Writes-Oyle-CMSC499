import os
import pandas as pd
import datetime
import re
import time
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

with open("prompts/ner_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

#Collect text posts from selected dataset
DATASET = "../data_input/cleaned_annotated_dataset.csv"
df = pd.read_csv(DATASET)

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

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = output_dir / "qwen_ner_checkpoint.csv"

#Checkpoint file

if checkpoint_file.exists():
    checkpoint_df = pd.read_csv(checkpoint_file)

    if len(checkpoint_df) == len(df):
        df = checkpoint_df
        print("Checkpoint loaded.")
    else:
        print("Checkpoint size mismatch. Starting over.")

if "qwen_ner" not in df.columns:
    df["qwen_ner"] = pd.NA

#*** QWEN NER ***#

def get_ner_labels(post, retries=6):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                extra_body={},
                model="qwen/qwen3-8b",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": post
                            },
                        ]
                    }
                ]
            )
            return completion.choices[0].message.content
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,            
        ) as e:
            wait = min(2 ** attempt, 60)
            print(
                f"\nRequest failed ({type(e).__name__}). "
                f"Retrying in {wait} seconds..."
            )
            time.sleep(wait)
    raise RuntimeError("Maximum retries exceeded.")

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

processed_docs = 0
doc_groups = list(df.groupby("doc_id"))

for doc_id, group in tqdm(doc_groups, desc="Processing posts"):

    # Skip documents already completed
    if group["qwen_ner"].notna().all():
        continue
    numbered_tokens = "\n".join(
        f"{i}: {token}"
        for i, token in enumerate(group["token"].tolist(), start=1)
    )    
    try:
        result = get_ner_labels(numbered_tokens)
    except Exception as e:
        print(f"\nFailed on document {doc_id}")
        print(e)
        continue

    label_dict = parse_ner_output(result)

    expected_indices = list(range(1, len(group) + 1))
    returned_indices = sorted(label_dict.keys())
    
    #Check to ensure exact same number of items as the group is returned
    if returned_indices == expected_indices:
        ordered_labels = [
            label_dict[i]
            for i in expected_indices
        ]
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
        ordered_labels = ["UNK"] * len(group)
        
    df.loc[group.index, "qwen_ner"] = ordered_labels
    processed_docs += 1

    if processed_docs % 25 == 0:
        df.to_csv(checkpoint_file, index=False)
        print(f"\nCheckpoint saved ({processed_docs} documents).")

    time.sleep(0.5)


#Save to CSV
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

final_file = output_dir / f"qwen_ner_{timestamp}.csv"

df.to_csv(final_file, index=False)

# Remove checkpoint after successful completion
if checkpoint_file.exists():
    checkpoint_file.unlink()

print(f"\nFinished!\nSaved to:\n{final_file}")