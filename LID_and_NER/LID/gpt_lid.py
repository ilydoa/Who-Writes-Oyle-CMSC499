import os
import pandas as pd
import datetime
import re
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

with open("lid_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

DATASET = "dataset_path"
df = pd.read_csv(DATASET)

client = OpenAI(api_key="key")


# *** LID: GPT-4o *** #
def get_lid_labels(post):
    response = client.responses.create(
        model="gpt-4o",
        instructions=system_prompt,
        input=post,
        temperature=0
    )
    return response.output_text


def parse_lid_output(result):
    """
    Parse model output of the form:

        1|TR
        2|EN
        3|NE

    Returns a dictionary mapping index -> label.
    """

    labels = {}

    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*(TR|EN|MIXED|OTHER|NE|AMBIGUOUS)\s*$"
    )

    for line in result.splitlines():
        line = line.strip()

        match = pattern.match(line)

        if match:
            idx = int(match.group(1))
            tag = match.group(2)
            labels[idx] = tag

    return labels


# *** COLLECTING LABELS *** #

all_labels = []
total_posts = df["doc_id"].nunique()

for doc_id, group in tqdm(
    df.groupby("doc_id"),
    total=total_posts,
    desc="Processing posts"
):

    # Create numbered token input:
    # 1: token1
    # 2: token2
    # 3: token3
    numbered_tokens = "\n".join(
        f"{i}: {token}"
        for i, token in enumerate(group["token"].tolist(), start=1)
    )

    result = get_lid_labels(numbered_tokens)
    label_dict = parse_lid_output(result)

    expected_indices = list(range(1, len(group) + 1))
    returned_indices = sorted(label_dict.keys())

    # Check whether every token index received exactly one label
    if returned_indices == expected_indices:
        ordered_labels = [
            label_dict[i]
            for i in expected_indices
        ]
        all_labels.extend(ordered_labels)

    else:
        all_labels.extend(["UNK"] * len(group))


# Add predictions
df["gpt_langid"] = all_labels


# Save to CSV
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

name = output_dir / f"gpt_lid_{timestamp}.csv"

df.to_csv(name, index=False)