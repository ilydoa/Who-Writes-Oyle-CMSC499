import os
import pandas as pd
import datetime
import re
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import InferenceClient #NOT SUPPORTED FOR THIS MODEL; NEED TRANSFORMERS LIBRARY

with open("prompts/lid_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()


DATASET = "../data_input/cleaned_annotated_dataset.csv"
df = pd.read_csv(DATASET)

HF_TOKEN = os.environ["HF_TOKEN"]

client = InferenceClient(
    api_key=HF_TOKEN,
)


def get_lid_labels(post):

    response = client.chat_completion(
        model="WiroAI/wiroai-turkish-llm-9b:featherless-ai",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": post,
            },
        ],
        max_tokens=512,
        temperature=0,
    )

    return response.choices[0].message


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


all_labels = []

total_posts = df["doc_id"].nunique()


for doc_id, group in tqdm(
    df.groupby("doc_id"),
    total=total_posts,
    desc="Processing posts"
):

    numbered_tokens = "\n".join(
        f"{i}: {token}"
        for i, token in enumerate(group["token"].tolist(), start=1)
    )

    result = get_lid_labels(numbered_tokens)

    label_dict = parse_lid_output(result)

    expected_indices = list(range(1, len(group) + 1))
    returned_indices = sorted(label_dict.keys())

    if returned_indices == expected_indices:

        ordered_labels = [
            label_dict[i]
            for i in expected_indices
        ]

        all_labels.extend(ordered_labels)

    else:
        # Preserve dataframe alignment
        all_labels.extend(["UNK"] * len(group))


df["wiro_langid"] = all_labels


output_dir = Path("../data_output")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = (output_dir /f"wiro_lid_{timestamp}.csv")
df.to_csv(output_file,index=False)


