#imports
import os
import pandas as pd
import datetime
from pathlib import Path

#fasttext set-up
import fasttext
model = fasttext.load_model('../lid.176.bin')

#tokens set-up
DATASET = "dataset_name"
df = pd.read_csv(DATASET)
tokens = df["token"].to_list()
df.drop(columns=["borrowed_suffix", "ner"]) 

langids = []
for token in tokens:
    predictions = model.predict(token)
    langids.append(predictions[0][0].replace('__label__', '').upper())
df["fasttext_lid"] = langids

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f"fasttext_lid_{timestamp}.csv"
df.to_csv(name)