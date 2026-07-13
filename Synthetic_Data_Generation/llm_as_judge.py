import os
import json
import pandas as pd
import re
import datetime
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

client=OpenAI(api_key="key")
DATASET = "dataset_name"
df = pd.read_csv(DATASET)
samples = df["synthetic_code_mixed"].to_list()
completions = []

for sample in tqdm(samples, desc = "Processing topics:"):
    prompt = f"""You will be presented with a code-mixed sentence. 
    Your task is to evaluate the sentence based on three separate metrics. 
    Assuming the readers are people familiar with each language in the sentence.
    
    Evaluation Criteria:
    Coherence (1-3): Assesses how well the sentence elements are connected and 
    flow together, considering the mixing of languages.
    
    1: Poor. The sentence lacks logical flow or connection between its parts, 
    making it hard to understand.
    2: Fair. The sentence has some logical connections between its parts, but the
    flow might be interrupted by awkward language mixing.
    3: Good. The sentence demonstrates a clear and logical connection between
    its parts, with the mixing of languages not hindering understanding.
    
    Naturalness (1-3): Evaluate the sentence for its natural-sounding language
    use and integration of the code-mixed elements.
    
    1: Poor. The sentence sounds unnatural or forced, with the mixing of languages
    seeming out of place.
    2: Fair. The sentence sounds somewhat natural, though the integration of 
    different languages can occasionally feel awkward.
    3: Good. The sentence sounds natural and the mixing of languages appears
    seamless and intentional.
    
    Readability (1-3): Measures how easy it is to read and understand the sentence, 
    considering the impact of code-mixing on readability.
    1: Poor. The sentence is difficult to read, with the mixing of languages
    significantly hindering comprehension.
    2: Fair. The sentence is readable, though the reader may need to pause
    to understand the mixed languages.
    3: Good. The sentence is easy to read, with the code-mixing enhancing or
    not detracting from the ability to understand the content.
    
    Output your evaluation as valid JSON only. Do not include any additional text.

    {{
      "analysis": "<concise and refined evaluation analysis>",
      "coherence": <1-3>,
      "naturalness": <1-3>,
      "readability": <1-3>
    }}

    Here is the sentence: {sample}
    """

    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    result = json.loads(response.output_text)
    completions.append(result)

results_df = pd.DataFrame(completions)
df = pd.concat([df, results_df], axis=1)

output_dir = Path("data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'llm_as_judge_tr_synthetic_cm_{timestamp}.csv'
df.to_csv(name, index=False)