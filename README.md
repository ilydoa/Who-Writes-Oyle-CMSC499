This is the repository for the CMSC499 "Who Writes Oyle" research project.

Scripts found here include scripts for collecting and filtering Turkish-English code-mixed posts from Eksi Sozluk, tokenizing posts, running LID and NER tests, and generating synthetic code-mixed data and annotating with LLM-as-Judge.

**/Dataset_Creation**: This folder includes the **eksi_sozluk_scraping.py** and the **tokenizer.py** scripts. **eksi_sozluk_scraping.py** contains the full script for scraping 200 pages of posts from Eksi Sozluk using a given topic list, filtering them with langdetect, and filtering them with two separate GPT4o prompts, then saving them all to a csv. **tokenizer.py** tokenizes a csv of given posts collected using the scraping script.

**/LID_and_NER**: This folder includes the **/LID** folder and the **/NER** folders. The **LID** folder contains the **wiro_ai.py**, **gpt_lid.py**, and **qwen_lid.py** scripts. Each script takes the csv creates by the tokenizer script and runs the model on each token, creating an output csv with the initial csv and a column of LID predictions. API keys need to be added or loaded in from the environment. The **NER** folder contains the **wiro_ai.py**, **gpt_ner.py**, and **qwen_ner.py** scripts. Each script takes the csv creates by the tokenizer script and runs the model on each token, creating an output csv with the initial csv and a column of NER predictions. 

Both folders also contain **/metrics** folders to calculate and visualize results, **/output** folders with some prior runs, and the **\prompts** folder with prompt text. **LID** contains a zero-shot and three-sht prompt. **NER** contains a one-shot prompt. To compare model results to true answers, the scripts need to be run with the "Cleaned Annotations" csv, in the **/data_input** folder, which contains the tokens and the human annotations for LID and NER from the Golden Dataset.

This folder also includes the **encoder_finetuning** folder, which contains **encoder_LID** and **encoder_NER**, with potential scripts for splitting the dataset, training the models, and testing model capabilities for encoder models.

**/Synthetic_Data_Generation**: This folder includes the **synthetic_generations.py** and the **llm_as_judge** scripts. **synthetic_generations.py** is a short, simple script that uses GPT4o to generate 300 code-mixed Turkish-English sentences based on different topics, and saves them all to a csv. **llm_as_judge** takes the csv with the synthetic generations and outputs a csv with LLM annotations for coherence, naturalness, and readability.

