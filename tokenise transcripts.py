import os
import pandas as pd
import numpy as np
import multiprocessing
from bs4 import BeautifulSoup
import re
from bs4 import Tag
from transformers import AutoTokenizer
import spacy
import json
import nltk

def mask_names(text, custom_surnames=[]):
    doc = nlp(text)
    masked_text = []
    custom_surnames_lower = [surname.lower() for surname in custom_surnames]
    for token in doc:
        if token.ent_type_ == "PERSON" or token.text.lower() in custom_surnames_lower:
            masked_text.append("[MASK]")
        else:
            masked_text.append(token.text)
    return " ".join(masked_text)

def convert_to_lists(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (list, tuple)):
        return [convert_to_lists(item) for item in data]
    else:
        return data


def process_judgment(judgment_text, year):
    judgment_text = mask_names(judgment_text, surnames)
    judgment_text = re.sub(r'\s+', ' ', judgment_text)
    encoded_dict = tokenizer(judgment_text)
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    if len(input_ids) < 50:
        return None, None
    max_length = 510
    stride = 255
    input_ids_chunks = []
    attention_mask_chunks = []
    for i in range(0, len(input_ids), stride):
        chunk = input_ids[i:i + max_length]  # Reserve one spot for the year token  # Prepend the year token
        input_ids_chunks.append(chunk)
        attention_mask_chunks.append(attention_mask[i:i + max_length])
    output_ids = []
    out_masks = []
    for chunk, mask in zip(input_ids_chunks, attention_mask_chunks):
        if len(chunk) < 50:
            continue
        pad_width = max_length - len(chunk)
        output_ids.append(
            np.pad(chunk, (0, pad_width), mode='constant', constant_values=tokenizer.pad_token_id).tolist())
        out_masks.append(np.pad(mask, (0, pad_width), mode='constant', constant_values=0).tolist())

    return output_ids, out_masks

# Load spacy
nlp = spacy.load("en_core_web_sm")

years = list(range(1900, 2025))  # Adjust the range as needed
year_tags = [f"[YEAR_{year}]" for year in years]
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
tokenizer.add_special_tokens({"additional_special_tokens": year_tags})
# Define surnames by party of appoint or clover score
medialaborsurnames = ["BRENNAN", "DEANE", "MASON", 'TOOHEY', 'GAUDRON', 'McHUGH', 'KIRBY', 'FRENCH', 'BELL', 'GAGELER',
                 'JAGOT', 'BEECH-JONES','RICH','McTIERNAN','WEBB','JACOBS','MURPHY']
medialnpsurnames = ['DAWSON', 'HAYNE', 'CALLINAN', 'M_GLEESON', 'J_GLEESON', 'HEYDON',
               'CRENNAN', 'KIEFEL', 'NETTLE', 'GORDON', 'EDELMAN', 'STEWARD','STARKE','DIXON','LATHAM',
               'WILLAMS', 'FULLAGAR', 'KITTO', 'TAYLOR', 'MENZIES', 'WINDEYER', 'OWEN', 'BARWICK', 'WALSH',
               'GIBBS', 'STEPHEN','AICKIN','WILSON','GUMMOW','KEANE']
laborsurnames = ["MASON", 'TOOHEY', 'GAUDRON', 'McHUGH', 'GUMMOW', 'KIRBY', 'FRENCH', 'BELL', 'GAGELER',
                 'KEANE', 'JAGOT', 'BEECH-JONES','RICH','McTIERNAN','WEBB','JACOBS','MURPHY']
lnpsurnames = ["BRENNAN", "DEANE", 'DAWSON', 'HAYNE', 'CALLINAN', 'M_GLEESON', 'J_GLEESON', 'HEYDON',
               'CRENNAN', 'KIEFEL', 'NETTLE', 'GORDON', 'EDELMAN', 'STEWARD','STARKE','DIXON','LATHAM',
               'WILLAMS', 'FULLAGAR', 'KITTO', 'TAYLOR', 'MENZIES', 'WINDEYER', 'OWEN', 'BARWICK', 'WALSH',
               'GIBBS', 'STEPHEN','AICKIN','WILSON']
surnames = lnpsurnames + laborsurnames

def process_judgment_files_multiprocess(args):
    judgments_folder, surname = args
    judgment_data = []
    judge_folder = os.path.join(judgments_folder, surname)
    #chnagge based on type of party classification desired appointment is the default
    direction = 1 if surname in laborsurnames else 0
    #direction = 1 if surname in medialaborsurnames else 0
    case_files = {}
    for root, dirs, files in os.walk(judge_folder):
        for filename in files:
            filename_parts = filename.split('_')
            if len(filename_parts) >= 4:
                case_key = filename_parts[0] + "_" + filename_parts[2]
                file_path = os.path.join(root, filename)
                if case_key not in case_files:
                    case_files[case_key] = []
                case_files[case_key].append(file_path)

    for case_key, files in case_files.items():
        year = int(case_key.split('_')[0])
        case_number = int(case_key.split('_')[1])
        concatenated_text = ""
        for file_path in sorted(files):
            with open(file_path, 'r', encoding='utf-8') as file:
                judgment_text = file.read()
                concatenated_text += " " + judgment_text

        if concatenated_text:
            tokens, attention_masks = process_judgment(concatenated_text, year)
            if tokens:
                for i in range(len(tokens)):
                    judgment_data.append((tokens[i], attention_masks[i], direction, case_number, year, [surname]))

    return judgment_data


def process_all_judgments(judgments_folder, surnames):
    all_transcript_data = []
    num_processes = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(num_processes)
    results = pool.map(process_judgment_files_multiprocess, [(judgments_folder, surname) for surname in surnames])
    for result in results:
        all_transcript_data.extend(result)
    pool.close()
    pool.join()



    with open('transcript_datasliding.json', 'w') as f:
        json.dump(all_transcript_data, f)
    print(f"Total instances: {len(all_transcript_data)}")
    return all_transcript_data

if __name__ == '__main__':
    # Define folder paths
    judgments_folder = 'HCATrans_sections'
    # Process judgment files
    process_all_judgments(judgments_folder, surnames)
