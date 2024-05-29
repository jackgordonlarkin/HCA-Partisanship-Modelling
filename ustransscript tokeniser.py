import os
import pandas as pd
import numpy as np
import multiprocessing
from bs4 import BeautifulSoup
import re
from bs4 import Tag
from transformers import BertTokenizer, AutoTokenizer
import spacy
import json
import torch
from torch.utils.data import Dataset

# Load spacy
nlp = spacy.load("en_core_web_sm")

# Load tokenizer
years = list(range(1900, 2025))  # Adjust the range as needed
year_tags = [f"[YEAR_{year}]" for year in years]
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
tokenizer.add_special_tokens({"additional_special_tokens": year_tags})

# Define party affiliation mapping
party_mapping = {"R": 0, "D": 1}
map_justice_to_party = {
    'j__sonia_sotomayor': 'D', 'j__elena_kagan': 'D', 'j__john_g_roberts_jr': 'R',
    'j__neil_gorsuch': 'R', 'j__clarence_thomas': 'R', 'j__antonin_scalia': 'R',
    'j__brett_m_kavanaugh': 'R', 'j__earl_warren': 'R', 'j__charles_e_whittaker': 'R',
    'j__lewis_f_powell_jr': 'R', 'j__harold_burton': 'D', 'j__sherman_minton': 'D',
    'j__abe_fortas': 'D', 'j__hugo_l_black': 'D', 'j__potter_stewart': 'R',
    'j__warren_e_burger': 'R', 'j__harry_a_blackmun': 'R', 'j__arthur_j_goldberg': 'D',
    'j__samuel_a_alito_jr': 'R', 'j__john_m_harlan2': 'R', 'j__anthony_m_kennedy': 'R',
    'j__ruth_bader_ginsburg': 'D', 'j__william_j_brennan_jr': 'R', 'j__john_m_harlan': 'R',
    'j__david_h_souter': 'R', 'j__william_o_douglas': 'D', 'j__stephen_g_breyer': 'D',
    'j__john_paul_stevens': 'R', 'j__thurgood_marshall': 'D', 'j__felix_frankfurter': 'D',
    'j__william_h_rehnquist': 'R', 'j__byron_r_white': 'D', 'j__tom_c_clark': 'D',
    'j__sandra_day_oconnor': 'R', 'j__stanley_reed': 'D'
}


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
    judgment_text = re.sub(r'\s+', ' ', judgment_text)
    year_token = tokenizer.convert_tokens_to_ids(f"[YEAR_{year}]")
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


def process_statements(data, party_mapping):
    all_data = []
    for case in data:
        case_id = case.get('case_id', '')
        year = case_id.split('_')[0] if case_id else None
        transcript_number = 10
        if len(case_id.split('_')) > 1:
            transcript_part = case_id.split('_')[1]
            if '-orig' in transcript_part:
                transcript_part = transcript_part.replace('-orig', '')
            if '-MISC' in transcript_part:
                transcript_part = transcript_part.replace('-MISC', '')
            transcript_number = transcript_part.split('-')[-1]
        speaker_texts = {}
        statements = case.get('convos', [])
        for statement_list in statements:
            for statement in statement_list:
                speaker = statement.get('speaker_id', '')
                text = statement.get('text', '')
                if speaker in map_justice_to_party:
                    if speaker not in speaker_texts:
                        speaker_texts[speaker] = ""
                    speaker_texts[speaker] += " " + text
        for speaker, text in speaker_texts.items():
            if speaker in map_justice_to_party:
                direction = party_mapping[map_justice_to_party[speaker]]
                tokens, attention_masks = process_judgment(text, year)
                if tokens:
                    for i in range(len(tokens)):
                        all_data.append(
                            (tokens[i], attention_masks[i], direction, int(transcript_number), int(year), [speaker]))

    return all_data

def main(json_file_path):
    with open(json_file_path, 'r') as file:
        raw_data = file.read().strip()

    data = []
    for obj in raw_data.split('\n'):
        data.append(json.loads(obj))

    all_transcript_data = process_statements(data, party_mapping)

    with open('transcript_dataslidingUS.json', 'w') as f:
        json.dump(convert_to_lists(all_transcript_data), f)

    print(f"Processed {len(all_transcript_data)} chunks of data.")

if __name__ == '__main__':
    json_file_path = 'constructed_cases_filter_different_years_mask_names.jsonl'
    main(json_file_path)