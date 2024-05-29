import os
import requests
from bs4 import BeautifulSoup
import re

def extract_text_after_surname(html_content, surnames):
    soup = BeautifulSoup(html_content, 'html.parser')
    hr_tags = soup.find_all('hr')
    if len(hr_tags) >= 2:
        # Get the second last <hr> tag
        second_last_hr_tag = hr_tags[1]

        # Remove everything after the second last <hr> tag
        for sibling in second_last_hr_tag.find_next_siblings():
            sibling.extract()
    text = soup.get_text()  # Extract text content from HTML

    # Find the indices of the first occurrence of either "CATCHWORDS" or "DECISION"
    catchwords_index = text.find("CATCHWORDS")
    decision_index = text.find("DECISION")
    # Choose the earliest index that is not -1
    try:
        start_index = min(index for index in [catchwords_index, decision_index] if index != -1)
    except:
        start_index = 0
    # Get the text after the chosen start index
    newtext = text[start_index:]

    # Find the end of the main content
    footnotes_index = newtext.rfind("Footnotes:")
    austlii_index = newtext.rfind("AustLII:")
    try:
        end_index = min(index for index in [footnotes_index, austlii_index] if index != -1)
        newtext = newtext[:end_index]
    except:
        pass
    # Find the minimum index of any surname after the start index
    min_surname_index = float('inf')  # Initialize with positive infinity
    for surname in surnames:
        surname_index = newtext.find(surname)  # Case-sensitive search
        if surname_index != -1 and surname_index < min_surname_index:
            min_surname_index = surname_index

    # Return the text after the minimum surname index
    if min_surname_index != float('inf'):
        return newtext[min_surname_index:]
    else:
        min_surname_index = float('inf')  # Initialize with positive infinity
        for surname in surnames:
            surname_index = text.find(surname)  # Case-sensitive search
            if surname_index != -1 and surname_index < min_surname_index:
                min_surname_index = surname_index
        if min_surname_index != float('inf'):
            return text[min_surname_index:]
        else:
            return ''
def get_matching_surname(text, surnames):
    for surname in surnames:
        if text.strip().startswith(surname):
            return surname
    return None

def judgement_splitter(text, surnames, year, judgment_number):
    judgments = []
    if text == '':
        return judgments

    judgment_start = 0
    for idx in range(len(text)):
        end_idx=-1
        if text[idx:].strip().startswith(tuple(surname + " " for surname in surnames)):
            judgment_text = text[judgment_start:idx].lstrip().rstrip()
            number_match = re.search(r'\d', judgment_text)
            if number_match:
                number_idx = number_match.start()
            else:
                number_idx = -1
            if ('.' in judgment_text or number_idx !=-1) and len(judgment_text)>30:
                fullstop_idx = judgment_text.rfind('. ')
                CJ_idx = -1
                if 'C.J' in judgment_text:
                    CJ_idx=judgment_text.find('C.J')
                # Find the next full stop after the name
                end_indices = [idx for idx in [fullstop_idx,number_idx] if idx != -1]
                if len(end_indices)>0:
                    end_idx = min(end_indices)
                if end_idx != -1 and CJ_idx < fullstop_idx-4:
                    startname = get_matching_surname(text[idx:], surnames)
                    filtered_surnames = surnames.copy()
                    filtered_surnames.remove(startname)
                    keyword_near = any(keyword in text[max(idx-20,0):idx+20] for keyword in
                                        [" J","JJ","CJ","Justice","J.","\nJ",
                                         "C.J"])
                    surname_near = any(
                        surname in text[max(idx - 20, 0):idx + 20] for surname in
                        filtered_surnames)
                    if keyword_near or surname_near:
                        judgment_surnames = [surname for surname in surnames if
                                             surname in judgment_text[:end_idx]]
                        judgments.append((judgment_text, judgment_surnames))
                        judgment_start = idx
    # Append the last judgment
    judgment_text = text[judgment_start:].lstrip().rstrip()
    fullstop_idx = judgment_text.find('.')
    number_match = re.search(r'\d', judgment_text)
    if number_match:
        number_idx = number_match.start()
    else:
        number_idx = -1
    # Find the next full stop after the name
    end_indices = [idx for idx in [fullstop_idx, number_idx] if idx != -1]
    end_idx = min(end_indices) if end_indices else -1
    if end_idx!=-1:
        judgment_surnames = [surname for surname in surnames if surname in judgment_text[:end_idx]]
    else:
        judgment_surnames = [surname for surname in surnames if
                             surname in judgment_text[:200]]
    judgments.append((judgment_text, judgment_surnames))
    # Create a folder to save judgments
    folder_name = ("judgmentsHCA1991")
    os.makedirs(folder_name, exist_ok=True)

    # Save judgments as files
    for i, (judgment_text, judgment_surnames) in enumerate(judgments, start=1):
        filename = f"{year}_HCA_{judgment_number}_{','.join(judgment_surnames)}.txt"
        file_path = os.path.join(folder_name, filename)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(judgment_text)
            print(f"Judgment saved as {file_path}")

    return judgments

def scrape_judgments(year_range, max_judgment_number, surnames):
    for year in year_range:
        for judgment_number in range(1, max_judgment_number + 1):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
            }
            url = f"https://classic.austlii.edu.au/au/cases/cth/HCA/{year}/{judgment_number}.html"
            response=''
            try:
                response = requests.get(url, headers=headers)
            except:
                break
            if response.status_code == 200:
                html_content = response.text
                text_after_surname = extract_text_after_surname(html_content, surnames)
                if text_after_surname:
                    # Save the extracted text content
                    judgement_splitter(text_after_surname,surnames,year,judgment_number)
                else:
                    print(f"No relevant section found after the surname for {year}/HCA/{judgment_number}")
            else:
                print(f"Failed to download HTML for {year}/HCA/{judgment_number}. Status code: {response.status_code}")
                # Skip to the next year if the webpage is not found
                break
# Example usage
start_year = 1948
end_year = 2024
year_range = range(start_year, end_year + 1)
max_judgment_number = 100  # Adjust this number based on the maximum expected judgment number for each year
surnames = [
    'McTIERNAN', 'LATHAM', 'WILLIAMS', 'WEBB', 'FULLAGAR', 'KITTO', 'TAYLOR', 'MENZIES', 'WINDEYER',
    'OWEN', 'BARWICK', 'WALSH', 'GIBBS', 'STEPHEN', 'JACOBS', 'MURPHY', 'AICKIN', 'WILSON',
    'BRENNAN', 'DEANE', 'DAWSON', 'TOOHEY', 'GAUDRON', 'McHUGH', 'GUMMOW', 'KIRBY', 'HAYNE',
    'CALLINAN', 'HEYDON', 'CRENNAN', 'KIEFEL', 'FRENCH', 'BELL', 'GAGELER', 'KEANE', 'NETTLE',
    'GORDON', 'EDELMAN', 'STEWARD', 'JAGOT', 'BEECH-JONES', 'DIXON','RICH','STARKE','MASON'
]
# Scrape judgments
scrape_judgments(year_range, max_judgment_number, surnames)
