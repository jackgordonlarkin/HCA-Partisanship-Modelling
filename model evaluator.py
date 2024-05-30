import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoConfig,BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
import json
from collections import defaultdict
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import csv
from collections import OrderedDict
import random
from itertools import chain
# Define dictionaries to store liberal probabilities by justice and year
liberal_prob_justice_year_best = {}  # Stores probabilities by justice and year
liberal_prob_overall_best = []  # Stores overall probabilities
liberal_prob_justice_best = defaultdict(list)  # Stores probabilities by justice
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')  # Adjust the range as needed
year_tokens = [f'[YEAR_{year}]' for year in range(1900, 2025)]
special_tokens_dict = {'additional_special_tokens': year_tokens}
tokenizer.add_special_tokens(special_tokens_dict)


# Define a custom dataset class
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
laborsurnames = ["MASON", 'TOOHEY', 'GAUDRON', 'McHUGH', 'GUMMOW', 'KIRBY', 'FRENCH', 'BELL', 'GAGELER',
                 'KEANE', 'JAGOT', 'BEECH-JONES','RICH','McTIERNAN','WEBB','JACOBS','MURPHY']
lnpsurnames = ["BRENNAN", "DEANE", 'DAWSON', 'HAYNE', 'CALLINAN', 'M_GLEESON', 'J_GLEESON', 'HEYDON',
               'CRENNAN', 'KIEFEL', 'NETTLE', 'GORDON', 'EDELMAN', 'STEWARD','STARKE','DIXON','LATHAM',
               'WILLAMS', 'FULLAGAR', 'KITTO', 'TAYLOR', 'MENZIES', 'WINDEYER', 'OWEN', 'BARWICK', 'WALSH',
               'GIBBS', 'STEPHEN','AICKIN','WILSON']
surnames = lnpsurnames + laborsurnames
class JudgmentDataset(Dataset):
    def __init__(self, input_ids, attention_masks, years, directions, transcript_numbers, judge_names,tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = [self.prepend_year_token(input_id, year) for input_id, year in zip(input_ids, years)]
        self.attention_masks = [self.adjust_attention_mask(attention_mask) for attention_mask in attention_masks]
        self.directions = torch.tensor(directions, dtype=torch.long)  # Convert directions to Long tensors
        self.transcript_numbers = torch.tensor(transcript_numbers,dtype=torch.long)  # Convert transcript_numbers to Long tensors
        self.judge_names = judge_names
        self.years = years


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        year = self.years[idx]
        direction = self.directions[idx]
        transcript_number = self.transcript_numbers[idx]
        judge_name = self.judge_names[idx]

        return input_id, attention_mask, direction, transcript_number, judge_name, year
    def prepend_year_token(self, input_id, year):
        year_token = self.tokenizer.convert_tokens_to_ids(f'[YEAR_{year}]')
        return torch.tensor([year_token] + input_id, dtype=torch.long)
    def adjust_attention_mask(self, attention_mask):
        return torch.tensor([1] + attention_mask, dtype=torch.long)
class JudgmentDatasetNoYear(Dataset):
    def __init__(self, input_ids, attention_masks, years, directions, transcript_numbers, judge_names,tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.directions = torch.tensor(directions, dtype=torch.long)  # Convert directions to Long tensors
        self.transcript_numbers = torch.tensor(transcript_numbers,dtype=torch.long)  # Convert transcript_numbers to Long tensors
        self.judge_names = judge_names
        self.years = years


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        year = self.years[idx]
        direction = self.directions[idx]
        transcript_number = self.transcript_numbers[idx]
        judge_name = self.judge_names[idx]

        return input_id, attention_mask, direction, transcript_number, judge_name, year

def custom_collate_fn(batch):
    input_ids, attention_masks, directions, transcript_numbers, judge_names, years = zip(*batch)

    # Convert input_ids, attention_masks, directions, transcript_numbers, and years to tensors
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    directions = torch.stack(directions)
    transcript_numbers = torch.stack(transcript_numbers)
    years = torch.tensor(years, dtype=torch.long)

    return input_ids, attention_masks, directions, transcript_numbers, judge_names, years


def modulo_split(dataset, counter, modulo):
    train_indices = []
    val_indices = []
    for i, (input_id, _, _, transcript_number, _, _) in enumerate(dataset):
        if transcript_number % modulo == counter % modulo:
            val_indices.append(i)
        else:
            train_indices.append(i)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


model_type = "nlpaueb/legal-bert-small-uncased"
dropout_prob=0
# Load the model configuration
config = AutoConfig.from_pretrained(model_type)
config.num_labels=1
config.hidden_dropout_prob=dropout_prob
config.attention_probs_dropout_prob=dropout_prob
# Load the model with the updated configuration
model = AutoModelForSequenceClassification.from_pretrained(model_type, config=config)


print(model.config)

with open('transcript_datasliding.json', 'r') as f:
    test_data = json.load(f)


input_ids_test = [item[0] for item in test_data]
attention_masks_test = [item[1] for item in test_data]
directions_test = [int(item[2]) for item in test_data]
transcript_numbers_test = [int(item[3]) for item in test_data]
judge_names_test = [item[5] for item in test_data]
year_test = [int(item[4]) for item in test_data]
print(len(year_test))

# Separate train and test data
test_dataset = JudgmentDatasetNoYear(input_ids_test, attention_masks_test, year_test, directions_test, transcript_numbers_test, judge_names_test,tokenizer)
# Create data loaders
train_batch_size = 16
val_batch_size = 64
test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size,collate_fn=custom_collate_fn)
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=.005)
epochs = 1 # Adjust as needed
patience = 1
best_val_loss = float('inf')
counter = 0
best_model_state_dict = None
best_accuracy = 0.0

# Lists to store training and validation losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f = []
val_f = []
test_f = []
train_f_ci = []
val_f_ci = []
test_f_ci = []
criterion = torch.nn.BCEWithLogitsLoss()
# Lists to store test accuracies
test_accuracies = []
#model.load_state_dict(torch.load('best_model_USA.pth')) have the mode be based on a previous one
# Directory to save the model checkpoints
loadedlength = 0
model.load_state_dict(torch.load('best_model_USA.pth'))
liberal_prob_justice_year = defaultdict(list)  # Stores probabilities by justice and year
liberal_prob_overall = []  # Stores overall probabilities
liberal_prob_justice = defaultdict(list)  # Stores probabilities by justice
test_predictions_all = []
test_labels_all = []
with torch.no_grad():
    for input_ids, attention_masks, directions, transcript_numbers, judge_names, years in tqdm(test_dataloader, desc="Testing"):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        directions = directions.to(device)
        directions = directions.float()
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        test_predictions_all.extend(predictions.cpu().numpy())
        test_labels_all.extend(directions.cpu().numpy())
        for i in range(len(probabilities)):
            for name in judge_names[i]:
                liberal_prob_justice_year[(name, int(years[i]))].append(probabilities[i])
                liberal_prob_overall.append(probabilities[i])
                liberal_prob_justice[name].append(probabilities[i])
    test_f1 = f1_score(test_labels_all, test_predictions_all, average='macro')
    test_accuracy = accuracy_score(test_labels_all, test_predictions_all)
    test_accuracies.append(test_accuracy)
    print(f"Test accuracy: {test_accuracy:.4f}, Test F1 score: {test_f1:.4f}")
    liberal_prob_overall_best=liberal_prob_overall
    liberal_prob_justice_best=liberal_prob_justice
    liberal_prob_justice_year_best=liberal_prob_justice_year

floatedliberal_prob_overall_best = [tensor.item() for tensor in liberal_prob_overall_best]
mean_overall = np.mean(floatedliberal_prob_overall_best)

lower_overall, upper_overall = stats.norm.interval(0.95, loc=mean_overall, scale=stats.sem(floatedliberal_prob_overall_best))
liberal_prob_justice_year_agg = OrderedDict()
sorted_keys = sorted(liberal_prob_justice_year_best.keys(), key=lambda x: x[1])
for key in sorted_keys:
    values=liberal_prob_justice_year_best[key]
    values= [tensor.item() for tensor in values]
    mean = np.mean(values)
    lower, upper = stats.norm.interval(0.95, loc=mean, scale=stats.sem(values))
    liberal_prob_justice_year_agg[key] = (mean, lower, upper)
# Create a DataFrame to store the aggregated data
data = {'Justice': [], 'Year': [], 'Mean_Probability': [], 'Lower_CI': [], 'Upper_CI': []}
print(liberal_prob_justice_year_agg.keys())
# Extract data from the dictionary and populate the DataFrame
for (justice, year), (mean, lower, upper) in liberal_prob_justice_year_agg.items():
    data['Justice'].append(justice)
    data['Year'].append(year)
    data['Mean_Probability'].append(mean)
    data['Lower_CI'].append(lower)
    data['Upper_CI'].append(upper)
# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('liberal_probabilities_by_year_by_justiceTransJustUS.csv', index=False)

earliest_years = df.groupby('Justice')['Year'].min().reset_index()
earliest_years.columns = ['Justice', 'Earliest_Year']

# Merge the earliest year back into the original DataFrame
df = df.merge(earliest_years, on='Justice')

# Sort by Earliest Year, Justice, and Year
df_sorted = df.sort_values(by=['Earliest_Year', 'Justice', 'Year'])
print(df_sorted)
# Define line styles
line_styles = ['-', '--', ':']
colors = plt.cm.tab20.colors  # Using tab10 colormap for 10 distinct colors
print(len(colors))
# Plot the score of each justice each year over time with different line types
fig, ax = plt.subplots(figsize=(14, 10))
counter = 0
trainnames=[]
testnames=np.unique(list(chain.from_iterable(judge_names_test)))
print(testnames)
for i, (justice, group) in enumerate(df_sorted.groupby('Justice', sort=False)):
    print(justice)
    line_style = line_styles[0]  # Cycle through line styles
    if counter%3 == 2:
        line_style = line_styles[1]
    if counter%3 == 1:
        line_style = line_styles[2]
    color = colors[i % len(colors)]  # Cycle through colors
    if justice in trainnames:
        ax.plot(group['Year'], group['Mean_Probability'], label=f'{justice}*', linestyle=line_style)
    else:
        ax.plot(group['Year'], group['Mean_Probability'], label=justice, linestyle=line_style)
    counter += 1
# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Mean Liberal Probability')
ax.set_title('Mean Liberal Probability by Year by Justice (Best Model)')
# Move the legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)  # Adjust ncol to fit all labels properly
# Adjust the layout to make space for the legend
plt.subplots_adjust(bottom=0.3)
# Save the plot
plt.savefig('liberal_probability_by_year_by_justice_plotTransJustUS.png')
# Show the plot
plt.show()

liberal_prob_justice_agg = {}
for key, values in liberal_prob_justice_best.items():
    floatedvalues= [tensor.item() for tensor in values]
    mean = np.mean(floatedvalues)
    lower, upper = stats.norm.interval(0.95, loc=mean, scale=stats.sem(floatedvalues))
    liberal_prob_justice_agg[key] = (mean, lower, upper)
sorted_liberal_prob_justice_agg = {k: v for k, v in sorted(liberal_prob_justice_agg.items(), key=lambda item: item[1][0])}

# Extract the data for plotting
justices = list(sorted_liberal_prob_justice_agg.keys())
means = [value[0] for value in sorted_liberal_prob_justice_agg.values()]
lower_errors = [value[0] - value[1] for value in sorted_liberal_prob_justice_agg.values()]
upper_errors = [value[2] - value[0] for value in sorted_liberal_prob_justice_agg.values()]
justices.append('Overall')
means.append(mean_overall)
lower_errors.append(lower_overall)
upper_errors.append(upper_overall)
# Plot the bar chart with error bars
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red' if justice in laborsurnames else 'blue' for justice in justices]
bars = ax.barh(justices, means, xerr=[lower_errors, upper_errors], capsize=5, color=colors)
# Add labels and title
ax.set_xlabel('Mean Liberal Probability')
ax.set_ylabel('Justice')
ax.set_title('Aggregated Liberal Probabilities by Justice (95% CI)')

plt.savefig('liberal_probability_by_justice_plotTransJustUS.png')
plt.show()
csv_filename = "aggregated_liberal_probabilitiesTransJustUS.csv"

# Write the data to the CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Justice', 'Mean Liberal Probability', 'Lower Error', 'Upper Error']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for justice, (mean, lower, upper) in sorted_liberal_prob_justice_agg.items():
        writer.writerow({'Justice': justice, 'Mean Liberal Probability': mean, 'Lower Error': lower, 'Upper Error': upper})
