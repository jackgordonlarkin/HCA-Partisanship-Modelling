import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoConfig,BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoModel,  LongformerForSequenceClassification
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
from torch.cuda.amp import GradScaler, autocast
from sklearn.utils.class_weight import compute_class_weight


# Define dictionaries to store liberal probabilities by justice and year
liberal_prob_justice_year_best = {}  # Stores probabilities by justice and year
liberal_prob_overall_best = []  # Stores overall probabilities
liberal_prob_justice_best = defaultdict(list)  # Stores probabilities by justice
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')  # Adjust the range as needed
year_tokens = [f'[YEAR_{year}]' for year in range(1948, 2025)]
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
        return torch.tensor(input_id, dtype=torch.long)
    def adjust_attention_mask(self, attention_mask):
        return torch.tensor([1] + attention_mask, dtype=torch.long)
        return torch.tensor(attention_mask, dtype=torch.long)
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
dropout_prob=.3
# Load the model configuration
config = AutoConfig.from_pretrained(model_type)
config.num_labels=1 #it's a binairy choice
config.hidden_dropout_prob=dropout_prob # forget hidden layhers
config.attention_probs_dropout_prob=dropout_prob # sometime signore attention masks
# Load the model with the updated configuration
model =  AutoModelForSequenceClassification.from_pretrained(model_type, config=config)
print(model.config)


#triued adding in freedom data
with open('transcript_dataslidingUS.json', 'r') as f:
    us_data=json.load(f)

train_cases=[]
#needed for when doing the original aprtisan grouping models
data=[]
val_data=[]
test_data=[]


#training us data split

for i in us_data:
    if i[4] < 2000:
        data.append(i)
    elif i[4] > 2009:
        test_data.append(i)
    else:
        val_data.append(i)
print(data[0])
input_ids_train = [item[0] for item in data]
attention_masks_train = [item[1] for item in data]
directions_train = [int(item[2]) for item in data]
transcript_numbers_train = [int(item[3]) for item in data]
judge_names_train = [list(item[5]) for item in data]
year_train = [int(item[4]) for item in data]
print(len(year_train))
print(np.unique(directions_train))
print(len([i for i in directions_train if i==1]))
print(len(year_train))
input_ids_test = [item[0] for item in test_data]
attention_masks_test = [item[1] for item in test_data]
directions_test = [int(item[2]) for item in test_data]
transcript_numbers_test = [int(item[3]) for item in test_data]
judge_names_test = [item[5] for item in test_data]
year_test = [int(item[4]) for item in test_data]
print(len(year_test))

input_ids_val = [i[0] for i in val_data]
attention_masks_val = [i[1] for i in val_data]
directions_val = [int(i[2]) for i in val_data]
transcript_numbers_val = [int(i[3]) for i in val_data]
judge_names_val = [i[5] for i in val_data]
year_val = [int(i[4]) for i in val_data]

# Separate train and test data
dataset = JudgmentDatasetNoYear(input_ids_train, attention_masks_train, year_train, directions_train, transcript_numbers_train, judge_names_train,tokenizer)
test_dataset = JudgmentDatasetNoYear(input_ids_test, attention_masks_test, year_test, directions_test, transcript_numbers_test, judge_names_test,tokenizer)
val_dataset = JudgmentDatasetNoYear(input_ids_val,attention_masks_val,year_val,directions_val,transcript_numbers_val,judge_names_val,tokenizer)
#ConsiderYears
'''
dataset = JudgmentDataset(input_ids_train, attention_masks_train, year_train, directions_train, transcript_numbers_train, judge_names_train,tokenizer)
test_dataset = JudgmentDataset(input_ids_test, attention_masks_test, year_test, directions_test, transcript_numbers_test, judge_names_test,tokenizer)
val_dataset = JudgmentDataset(input_ids_val,attention_masks_val,year_val,directions_val,transcript_numbers_val,judge_names_val,tokenizer)
'''
# Create data loaders
train_batch_size = 16
val_batch_size = 64
test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size,collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,collate_fn=custom_collate_fn)
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=.005)
epochs =30 # Adjust as needed
patience = 5
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
checkpoint_dir = "model_checkpoints_USA"
loadedlength = 0

# Create the directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if re.match(r"model_epoch_\d+\.pth", f)]
print(existing_checkpoints)
if existing_checkpoints:
    latest_checkpoint = max(existing_checkpoints, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    print(latest_checkpoint)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint)))
    checkpoint_info = torch.load(os.path.join(checkpoint_dir, latest_checkpoint + ".info"))
    best_model_state_dict = checkpoint_info["best_model_state_dict"]
    best_accuracy = checkpoint_info["best_accuracy"]
    test_accuracies = checkpoint_info["test_accuracies"]
    train_losses = checkpoint_info["train_losses"]
    val_losses = checkpoint_info["val_losses"]
    val_accuracies = checkpoint_info["val_accuracies"]
    train_accuracies = checkpoint_info["train_accuracies"]
    counter = checkpoint_info["counter"]
    liberal_prob_justice_year_best = checkpoint_info["liberal_prob_justice_year_best"]  # Stores probabilities by justice and year
    liberal_prob_overall_best = checkpoint_info["liberal_prob_overall_best"]
    liberal_prob_justice_best = checkpoint_info["liberal_prob_justice_best"]
    train_f = checkpoint_info["train_f"]
    test_f = checkpoint_info["test_f"]
    val_f = checkpoint_info["val_f"]
    val_f_ci = checkpoint_info["val_f_ci"]
    train_f_ci = checkpoint_info["train_f_ci"]
    test_f_ci = checkpoint_info["test_f_ci"]
    loadedlength = len(existing_checkpoints) // 2
    print(f"Loaded checkpoint: {latest_checkpoint}")
for epoch in range(loadedlength, epochs):
    liberal_prob_justice_year = defaultdict(list)  # Stores probabilities by justice and year
    liberal_prob_overall = []  # Stores overall probabilities
    liberal_prob_justice = defaultdict(list)  # Stores probabilities by justice
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0
    dataset, _ = torch.utils.data.random_split(dataset, [len(dataset), 0])
    train_dataset, _ = modulo_split(dataset, len(test_f), 9)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,collate_fn=custom_collate_fn, pin_memory=True)
    train_predictions_all = []
    train_labels_all = []
    for input_ids, attention_masks, directions, transcript_numbers, judge_names, years in tqdm(train_dataloader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        directions = directions.to(device)
        directions = directions.float()
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        loss = criterion(logits.squeeze(-1), directions)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        train_predictions_all.extend(predictions.cpu().numpy())
        train_labels_all.extend(directions.cpu().numpy())
        for i in range(len(probabilities)):
            for name in judge_names[i]:
                liberal_prob_justice_year[(name, int(years[i]))].append(probabilities[i])
                liberal_prob_overall.append(probabilities[i])
                liberal_prob_justice[name].append(probabilities[i])
    train_loss = total_loss / len(train_dataloader)
    train_accuracy = accuracy_score(train_labels_all, train_predictions_all)
    train_f1 = f1_score(train_labels_all, train_predictions_all, average='macro')
    train_f1_ci_lower, train_f1_ci_upper = stats.norm.interval(0.95, loc=train_f1, scale=np.sqrt((train_f1 * (1 - train_f1)) / len(train_labels_all)))
    train_f_ci.append((train_f1_ci_lower, train_f1_ci_upper))
    train_f.append(train_f1)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Train F1 score: {train_f1:.4f}")

    # Validation loop
    model.eval()
    total_loss = 0
    val_predictions_all = []
    val_labels_all = []
    with (torch.no_grad()):
        for input_ids, attention_masks, directions, transcript_numbers, judge_names, years in tqdm(val_dataloader, desc="Validation"):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            directions = directions.to(device)
            directions = directions.float()
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            loss = criterion(logits.squeeze(-1), directions)
            total_loss += loss.item()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            val_predictions_all.extend(predictions.cpu().numpy())
            val_labels_all.extend(directions.cpu().numpy())
            for i in range(len(probabilities)):
                for name in judge_names[i]:
                    liberal_prob_justice_year[(name, int(years[i]))].append(probabilities[i])
                    liberal_prob_overall.append(probabilities[i])
                    liberal_prob_justice[name].append(probabilities[i])
    val_loss = total_loss / len(val_dataloader)
    val_accuracy = accuracy_score(val_labels_all, val_predictions_all)
    val_f1 = f1_score(val_labels_all, val_predictions_all, average='macro')
    val_f1_ci_lower, val_f1_ci_upper = stats.norm.interval(0.95, loc=val_f1,scale=np.sqrt((val_f1 * (1 - val_f1)) / len(val_labels_all)))
    val_f_ci.append((val_f1_ci_lower, val_f1_ci_upper))
    val_f.append(val_f1)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}, Val F1 score: {val_f1:.4f}")
    # Testing
    model.eval()
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
    test_f1_ci_lower, test_f1_ci_upper = stats.norm.interval(0.95, loc=test_f1, scale=np.sqrt(
        (test_f1 * (1 - test_f1)) / len(test_labels_all)))
    test_f_ci.append((test_f1_ci_lower, test_f1_ci_upper))
    test_f.append(test_f1)
    test_accuracies.append(test_accuracy)
    print(f"Test accuracy: {test_accuracy:.4f}, Test F1 score: {test_f1:.4f}")
    counter += 1
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
    if val_f1 >= max(val_f):
        counter = 0
        best_model_state_dict = model.state_dict()
        liberal_prob_overall_best = liberal_prob_overall
        liberal_prob_justice_best = liberal_prob_justice
        liberal_prob_justice_year_best = liberal_prob_justice_year
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_info = {
            "best_model_state_dict": best_model_state_dict,
            "best_accuracy": best_accuracy,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "test_accuracies": test_accuracies,
            "counter": counter,
            "liberal_prob_overall_best": liberal_prob_overall_best,
            "liberal_prob_justice_best": liberal_prob_justice_best,
            "liberal_prob_justice_year_best": liberal_prob_justice_year_best,
            "train_f": train_f,
            "test_f": test_f,
            "val_f": val_f,
            "train_accuracies": train_accuracies,
            "train_f_ci": train_f_ci,
            "val_f_ci": val_f_ci,
            "test_f_ci": test_f_ci
        }
        torch.save(checkpoint_info, checkpoint_path + ".info")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_info = {
            "best_model_state_dict": best_model_state_dict,
            "best_accuracy": best_accuracy,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "test_accuracies": test_accuracies,
            "counter": counter,
            "liberal_prob_overall_best": liberal_prob_overall_best,
            "liberal_prob_justice_best": liberal_prob_justice_best,
            "liberal_prob_justice_year_best": liberal_prob_justice_year_best,
            "train_f": train_f,
            "test_f": test_f,
            "val_f": val_f,
            "train_accuracies": train_accuracies,
            "train_f_ci": train_f_ci,
            "val_f_ci": val_f_ci,
            "test_f_ci": test_f_ci
            }
        torch.save(checkpoint_info, checkpoint_path + ".info")
        if counter >= patience:
            print("Early stopping triggered.")
            torch.save(model.state_dict(), checkpoint_path)
            break
# Save the best model
torch.save(best_model_state_dict, "best_model_USA.pth")
# Plot training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses')
plt.savefig('lossesUSA.png')
plt.show()
# Plot validation accuracy
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation & Test Accuracy')
plt.legend()
plt.savefig('accuracyUSA.png')
plt.show()

# Assuming train_f, val_f, test_f, train_f_ci, val_f_ci, test_f_ci, and epochs are defined
# Ensure test_f, test_f_ci_lower, and test_f_ci_upper have the same length

# Convert confidence intervals from tuples to separate lists for lower and upper bounds
train_f_ci_lower = [ci[0] for ci in train_f_ci]
train_f_ci_upper = [ci[1] for ci in train_f_ci]
val_f_ci_lower = [ci[0] for ci in val_f_ci]
val_f_ci_upper = [ci[1] for ci in val_f_ci]
test_f_ci_lower = [ci[0] for ci in test_f_ci]
test_f_ci_upper = [ci[1] for ci in test_f_ci]

# Plot the lines for training, validation, and test data
plt.plot(train_f, label='Train F')
plt.plot(val_f, label='Validation F')
plt.plot(test_f, label='Test F')

# Fill the confidence intervals
plt.fill_between(range(len(train_f)), train_f_ci_lower, train_f_ci_upper, color='blue', alpha=0.2)
plt.fill_between(range(len(val_f)), val_f_ci_lower, val_f_ci_upper, color='orange', alpha=0.2)
plt.fill_between(range(len(test_f)), test_f_ci_lower, test_f_ci_upper, color='green', alpha=0.2)

plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Test F1 Scores with Confidence Intervals')
plt.savefig('fscoreUSA.png')
plt.show()
metrics_dict = {
    'Epoch': list(range(1, len(train_losses) + 1)),
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Validation Accuracy': val_accuracies,
    'Test Accuracy': test_accuracies,
    'Train F-score': train_f,
    'Train F-CI': train_f_ci,
    'Validation F-score': val_f,
    'Validation F-CI': val_f_ci,
    'Test F-score': test_f,
    'Test F-CI': test_f_ci
}

# Convert the dictionary to a pandas DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Save the DataFrame to a CSV file
metrics_df.to_csv('metricsUSA.csv', index=False)
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
df.to_csv('liberal_probabilities_by_year_by_justiceUSA.csv', index=False)

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
trainnames=np.unique(list(chain.from_iterable(judge_names_train)))
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
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=7)  # Adjust ncol to fit all labels properly
# Adjust the layout to make space for the legend
plt.subplots_adjust(bottom=0.3)
# Save the plot
plt.savefig('liberal_probability_by_year_by_justice_plotUSA.png')
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

plt.savefig('liberal_probability_by_justice_plotUSA.png')
plt.show()
csv_filename = "aggregated_liberal_probabilitiesUSA.csv"

# Write the data to the CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Justice', 'Mean Liberal Probability', 'Lower Error', 'Upper Error']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for justice, (mean, lower, upper) in sorted_liberal_prob_justice_agg.items():
        writer.writerow({'Justice': justice, 'Mean Liberal Probability': mean, 'Lower Error': lower, 'Upper Error': upper})
