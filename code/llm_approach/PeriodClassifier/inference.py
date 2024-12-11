from datetime import datetime
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import wandb
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from periodclassifier import PeriodClassifier, PeriodClassifierConfig
import sys
sys.path.append('../.')
from preprocessor import filter_tweet
from transformers import LongformerForSequenceClassification

# Define constants
DATA_DIR = '/Data/tlh45/'
os.environ["HF_HOME"] = DATA_DIR
wandb_project = "ml-data-challenge-24"
wandb_entity = "ml-data-challenge-24"
wandb_run_id = "39f501y7"
checkpoint_name = None  # "checkpoint-1250" 
best_model_dir = "MLDataChallenge2024/BestModel"  # Folder for the best model
MODEL = "allenai/longformer-base-4096"  # or 'vinai/bertweet-base'
MAX_TWEETS = 200  # Maximum number of tweets per period for Bertweet
BATCH_SIZE = 2 

if MODEL == 'vinai/bertweet-base':
    MAX_LENGTH = 128
elif MODEL == 'allenai/longformer-base-4096':
    MAX_LENGTH = 4096
else:
    raise ValueError("Unknown MODEL specified.")

# Define mode: "best_model" or "checkpoint"
mode = "best_model"  
if mode == "best_model":
    download_dir = os.path.join(DATA_DIR, f"BestModel-{wandb_run_id}")
    model_folder = best_model_dir
elif mode == "checkpoint":
    if not checkpoint_name:
        raise ValueError("checkpoint_name must be specified when mode is 'checkpoint'.")
    download_dir = os.path.join(DATA_DIR, f"Checkpoints-{wandb_run_id}")
    model_folder = checkpoint_name
else:
    raise ValueError("Invalid mode. Use 'best_model' or 'checkpoint'.")

checkpoint_dir = os.path.join(download_dir, model_folder)

print(f"Using {mode} from W&B run {wandb_run_id}.")

# Check if the model or checkpoint is already downloaded from Weights & Biases
if os.path.exists(checkpoint_dir):
    print(f"Model or checkpoint already downloaded to {checkpoint_dir}")
else:
    print(f"Downloading {mode} to {checkpoint_dir}")

    api = wandb.Api()
    run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
    files = run.files()

    os.makedirs(download_dir, exist_ok=True)

    if mode == "best_model":
        best_model_files = [
            file for file in files if file.name.startswith(f"{best_model_dir}/")
        ]
        for file in best_model_files:
            relative_path = file.name.replace(f"{best_model_dir}/", "")
            file_path = os.path.join(download_dir, relative_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.download(root=os.path.dirname(file_path), replace=True)
    elif mode == "checkpoint":
        checkpoint_files = [
            file for file in files if file.name.startswith(f"{checkpoint_name}/")
        ]
        for file in checkpoint_files:
            file.download(root=download_dir, replace=True)

print(f"Loading model from {checkpoint_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL == 'vinai/bertweet-base':
    config = PeriodClassifierConfig.from_pretrained(checkpoint_dir)
    model = PeriodClassifier.from_pretrained(checkpoint_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False, normalization=True)
else:
    model = LongformerForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

model.to(device)
model.eval()
print(f"Model loaded successfully from {checkpoint_dir}")


eval_dir = os.path.join(DATA_DIR, "challenge_data", "eval_tweets")
today = datetime.today().strftime('%Y-%m-%d')
output_dir = os.path.join("..", "predictions", f"preds-{today}-{wandb_run_id}")  # Directory to save predictions
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset=['Tweet']).reset_index(drop=True)
    df['Tweet'] = df['Tweet'].map(filter_tweet)
    df = df.dropna(subset=['Tweet']).reset_index(drop=True)
    return df

def prepare_inference_data(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = load_and_preprocess_data(filepath)
            all_data.append(df)
    if not all_data:
        raise ValueError(f"No CSV files found in the directory: {directory}")
    df_original = pd.concat(all_data, ignore_index=True)
    len_prev = len(df_original)

    df_original = df_original.drop_duplicates(subset=['Tweet']).reset_index(drop=True)
    print(f"Total global duplicates removed: {len_prev - len(df_original)}")
    
    grouped = df_original.groupby('ID')

    df_time_periods = grouped.agg({
        'MatchID': 'first',
        'PeriodID': 'first',
        'Tweet': lambda x: x.tolist()
    }).reset_index()
    
    df_time_periods['Tweet'] = df_time_periods['Tweet'].map(lambda tweets: [t for t in tweets if t])
    df_time_periods = df_time_periods[df_time_periods['Tweet'].map(len) > 0].reset_index(drop=True)
    
    return df_time_periods

df_inference = prepare_inference_data(eval_dir)
print(f"Total time periods for inference: {len(df_inference)}")


if MODEL == 'vinai/bertweet-base':
    class InferenceDataset(Dataset):
        def __init__(self, tweets_list, tokenizer, max_length=128, max_tweets=200):
            self.tweets_list = tweets_list
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.max_tweets = max_tweets

        def __len__(self):
            return len(self.tweets_list)

        def __getitem__(self, index):
            tweets = self.tweets_list[index]
            tweets = [str(tweet) if pd.notnull(tweet) else "[Empty Tweet]" for tweet in tweets]

            if self.max_tweets is not None:
                tweets = tweets[:self.max_tweets]

            if len(tweets) == 0:
                encoding = self.tokenizer(
                    "[No Tweets]",
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
            else:
                encodings = self.tokenizer(
                    tweets,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_attention_mask=True
                )
                input_ids = encodings['input_ids'] 
                attention_mask = encodings['attention_mask']

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

    def collate_fn_inference(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]

        max_num_tweets = max([ids.size(0) for ids in input_ids])
        max_seq_length = max([ids.size(1) for ids in input_ids])

        batch_size = len(batch)
        padded_input_ids = torch.zeros((batch_size, max_num_tweets, max_seq_length), dtype=torch.long)
        padded_attention_masks = torch.zeros((batch_size, max_num_tweets, max_seq_length), dtype=torch.long)

        for i in range(batch_size):
            num_tweets = input_ids[i].size(0)
            seq_length = input_ids[i].size(1)

            padded_input_ids[i, :num_tweets, :seq_length] = input_ids[i]
            padded_attention_masks[i, :num_tweets, :seq_length] = attention_masks[i]

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks
        }

    inference_dataset = InferenceDataset(
        tweets_list=df_inference['Tweet'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_tweets=MAX_TWEETS
    )

elif MODEL == 'allenai/longformer-base-4096':
    class InferenceDataset(Dataset):
        def __init__(self, tweets_list, tokenizer, max_length=4096):
            self.texts = [' '.join(tweets) for tweets in tweets_list]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            text = self.texts[index]
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

    def collate_fn_inference(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    inference_dataset = InferenceDataset(
        tweets_list=df_inference['Tweet'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

inference_dataloader = DataLoader(
    inference_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_inference
)

def predict_batch(batch, model, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits'] 
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    return predicted_classes.cpu().numpy(), probabilities.cpu().numpy()

print("Starting inference...")
all_predicted_classes = []
all_probabilities = []

for batch in tqdm(inference_dataloader, desc="Inference"):
    predicted_classes, probabilities = predict_batch(batch, model, device)
    all_predicted_classes.extend(predicted_classes)
    all_probabilities.extend(probabilities[:, 1]) 

df_inference['EventType'] = all_predicted_classes
df_inference['Probability_Class1'] = all_probabilities

print("Combining predictions into a single CSV...")
output_path = os.path.join(output_dir, "predictions.csv")
df_inference.to_csv(output_path, index=False)

print(f"Combined predictions saved to {output_path}")
print(df_inference['EventType'].value_counts())