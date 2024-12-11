import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import sys
from transformers import TrainerCallback

DATA_DIR = '/Data/tlh45/'
os.environ["HF_HOME"] = DATA_DIR
TRAINING_DIR = f"{DATA_DIR}/challenge_data/train_tweets"
MODEL = 'bert-base-cased'

def load_training_data(directory):
    all_data = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


df_original = load_training_data(TRAINING_DIR)


class TweetDataset(Dataset):
    def __init__(self, tweets, labels, period_ids, tokenizer, max_length=512):
        self.tweets = tweets
        self.labels = labels
        self.period_ids = period_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        tweet = str(self.tweets[index])
        period_id = str(self.period_ids[index])
        label = self.labels[index]
        tweet_with_context = f"[Period: {period_id}] {tweet}"

        encoding = self.tokenizer(
            tweet_with_context,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

df_sampled = df_original.drop_duplicates(subset=['Tweet', 'PeriodID']).reset_index(drop=True)

# df_sampled = df_sampled.sample(frac=0.5, random_state=42).reset_index(drop=True)
df_sampled = df_original.reset_index(drop=True)
print(f"Original dataset size: {df_original.shape[0]}")
print(f"Sampled dataset size (5%): {df_sampled.shape[0]}")

df_sampled['UniqueHash'] = df_sampled.apply(lambda row: hash(f"{row['Tweet']}-{row['PeriodID']}"), axis=1)

train_hashes, val_hashes = train_test_split(
    df_sampled['UniqueHash'], test_size=0.2, random_state=42
)

train_data = df_sampled[df_sampled['UniqueHash'].isin(train_hashes)]
val_data = df_sampled[df_sampled['UniqueHash'].isin(val_hashes)]

print("Training set class distribution:")
print(train_data['EventType'].value_counts())
print("Validation set class distribution:")
print(val_data['EventType'].value_counts())

train_text_set = set(train_data['UniqueHash'])
val_text_set = set(val_data['UniqueHash'])
print(f"Overlap between train and validation sets: {len(train_text_set & val_text_set)}")

train_texts = train_data['Tweet']
train_labels = train_data['EventType']
train_periods = train_data['PeriodID']

val_texts = val_data['Tweet']
val_labels = val_data['EventType']
val_periods = val_data['PeriodID']

tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.path.join(DATA_DIR, "hf_cache")) # roberta-base
train_dataset = TweetDataset(
    tweets=train_texts.tolist(),
    labels=train_labels.tolist(),
    period_ids=train_periods.tolist(),
    tokenizer=tokenizer,
    max_length=512
)

val_dataset = TweetDataset(
    tweets=val_texts.tolist(),
    labels=val_labels.tolist(),
    period_ids=val_periods.tolist(),
    tokenizer=tokenizer,
    max_length=512
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, cache_dir=os.path.join(DATA_DIR, "hf_cache"))
model.to(device)

class WandbCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        wandb.save(os.path.join(checkpoint_dir, "*"), base_path=args.output_dir)

wandb.login()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # print(f"Logits (sample): {logits[:10]}")
    # print(f"Probabilities (sample): {torch.softmax(torch.tensor(logits[:10]), dim=-1)}")
    # print(f"Predictions (sample): {predictions[:10]}")
    # print(f"Ground Truth (sample): {labels[:10]}")
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_metrics_debug(eval_pred):
    logits, labels = eval_pred
    print(f"Logits (sample): {logits[:5]}")
    print(f"Labels (sample): {labels[:5]}")
    return compute_metrics(eval_pred)


learning_rate = 2e-5
epochs = 3
training_args = TrainingArguments(
    output_dir=os.path.join(DATA_DIR, "MLDataChallenge2024"),
    eval_strategy="steps",
    eval_steps=2500,
    save_strategy="steps",
    save_steps=2500,
    save_total_limit=2,
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_dir=os.path.join(DATA_DIR, "logs"),
    logging_steps=100,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    disable_tqdm=False,
    gradient_accumulation_steps=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  
)
trainer.add_callback(WandbCheckpointCallback)

wandb.init(
    project="ml-data-challenge-24",
    entity="ml-data-challenge-24",
    dir=os.path.join(DATA_DIR, "wandb"),
    resume='must',
    # name=run_name,
    id='o3ki09bd'
)

checkpoint_path = os.path.join(DATA_DIR, "MLDataChallenge2024/checkpoint-17500") 
trainer.train(resume_from_checkpoint=checkpoint_path)

output_dir = os.path.join(DATA_DIR, "MLDataChallenge2024/BestModel")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
wandb.save(os.path.join(output_dir, "*"), base_path=DATA_DIR)

wandb.finish()