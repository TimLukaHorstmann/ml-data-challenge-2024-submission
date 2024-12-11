import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
from transformers import TrainerCallback
from periodclassifier import PeriodClassifier, PeriodClassifierConfig
import sys
from sklearn.utils.class_weight import compute_class_weight
from transformers import LongformerForSequenceClassification
import datasets

sys.path.append('../.')
from preprocessor import filter_tweet


DATA_DIR = '/Data/tlh45/'
os.environ["HF_HOME"] = DATA_DIR
TRAINING_DIR = f"{DATA_DIR}/challenge_data/train_tweets"
MODEL = 'vinai/bertweet-base' or 'bert-base-cased'  "allenai/longformer-base-4096"
MAX_TWEETS = 200
EPOCHS = 10 # 35 for bertweet-base, 5 for longformer-base-4096
BATCH_SIZE = 2

def load_training_data(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


print("Loading training data...")
df_original = load_training_data(TRAINING_DIR)
len_prev = len(df_original)


df_original = df_original.drop_duplicates(subset=['Tweet']).reset_index(drop=True)
print(f"Total global duplicates removed: {len_prev - len(df_original)}")

df_original['Tweet'] = df_original['Tweet'].map(filter_tweet)

df_original = df_original.dropna(subset=['Tweet']).reset_index(drop=True)


grouped = df_original.groupby('ID')

df_time_periods = grouped.agg({
    'MatchID': 'first',
    'PeriodID': 'first',
    'EventType': 'first',
    'Tweet': lambda x: x.tolist() 
}).reset_index()

df_time_periods['Tweet'] = df_time_periods['Tweet'].map(lambda tweets: [t for t in tweets if t])  
num_no_tweets = df_time_periods['Tweet'].map(len).eq(0).sum()
print(f"Found {num_no_tweets} time periods with no tweets.")
df_time_periods = df_time_periods[df_time_periods['Tweet'].map(len) > 0].reset_index(drop=True)

missing_labels = df_time_periods['EventType'].isnull().sum()
if missing_labels > 0:
    print(f"Found {missing_labels} missing labels. Removing these entries.")
    df_time_periods = df_time_periods.dropna(subset=['EventType']).reset_index(drop=True)

df_time_periods['EventType'] = df_time_periods['EventType'].astype(int)

df_time_periods['NumTweets'] = df_time_periods['Tweet'].map(len)

df_time_periods['NumTweets'] = (df_time_periods['NumTweets'] - df_time_periods['NumTweets'].mean()) / df_time_periods['NumTweets'].std()

df_time_periods['AvgTweetLength'] = df_time_periods['Tweet'].map(lambda tweets: np.mean([len(str(tweet)) for tweet in tweets]))
df_time_periods['AvgTweetLength'] = (df_time_periods['AvgTweetLength'] - df_time_periods['AvgTweetLength'].mean()) / df_time_periods['AvgTweetLength'].std()
print(df_time_periods[['ID', 'NumTweets']])
print("Statistics for tweets per time period:")
print(df_time_periods['NumTweets'].describe())
max_tweets = df_time_periods['NumTweets'].max()
min_tweets = df_time_periods['NumTweets'].min()
print(f"Maximum number of tweets in a single time period: {max_tweets}")
print(f"Minimum number of tweets in a single time period: {min_tweets}")
print("Time periods with 0 tweets:")
print(df_time_periods[df_time_periods['NumTweets'] == 0])
print("Time periods with maximum tweets:")
print(df_time_periods[df_time_periods['NumTweets'] == max_tweets])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_idx, val_idx in sss.split(df_time_periods, df_time_periods['EventType']):
    train_data = df_time_periods.iloc[train_idx]
    val_data = df_time_periods.iloc[val_idx]

missing_labels_train = train_data['EventType'].isnull().sum()
if missing_labels_train > 0:
    print(f"Found {missing_labels_train} missing labels in training data. Removing these entries.")
    train_data = train_data.dropna(subset=['EventType']).reset_index(drop=True)
missing_labels_val = val_data['EventType'].isnull().sum()
if missing_labels_val > 0:
    print(f"Found {missing_labels_val} missing labels in validation data. Removing these entries.")
    val_data = val_data.dropna(subset=['EventType']).reset_index(drop=True)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if MODEL == 'vinai/bertweet-base':
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.path.join(DATA_DIR, "hf_cache"), use_fast=False, normalization=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.path.join(DATA_DIR, "hf_cache"))

if MODEL == 'vinai/bertweet-base':
    train_dataset = datasets.TimePeriodDataset(
        tweets_list=train_data['Tweet'].tolist(),
        labels=train_data['EventType'],
        tokenizer=tokenizer,
        max_length=128,
        max_tweets=MAX_TWEETS
    )

    val_dataset = datasets.TimePeriodDataset(
        tweets_list=val_data['Tweet'].tolist(),
        labels=val_data['EventType'],
        tokenizer=tokenizer,
        max_length=128,
        max_tweets=MAX_TWEETS
    )
    train_labels = train_data['EventType'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    config = PeriodClassifierConfig(model_name=MODEL, hidden_size=768)
    model = PeriodClassifier(config)

elif MODEL == 'allenai/longformer-base-4096':
    train_dataset = datasets.LongformerDataset(
        tweets_list=train_data['Tweet'].tolist(),
        labels=train_data['EventType'],
        tokenizer=tokenizer,
        max_length=4096
    )

    val_dataset = datasets.LongformerDataset(
        tweets_list=val_data['Tweet'].tolist(),
        labels=val_data['EventType'],
        tokenizer=tokenizer,
        max_length=4096
    )
    model = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096",
    num_labels=2
    )

model.to(device)


def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        raise ValueError("Batch is empty after filtering out None items.")

    for i, item in enumerate(batch):
        if 'labels' not in item:
            print(f"Batch item at index {i} is missing 'labels': {item}")
            raise KeyError(f"'labels' key not found in batch item at index {i}.")

    labels = torch.stack([item['labels'] for item in batch])

    max_num_tweets = max([item['input_ids'].size(0) for item in batch])
    max_seq_length = max([item['input_ids'].size(1) for item in batch])

    input_ids_padded = []
    attention_masks_padded = []

    for item in batch:
        num_tweets = item['input_ids'].size(0)
        seq_length = item['input_ids'].size(1)

        if seq_length < max_seq_length:
            seq_padding = torch.zeros((num_tweets, max_seq_length - seq_length), dtype=torch.long)
            input_ids = torch.cat([item['input_ids'], seq_padding], dim=1)
            attention_mask = torch.cat([
                item['attention_mask'],
                torch.zeros((num_tweets, max_seq_length - seq_length), dtype=torch.long)
            ], dim=1)
        else:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']

        num_padding_tweets = max_num_tweets - num_tweets
        if num_padding_tweets > 0:
            padding_input_ids = torch.zeros((num_padding_tweets, max_seq_length), dtype=torch.long)
            padding_attention_mask = torch.zeros((num_padding_tweets, max_seq_length), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, padding_attention_mask], dim=0)

        input_ids_padded.append(input_ids)
        attention_masks_padded.append(attention_mask)

    input_ids_padded = torch.stack(input_ids_padded) 
    attention_masks_padded = torch.stack(attention_masks_padded) 

    tweet_masks = []
    for item in batch:
        num_tweets = item['input_ids'].size(0)
        num_padding_tweets = max_num_tweets - num_tweets
        mask = torch.cat(
            [
                torch.ones(num_tweets, dtype=torch.float32),
                torch.zeros(num_padding_tweets, dtype=torch.float32)
            ],
            dim=0
        )
        tweet_masks.append(mask)

    tweet_masks = torch.stack(tweet_masks)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'tweet_mask': tweet_masks,
        'labels': labels
    }

def collate_fn_longformer(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

wandb.login()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if MODEL == 'vinai/bertweet-base':
    learning_rate = 2e-5
elif MODEL == 'allenai/longformer-base-4096':
    learning_rate = 1e-5

training_args = TrainingArguments(
    output_dir=os.path.join(DATA_DIR, "MLDataChallenge2024"),
    evaluation_strategy="epoch",
    # eval_steps=50,
    save_strategy="epoch",
    # save_steps=250,
    save_total_limit=2,
    learning_rate=learning_rate,
    lr_scheduler_type="linear",
    warmup_steps=100,
    per_device_train_batch_size=BATCH_SIZE, 
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=os.path.join(DATA_DIR, "logs"),
    logging_steps=10,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    disable_tqdm=False,
    gradient_accumulation_steps=16,
    remove_unused_columns=False, 
)

run_name = f"{MODEL}-agg-lr-{learning_rate}-epochs-{EPOCHS}-WITH-PERIOD-preprocessed"
wandb.init(
    project="ml-data-challenge-24",
    entity="ml-data-challenge-24",
    dir=os.path.join(DATA_DIR, "wandb"),
    resume=False,
    name=run_name
)

class WandbCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        wandb.save(os.path.join(checkpoint_dir, "*"), base_path=args.output_dir)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

if MODEL == 'vinai/bertweet-base':
    data_collator = collate_fn 
elif MODEL == 'allenai/longformer-base-4096':
    data_collator = collate_fn_longformer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[early_stopping]
)

trainer.add_callback(WandbCheckpointCallback)

trainer.train()

output_dir = os.path.join(DATA_DIR, "MLDataChallenge2024/BestModel")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.save(os.path.join(output_dir, "*"), base_path=DATA_DIR)

wandb.finish()