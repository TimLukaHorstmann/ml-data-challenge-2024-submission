import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import wandb
from enhancedperiodclassifier_modified import EnhancedPeriodClassifier, EnhancedPeriodClassifierConfig
import gc
import json
import sys
from torch.cuda.amp import autocast

sys.path.append('../.')
from preprocessor import filter_tweet
import EnhancedPeriodClassifier.train as train

# Import nltk and initialize SentimentIntensityAnalyzer
import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
# sia = SentimentIntensityAnalyzer()

# Parse arguments for wandb_run_id (we pull the model from this run)
parser = argparse.ArgumentParser(description="Inference script for EnhancedPeriodClassifier")
parser.add_argument('-w', '--wandb_run_id', type=str, required=True, help='The WandB run ID to use for downloading the model/checkpoint')
args = parser.parse_args()


DATA_DIR = '/Data/tlh45/'
os.environ["HF_HOME"] = DATA_DIR
wandb_project = "ml-data-challenge-24"
wandb_entity = "ml-data-challenge-24"
wandb_run_id = args.wandb_run_id
best_model_dir = "MLDataChallenge2024/BestModel-None" 
MAX_TWEETS = 1500
EMBEDDING_DIM = 768
MAX_LENGTH = 128
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mode: "best_model" or "checkpoint"
mode = "best_model"

download_dir = os.path.join(DATA_DIR, f"BestModel-{wandb_run_id}") if mode == "best_model" else os.path.join(DATA_DIR, f"Checkpoints-{wandb_run_id}")
checkpoint_dir = os.path.join(download_dir, best_model_dir)

if not os.path.exists(checkpoint_dir):
    print(f"Downloading {mode} from W&B to {checkpoint_dir}...")
    api = wandb.Api()
    run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
    files = run.files()
    os.makedirs(download_dir, exist_ok=True)
    for file in files:
        file.download(root=download_dir, replace=True)

print(f"Loading model from {checkpoint_dir}...")

try:
    model = EnhancedPeriodClassifier.from_pretrained(checkpoint_dir)
    model.to(device).eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

tokenizer = AutoTokenizer.from_pretrained(
    'vinai/bertweet-base',
    cache_dir=os.path.join(DATA_DIR, "hf_cache"),
    normalization=True,
    use_fast=False
)
bertweet_model = AutoModel.from_pretrained(
    'vinai/bertweet-base',
    cache_dir=os.path.join(DATA_DIR, "hf_cache")
)
bertweet_model.to(device).eval()
torch.backends.cudnn.benchmark = True

# Define output directory
today = datetime.today().strftime('%Y-%m-%d')
output_dir = os.path.join("..", "predictions", f"preds-{today}-{wandb_run_id}")
os.makedirs(output_dir, exist_ok=True)

normalization_params_path = os.path.join(DATA_DIR, "normalization_params_final.json")
if not os.path.exists(normalization_params_path):
    raise FileNotFoundError(f"Normalization parameters not found at {normalization_params_path}")

with open(normalization_params_path, 'r') as f:
    normalization_params = json.load(f)

def compute_and_normalize_features(tweets, period_id, normalization_params):
    num_tweets = len(tweets)
    if num_tweets > 0:
        avg_tweet_length = np.mean([len(str(tweet)) for tweet in tweets])
        # avg_sentiment = train.compute_sentiment_scores(tweets, sia)
    else:
        avg_tweet_length = 0 
        avg_sentiment = 0

    num_tweets_norm = (num_tweets - normalization_params['num_tweets_mean']) / normalization_params['num_tweets_std']
    avg_tweet_length_norm = (avg_tweet_length - normalization_params['avg_tweet_length_mean']) / normalization_params['avg_tweet_length_std']
    normalized_period = (period_id - normalization_params['periodID_mean']) / normalization_params['periodID_std']
    # normalized_sentiment = (avg_sentiment - normalization_params['sentiment_mean']) / normalization_params['sentiment_std']

    return num_tweets_norm, avg_tweet_length_norm, normalized_period #, normalized_sentiment

def prepare_embeddings_and_masks(tweets, tokenizer, bertweet_model, device):
    if len(tweets) > MAX_TWEETS:
        tweets = tweets[:MAX_TWEETS]
        mask = [1] * MAX_TWEETS
    else:
        mask = [1] * len(tweets) + [0] * (MAX_TWEETS - len(tweets))
        tweets += [tokenizer.pad_token] * (MAX_TWEETS - len(tweets)) 

    # embeddings = train.compute_mean_embeddings_batch(tweets, bertweet_model=bertweet_model, tokenizer=tokenizer, device=device, max_length=MAX_LENGTH)
    # OR
    embeddings = train.compute_cls_embeddings_batch(tweets, bertweet_model=bertweet_model, tokenizer=tokenizer, device=device, max_length=MAX_LENGTH)
    return embeddings, mask

def predict_batch(tweets_batch, period_ids, normalization_params, tokenizer, bertweet_model, model, device):
    batch_size = len(tweets_batch)

    num_tweets_norm = []
    avg_tweet_length_norm = []
    normalized_period = []
    normalized_sentiment = []
    for tweets, period_id in zip(tweets_batch, period_ids):
        # nt_norm, at_norm, np_norm, ns_norm = compute_and_normalize_features(tweets, period_id, normalization_params)
        nt_norm, at_norm, np_norm = compute_and_normalize_features(tweets, period_id, normalization_params)
        num_tweets_norm.append(nt_norm)
        avg_tweet_length_norm.append(at_norm)
        normalized_period.append(np_norm)
        # normalized_sentiment.append(ns_norm)

    embeddings_batch = []
    masks_batch = []
    for tweets in tweets_batch:
        embeddings, mask = prepare_embeddings_and_masks(tweets, tokenizer, bertweet_model, device)
        embeddings_batch.append(embeddings)
        masks_batch.append(mask)

    embeddings_tensor = torch.tensor(np.stack(embeddings_batch), dtype=torch.float32).to(device)
    masks_tensor = torch.tensor(np.stack(masks_batch), dtype=torch.float32).to(device)
    additional_features = torch.tensor([
        # [nt, at, np, ns] for nt, at, np, ns in zip(num_tweets_norm, avg_tweet_length_norm, normalized_period, normalized_sentiment)
        [nt, at, np] for nt, at, np in zip(num_tweets_norm, avg_tweet_length_norm, normalized_period)
    ], dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(
            embeddings=embeddings_tensor,
            masks=masks_tensor,
            additional_features=additional_features
        )
        logits = outputs['logits']
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

    torch.cuda.empty_cache()
    gc.collect()

    return predictions.cpu().numpy(), probabilities.cpu().numpy()[:, 1]

def group_evaluation_data(df):
    grouped = df.groupby('ID').agg({
        'MatchID': 'first',
        'PeriodID': 'first',
        'Tweet': list 
    }).reset_index()
    return grouped

def load_and_preprocess_eval_data(eval_path):
    df = pd.read_csv(eval_path)
    df = df.drop_duplicates(subset=['Tweet'])
    df['Tweet'] = df['Tweet'].apply(filter_tweet)
    df = df.dropna(subset=['Tweet']).reset_index(drop=True)
    return df

def load_predictions(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.startswith("predictions_") and filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

eval_dir = os.path.join(DATA_DIR, "challenge_data", "eval_tweets")
for eval_file in tqdm(os.listdir(eval_dir), desc="Processing evaluation files"):
    if eval_file.endswith(".csv"):
        eval_path = os.path.join(eval_dir, eval_file)
        df = load_and_preprocess_eval_data(eval_path)
        grouped = group_evaluation_data(df)

        tweets_list = grouped['Tweet'].tolist()
        period_ids = grouped['PeriodID'].tolist()

        predictions = []
        probabilities = []

        for i in tqdm(range(0, len(tweets_list), BATCH_SIZE), desc=f"Evaluating {eval_file}"):
            batch_tweets = tweets_list[i:i + BATCH_SIZE]
            batch_period_ids = period_ids[i:i + BATCH_SIZE]

            preds, probs = predict_batch(
                batch_tweets, batch_period_ids, normalization_params, tokenizer, bertweet_model, model, device
            )
            predictions.extend(preds)
            probabilities.extend(probs)

        grouped['EventType'] = predictions
        grouped['Probability'] = probabilities

        output_path = os.path.join(output_dir, f"predictions_{eval_file}")
        grouped.to_csv(output_path, index=False)

print("Inference completed. Predictions saved to:", output_dir)

df_predictions = load_predictions(output_dir)
combined_output_path = os.path.join(output_dir, "predictions.csv")
df_predictions.to_csv(combined_output_path, index=False)

print(f"Combined predictions saved to {combined_output_path}")
print(df_predictions['EventType'].value_counts())