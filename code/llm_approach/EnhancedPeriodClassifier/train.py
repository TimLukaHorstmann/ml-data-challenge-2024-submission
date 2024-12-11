import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModel,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
from tqdm import tqdm
from datasets import Dataset as HFDataset, load_from_disk, DatasetDict, concatenate_datasets
import gc
import concurrent.futures
from torch.cuda.amp import autocast
import optuna
import sys
import h5py
from concurrent.futures import ProcessPoolExecutor
# import nltk
from sklearn.utils.class_weight import compute_class_weight
# from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

sys.path.append('../.')
from preprocessor import filter_tweet

# CHOOSE WHICH MODEL TO USE ("normal" version or "modified" version)
from enhancedperiodclassifier_modified import EnhancedPeriodClassifier, EnhancedPeriodClassifierConfig

# MEAN OR CLS EMBEDDINGS
USE_MEAN_EMBEDDING = True # (CLS use is experimental)

# ############## Function Definitions ############## #

def read_and_process_csv(filepath):
    processed_chunks = []
    chunksize = 100000  
    for chunk in pd.read_csv(filepath, chunksize=chunksize):

        chunk = chunk.drop_duplicates(subset=['Tweet'])
        chunk['Tweet'] = chunk['Tweet'].apply(filter_tweet)
        chunk = chunk.dropna(subset=['Tweet']).reset_index(drop=True)
        
        processed_chunks.append(chunk)
        
        # Clean up
        del chunk
        gc.collect()
    return processed_chunks

@staticmethod
def compute_sentiment_scores(tweets, sia):
    sentiments = [sia.polarity_scores(tweet)['compound'] for tweet in tweets]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment

@staticmethod
def load_and_process_data(directory):
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    all_processed_chunks = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(read_and_process_csv, file): file for file in csv_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing CSV files"):
            processed_chunks = future.result()
            all_processed_chunks.extend(processed_chunks)
            del processed_chunks
            gc.collect()
    
    df = pd.concat(all_processed_chunks, ignore_index=True)

    len_before = len(df)
    df = df.drop_duplicates(subset=['Tweet']).reset_index(drop=True)
    print(f"Total unique tweets after global deduplication: {len(df)}")
    print(f"Removed {len_before - len(df)} duplicate tweets.")

    del all_processed_chunks
    gc.collect()
    return df

def group_time_periods(df):
    grouped = df.groupby('ID').agg({
        'MatchID': 'first',
        'PeriodID': 'first',
        'EventType': 'first',
        'Tweet': list 
    }).reset_index()
    return grouped

@staticmethod
def normalize_features(df):
    """
    Normalizes 'NumTweets', 'AvgTweetLength', and 'PeriodID'.
    Saves normalization parameters to disk.
    """

    df['NumTweets'] = df['Tweet'].map(len)
    tweet_counts = df['NumTweets']
    num_tweets_mean = tweet_counts.mean()
    num_tweets_std = tweet_counts.std()
    df['NumTweets'] = (df['NumTweets'] - num_tweets_mean) / num_tweets_std

    tqdm_desc = "Calculating average tweet length"
    avg_lengths = []
    # sentiments = []
    for tweets in tqdm(df['Tweet'], desc=tqdm_desc):
        avg_lengths.append(
            np.mean([len(str(tweet)) for tweet in tweets]) if tweets else 0
        )
        #sentiments.append(compute_sentiment_scores(tweets, sia))
    
    df['AvgTweetLength'] = avg_lengths
    avg_tweet_length_mean = df['AvgTweetLength'].mean()
    avg_tweet_length_std = df['AvgTweetLength'].std()
    df['AvgTweetLength'] = (df['AvgTweetLength'] - avg_tweet_length_mean) / avg_tweet_length_std
    
    # Add sentiment scores
    # df['AvgSentiment'] = sentiments
    # sentiment_mean = df['AvgSentiment'].mean()
    # sentiment_std = df['AvgSentiment'].std()
    #  df['NormalizedSentiment'] = (df['AvgSentiment'] - sentiment_mean) / sentiment_std

    periodID_mean = df['PeriodID'].mean()
    periodID_std = df['PeriodID'].std()
    df['NormalizedPeriod'] = (df['PeriodID'] - periodID_mean) / periodID_std

    normalization_params = {
        'num_tweets_mean': num_tweets_mean,
        'num_tweets_std': num_tweets_std,
        'avg_tweet_length_mean': avg_tweet_length_mean,
        'avg_tweet_length_std': avg_tweet_length_std,
        'periodID_mean': periodID_mean,
        'periodID_std': periodID_std,
        # 'sentiment_mean': sentiment_mean,
        # 'sentiment_std': sentiment_std
    }
    print("Saving normalization parameters to", NORMALIZATION_PARAMS_PATH)
    with open(NORMALIZATION_PARAMS_PATH, 'w') as f:
        json.dump(normalization_params, f)
    
    return df

@staticmethod
def compute_mean_embeddings_batch(tweets_batch, bertweet_model=None, tokenizer=None, device='cuda', max_length=128):
    encoding = tokenizer(
        tweets_batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(device, non_blocking=True)
    attention_mask = encoding['attention_mask'].to(device, non_blocking=True)

    with torch.no_grad(), autocast(enabled=True):
        outputs = bertweet_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.cpu().numpy().astype('float16')

@staticmethod
def compute_cls_embeddings_batch(tweets_batch, bertweet_model=None, tokenizer=None, device='cuda', max_length=128):
    encoding = tokenizer(
        tweets_batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(device, non_blocking=True)
    attention_mask = encoding['attention_mask'].to(device, non_blocking=True)

    with torch.no_grad(), autocast(enabled=True):
        outputs = bertweet_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        cls_embeddings = last_hidden_state[:, 0, :]
        
    return cls_embeddings.cpu().numpy().astype('float16')

def compute_and_save_embeddings(df, embeddings_h5_path, masks_h5_path, batch_size_rows=2):

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=os.path.join(DATA_DIR, "hf_cache"),
        normalization=True,
        use_fast=False 
    )

    bertweet_model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=os.path.join(DATA_DIR, "hf_cache")
    )
    bertweet_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bertweet_model.to(device)
    torch.backends.cudnn.benchmark = True 

    num_time_periods = len(df)
    with h5py.File(embeddings_h5_path, 'w') as emb_h5, h5py.File(masks_h5_path, 'w') as mask_h5:
        emb_dataset = emb_h5.create_dataset(
            'embeddings',
            shape=(num_time_periods, MAX_TWEETS, EMBEDDING_DIM),
            dtype='float16',
            compression="gzip",
            chunks=(1, MAX_TWEETS, EMBEDDING_DIM)
        )
        mask_dataset = mask_h5.create_dataset(
            'masks',
            shape=(num_time_periods, MAX_TWEETS),
            dtype='bool',
            compression="gzip",
            chunks=(1, MAX_TWEETS) 
        )
        data_tuples = list(df.itertuples(index=False))
        total_batches = (num_time_periods + batch_size_rows - 1) // batch_size_rows

        if USE_MEAN_EMBEDDING:
            for batch_idx in tqdm(range(total_batches), total=total_batches, desc="Assigning embeddings"):
                start_idx = batch_idx * batch_size_rows
                end_idx = min(start_idx + batch_size_rows, num_time_periods)
                batch_rows = data_tuples[start_idx:end_idx]

                batch_tweets = []
                batch_masks = []
                for row in batch_rows:
                    tweets = row.Tweet
                    num_tweets = len(tweets)
                    
                    if num_tweets > MAX_TWEETS:
                        tweets = tweets[:MAX_TWEETS]
                        mask = [1] * MAX_TWEETS
                    else:
                        mask = [1] * num_tweets + [0] * (MAX_TWEETS - num_tweets)
                        tweets += [tokenizer.pad_token] * (MAX_TWEETS - num_tweets)
                    
                    batch_tweets.extend(tweets)
                    batch_masks.append(mask)
                embeddings = compute_mean_embeddings_batch(batch_tweets, bertweet_model=bertweet_model, tokenizer=tokenizer, device=device, max_length=MAX_LENGTH)
                embeddings = embeddings.reshape((end_idx - start_idx), MAX_TWEETS, EMBEDDING_DIM)

                masks = np.array(batch_masks, dtype='bool')

                emb_dataset[start_idx:end_idx] = embeddings
                mask_dataset[start_idx:end_idx] = masks

                del embeddings, masks, batch_tweets, batch_masks
                gc.collect()
        else:
            for batch_idx in tqdm(range(total_batches), total=total_batches, desc="Assigning embeddings"):
                start_idx = batch_idx * batch_size_rows
                end_idx = min(start_idx + batch_size_rows, num_time_periods)
                batch_rows = data_tuples[start_idx:end_idx]

                batch_embeddings = []
                batch_masks = []

                for row in batch_rows:
                    tweets = row.Tweet
                    num_tweets = len(tweets)

                    if num_tweets > MAX_TWEETS:
                        tweets = tweets[:MAX_TWEETS]
                        mask = [1] * MAX_TWEETS
                    else:
                        mask = [1] * num_tweets + [0] * (MAX_TWEETS - num_tweets)
                        tweets += [tokenizer.pad_token] * (MAX_TWEETS - num_tweets)
                    embeddings = compute_cls_embeddings_batch(
                        tweets, bertweet_model=bertweet_model, tokenizer=tokenizer, device=device, max_length=MAX_LENGTH
                    )

                    embeddings = embeddings.reshape((MAX_TWEETS, EMBEDDING_DIM))
                    batch_embeddings.append(embeddings)
                    batch_masks.append(mask)

                batch_embeddings = np.array(batch_embeddings, dtype='float16')
                batch_masks = np.array(batch_masks, dtype='bool')   

                emb_dataset[start_idx:end_idx] = batch_embeddings
                mask_dataset[start_idx:end_idx] = batch_masks

                del batch_embeddings, batch_masks
                gc.collect()

        del bertweet_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

def save_augmented_dataframe(df, augmented_df_path):
    df = df.drop(columns=['Embeddings', 'Masks'], errors='ignore')
    
    os.makedirs(os.path.dirname(augmented_df_path), exist_ok=True)
    df.to_parquet(augmented_df_path, index=False)
    print(f"Augmented DataFrame without embeddings saved to {augmented_df_path}")


if __name__ == "__main__":

    DATA_DIR = '/Data/tlh45/'
    TRAINING_DIR = os.path.join(DATA_DIR, 'challenge_data/train_tweets')
    AUGMENTED_DF_PATH = os.path.join(DATA_DIR, "NumLenPeriod_final.parquet")
    EMBEDDINGS_PATH = os.path.join(DATA_DIR, "NumLenPeriod_embeddings_final.h5")
    MASKS_PATH = os.path.join(DATA_DIR, "NumLenPeriod_masks_final.h5")
    DATASET_NAME = "period_level_agg_embds_NumLenPeriod_final"
    MODEL_NAME = 'vinai/bertweet-base'
    MAX_LENGTH = 128 
    MAX_TWEETS = 1500  # Maximum number of tweets per time period
    EMBEDDING_DIM = 768 
    BATCH_SIZE_EMBEDDINGS = 512  # Batch size for computing embeddings
    BATCH_SIZE = 32  # Batch size for training
    NUM_WORKERS = 4 
    EPOCHS = 80  
    NORMALIZATION_PARAMS_PATH = os.path.join(DATA_DIR, "normalization_params_final.json")
    EARLY_STOPPING_PATIENCE = 10 
    PERCENTAGE_VAL_DATA = 0.02 

    CHECKPOINT = None  # e.g., "checkpoint-795"
    WANDB_RUN_ID = None  # e.g., "rmhj5ijn"

    PREPROCESSING1 = True 
    PREPROCESSING2 = True  
    EXCLUDED_MATCH_IDS = None # Add MatchIDs to exclude from training, if any (did not improve performance)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    RUN_OPTUNA = False

    # sia = SentimentIntensityAnalyzer()

    # ############## Preprocessing1: Computing and saving embeddings ##############

    if PREPROCESSING1:
        print("Loading and processing data in chunks...")
        df_original = load_and_process_data(TRAINING_DIR)
        
        print("Grouping data by time periods...")
        df_time_periods = group_time_periods(df_original)
        
        del df_original
        gc.collect()

        print("Normalizing features...")
        df_time_periods = normalize_features(df_time_periods)
        
        print("Computing and saving embeddings and masks...")
        compute_and_save_embeddings(df_time_periods, EMBEDDINGS_PATH, MASKS_PATH)
        
        print("Saving augmented DataFrame...")
        save_augmented_dataframe(df_time_periods, AUGMENTED_DF_PATH)
        
        del df_time_periods
        gc.collect()
        torch.cuda.empty_cache()
        print("Preprocessing1 completed successfully!")

    # ############## Preprocessing2: Creating HF Dataset ##############

    if PREPROCESSING2:
        df_time_periods = pd.read_parquet(AUGMENTED_DF_PATH)

        with h5py.File(EMBEDDINGS_PATH, 'r') as emb_h5, h5py.File(MASKS_PATH, 'r') as mask_h5:
            embeddings = emb_h5['embeddings'][:] 
            masks = mask_h5['masks'][:]     
        
        if EXCLUDED_MATCH_IDS:
            df_time_periods = df_time_periods[~df_time_periods['MatchID'].isin(EXCLUDED_MATCH_IDS)].reset_index(drop=True)
            print(f"Filtered out matches: {EXCLUDED_MATCH_IDS}. Remaining time periods: {len(df_time_periods)}")
            filtered_ids = df_time_periods['ID'].values
            original_ids = pd.read_parquet(AUGMENTED_DF_PATH)['ID'].values
            id_to_index = {id_val: idx for idx, id_val in enumerate(original_ids)}
            filtered_indices = [id_to_index[i] for i in filtered_ids]

            embeddings = embeddings[filtered_indices]
            masks = masks[filtered_indices]

        labels = df_time_periods['EventType'].values 
        additional_features = df_time_periods[['NumTweets', 'AvgTweetLength', 'NormalizedPeriod']].values

        del df_time_periods
        gc.collect()

        print("Performing stratified train-validation split...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=PERCENTAGE_VAL_DATA, random_state=42)
        for train_idx, val_idx in sss.split(embeddings, labels): 
            train_embeddings = embeddings[train_idx]
            train_masks = masks[train_idx]
            train_labels = labels[train_idx]
            train_additional_features = additional_features[train_idx] 

            val_embeddings = embeddings[val_idx]
            val_masks = masks[val_idx]
            val_labels = labels[val_idx]
            val_additional_features = additional_features[val_idx] 

        del embeddings, masks, labels, additional_features
        gc.collect()

        with open(NORMALIZATION_PARAMS_PATH, 'r') as f:
            normalization_params = json.load(f)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )

        class_weights = class_weights.astype(np.float32)
        normalization_params['class_weights'] = class_weights.tolist()

        with open(NORMALIZATION_PARAMS_PATH, 'w') as f:
            json.dump(normalization_params, f)

        def create_hf_dataset(embeddings, masks, additional_features, labels, chunk_size=500):
            datasets = []
            for i in range(0, len(embeddings), chunk_size):
                chunk_embeddings = embeddings[i:i + chunk_size]
                chunk_masks = masks[i:i + chunk_size]
                chunk_additional_features = additional_features[i:i + chunk_size]
                chunk_labels = labels[i:i + chunk_size]
                datasets.append(HFDataset.from_dict({
                    'embeddings': chunk_embeddings,
                    'masks': chunk_masks,
                    'additional_features': chunk_additional_features,
                    'labels': chunk_labels
                }))
            return concatenate_datasets(datasets)

        print("Creating HF train dataset in chunks...")
        train_dataset = create_hf_dataset(
            train_embeddings, train_masks, train_additional_features, train_labels, len(train_embeddings) // 5
        )
        del train_embeddings, train_masks, train_additional_features, train_labels
        gc.collect()

        print("Creating HF validation dataset in chunks...")
        val_dataset = create_hf_dataset(
            val_embeddings, val_masks, val_additional_features, val_labels, len(val_embeddings) // 5
        )
        del val_embeddings, val_masks, val_additional_features, val_labels
        gc.collect()

        dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

        processed_data_dir = os.path.join(DATA_DIR, DATASET_NAME)
        dataset_dict.save_to_disk(processed_data_dir)
        print(f"Processed DatasetDict saved to {processed_data_dir}")

    # ############## Loading Processed Dataset ############## 
    if not PREPROCESSING2:
        print("Loading processed dataset from disk...")
        processed_data_dir = os.path.join(DATA_DIR, DATASET_NAME)
        dataset_dict = load_from_disk(processed_data_dir)
        print("Dataset loaded successfully!")

    with open(NORMALIZATION_PARAMS_PATH, 'r') as f:
        normalization_params = json.load(f)
    class_weights = torch.tensor(normalization_params['class_weights'], dtype=torch.float32).to(DEVICE)

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
    
    # ############## Define Callbacks ##############

    class WandbCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            wandb.save(os.path.join(checkpoint_dir, "*"), base_path=args.output_dir)
    
    class CustomPruningCallback(TrainerCallback):
        def __init__(self, trial, metric_name="eval_accuracy"):
            super().__init__()
            self.trial = trial
            self.metric_name = metric_name

        def on_evaluate(self, args, state, control, **kwargs):
            logs = kwargs.get("metrics", {})
            if self.metric_name in logs:
                current_score = logs[self.metric_name]
                self.trial.report(current_score, step=state.global_step)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
    callbacks = [WandbCheckpointCallback(), early_stopping_callback]

    # ############## Initialize Data Collator ############## 
    def custom_data_collator(features):
        embeddings = torch.tensor([f['embeddings'] for f in features], dtype=torch.float32) 
        masks = torch.tensor([f['masks'] for f in features], dtype=torch.float32) 
        additional_features = torch.tensor([f['additional_features'] for f in features], dtype=torch.float32)  
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)  
        return {
            'embeddings': embeddings,
            'masks': masks,
            'additional_features': additional_features,
            'labels': labels
        }

    def train_model():

        print("Initializing WandB...")
        wandb.login()

        run_name = f"{MODEL_NAME}-attention-FROZEN-filtered"
        wandb.init(
            project="ml-data-challenge-24",
            entity="ml-data-challenge-24",
            dir=os.path.join(DATA_DIR, "wandb"),
            resume="must" if CHECKPOINT is not None else "allow",
            id=WANDB_RUN_ID,
            name=run_name
        )

        with open(NORMALIZATION_PARAMS_PATH, 'r') as f:
            normalization_params = json.load(f)
        wandb.config.update(normalization_params)

        wandb.save(NORMALIZATION_PARAMS_PATH, base_path=DATA_DIR)

        if CHECKPOINT is not None:
            print(f"Resuming training from checkpoint: {CHECKPOINT}")
            checkpoint_path = os.path.join(DATA_DIR, "MLDataChallenge2024", CHECKPOINT)
        else:
            checkpoint_path = None

        config = EnhancedPeriodClassifierConfig(
            embedding_dim=EMBEDDING_DIM,
            num_classes=2,
            num_heads=8,
            dropout_prob=0.5,
            additional_features_dim= 3, #4
            fc1_dim= 4096 # 2048
        )
        model = EnhancedPeriodClassifier(config, class_weights=class_weights)
        model.to(DEVICE)


        total_steps = (len(dataset_dict['train']) // BATCH_SIZE) * EPOCHS
        warmup_steps = int(0.1 * total_steps)
        print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

        training_args = TrainingArguments(
            output_dir=os.path.join(DATA_DIR, "MLDataChallenge2024"),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=1e-4,
            lr_scheduler_type='linear',
            warmup_steps=warmup_steps,
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
            gradient_accumulation_steps=1,
            dataloader_num_workers=NUM_WORKERS
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            compute_metrics=compute_metrics,
            data_collator=custom_data_collator,
            callbacks=callbacks
        )


        print("Cleaning up unused memory...")
        torch.cuda.empty_cache()
        gc.collect()

        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            print("Starting fresh training...")
            trainer.train()

        best_model_dir = os.path.join(DATA_DIR, f"MLDataChallenge2024/BestModel-{WANDB_RUN_ID}")
        os.makedirs(best_model_dir, exist_ok=True)
        model.save_pretrained(best_model_dir) 
        print(f"Best model saved to {best_model_dir}")

        wandb.save(os.path.join(best_model_dir, "*"), base_path=DATA_DIR)

        wandb.finish()

    ### Optuna Hyperparameter Optimization ###

    def objective(trial):
        run_name = f"Optuna-Trial-{trial.number}"
        wandb.init(
            project="ml-data-challenge-24",
            entity="ml-data-challenge-24",
            dir=os.path.join(DATA_DIR, "wandb"),
            name=run_name,
            reinit=True, 
            config=trial.params  
        )
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5, log=False)
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 12])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        fc1_dim = trial.suggest_categorical('fc1_dim', [512, 1024, 2048, 4096])

        print(f"Candidate hyperparameters for Trial {trial.number}:")
        print(f"Learning rate: {learning_rate}, Dropout: {dropout_prob}, Num heads: {num_heads}, Weight decay: {weight_decay}, FC1 dim: {fc1_dim}")


        total_steps = (len(dataset_dict['train']) // BATCH_SIZE) * EPOCHS
        warmup_steps = int(0.1 * total_steps)
        print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        training_args = TrainingArguments(
            output_dir=os.path.join(DATA_DIR, "MLDataChallenge2024"),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=learning_rate, 
            lr_scheduler_type='linear',
            warmup_steps=warmup_steps,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=weight_decay,
            logging_dir=os.path.join(DATA_DIR, "logs"),
            logging_steps=10,
            report_to="wandb",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=torch.cuda.is_available(),
            disable_tqdm=False,
            gradient_accumulation_steps=1,
            dataloader_num_workers=NUM_WORKERS
        )

        config = EnhancedPeriodClassifierConfig(
            embedding_dim=EMBEDDING_DIM,
            num_classes=2,
            num_heads=num_heads,  
            dropout_prob=dropout_prob,  
            additional_features_dim=3,
            fc1_dim=fc1_dim 
        )

        model = EnhancedPeriodClassifier(config, class_weights=class_weights)
        model.to(DEVICE)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            compute_metrics=compute_metrics,
            data_collator=custom_data_collator,
            callbacks=[WandbCheckpointCallback(), early_stopping_callback, CustomPruningCallback(trial, metric_name="eval_accuracy")]
        )

        trainer.train()

        eval_result = trainer.evaluate()

        accuracy = eval_result['eval_accuracy']

        wandb.log(eval_result) 
        wandb.finish()

        return accuracy 

    def run_optuna():
        study = optuna.create_study(
            direction='maximize', 
            study_name='EnhancedPeriodClassifier_HPO',
            storage=f'sqlite:///{os.path.join(DATA_DIR, "hpo_study_new.db")}',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=30, timeout=86400)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        study_results_path = os.path.join(DATA_DIR, "hpo_study_results.json")
        with open(study_results_path, 'w') as f:
            json.dump({
                'best_value': trial.value,
                'best_params': trial.params,
                'trials': [t.__dict__ for t in study.trials]
            }, f, indent=4)
        print(f"Optuna study results saved to {study_results_path}")

    # Here we can choose to run Optuna or train the model --> MAIN PROGRAM!
    if RUN_OPTUNA:
        run_optuna()
    else:
        train_model()