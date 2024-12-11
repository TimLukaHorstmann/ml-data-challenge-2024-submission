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
import re
import sys
import h5py
from concurrent.futures import ProcessPoolExecutor
import nltk
from sklearn.utils.class_weight import compute_class_weight
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sys.path.append('../.')
from preprocessor import filter_tweet

from enhancedperiodclassifier import EnhancedPeriodClassifier, EnhancedPeriodClassifierConfig

def read_and_process_csv(filepath):
    processed_chunks = []
    chunksize = 100000 
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk = chunk.drop_duplicates(subset=['Tweet'])

        chunk['Tweet'] = chunk['Tweet'].apply(filter_tweet)
        
        chunk = chunk.dropna(subset=['Tweet']).reset_index(drop=True)
        
        processed_chunks.append(chunk)
        
        del chunk
        gc.collect()
    return processed_chunks


def compute_sentiment_scores(tweets, sia):
    sentiments = [sia.polarity_scores(tweet)['compound'] for tweet in tweets]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment


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

def get_additional_features_scenario(scenario):
    all_features = {
        'NumTweets': 'NumTweets',
        'AvgTweetLength': 'AvgTweetLength',
        'NormalizedPeriod': 'NormalizedPeriod',
        'NormalizedSentiment': 'NormalizedSentiment'
    }
    
    if scenario == 'all':
        return list(all_features.values()), len(all_features)
    elif scenario == 'exclude_numtweets':
        features = [v for k, v in all_features.items() if k != 'NumTweets']
        return features, len(features)
    elif scenario == 'exclude_avgtweetlength':
        features = [v for k, v in all_features.items() if k != 'AvgTweetLength']
        return features, len(features)
    elif scenario == 'exclude_normalizedperiod':
        features = [v for k, v in all_features.items() if k != 'NormalizedPeriod']
        return features, len(features)
    elif scenario == 'exclude_normalizedsentiment':
        features = [v for k, v in all_features.items() if k != 'NormalizedSentiment']
        return features, len(features)
    elif scenario == 'exclude_all':
        return [], 0
    else:
        raise ValueError("Invalid scenario name.")


def normalize_features(df, selected_features, scenario):
    normalization_params = {}
    
    if 'NumTweets' in selected_features:
        print("Calculating number of tweets per time period...")
        df['NumTweets'] = df['Tweet'].map(len)
        num_tweets_mean = df['NumTweets'].mean()
        num_tweets_std = df['NumTweets'].std()
        df['NumTweets'] = (df['NumTweets'] - num_tweets_mean) / num_tweets_std
        normalization_params['num_tweets_mean'] = num_tweets_mean
        normalization_params['num_tweets_std'] = num_tweets_std
    
    if 'AvgTweetLength' in selected_features:
        print("Calculating average tweet length per time period...")
        avg_lengths = []
        for tweets in tqdm(df['Tweet'], desc="Calculating average tweet length and sentiments"):
            avg_length = np.mean([len(str(tweet)) for tweet in tweets]) if len(tweets) > 0 else 0
            avg_lengths.append(avg_length)
        df['AvgTweetLength'] = avg_lengths
        avg_tweet_length_mean = df['AvgTweetLength'].mean()
        avg_tweet_length_std = df['AvgTweetLength'].std()
        df['AvgTweetLength'] = (df['AvgTweetLength'] - avg_tweet_length_mean) / avg_tweet_length_std
        normalization_params['avg_tweet_length_mean'] = avg_tweet_length_mean
        normalization_params['avg_tweet_length_std'] = avg_tweet_length_std
    
    if 'NormalizedSentiment' in selected_features:
        print("Calculating sentiment scores per time period...")
        df['AvgSentiment'] = [compute_sentiment_scores(tweets, sia) for tweets in df['Tweet']]
        sentiment_mean = df['AvgSentiment'].mean()
        sentiment_std = df['AvgSentiment'].std()
        df['NormalizedSentiment'] = (df['AvgSentiment'] - sentiment_mean) / sentiment_std
        normalization_params['sentiment_mean'] = sentiment_mean
        normalization_params['sentiment_std'] = sentiment_std
    
    if 'NormalizedPeriod' in selected_features:
        print("Normalizing PeriodID...")
        periodID_mean = df['PeriodID'].mean()
        periodID_std = df['PeriodID'].std()
        df['NormalizedPeriod'] = (df['PeriodID'] - periodID_mean) / periodID_std
        normalization_params['periodID_mean'] = periodID_mean
        normalization_params['periodID_std'] = periodID_std
    
    normalization_params_path = os.path.join(DATA_DIR, f"normalization_params-{model_suffix}-{scenario}.json")
    with open(normalization_params_path, 'w') as f:
        json.dump(normalization_params, f)
    
    return df, normalization_params_path


def prepare_additional_features(additional_features, selected_features):
    feature_indices = {
        'NumTweets': 0,
        'AvgTweetLength': 1,
        'NormalizedPeriod': 2,
        'NormalizedSentiment': 3
    }
    selected_indices = [feature_indices[feat] for feat in selected_features]
    if not selected_indices:
        return np.empty((additional_features.shape[0], 0), dtype=np.float32)
    return additional_features[:, selected_indices]


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

            embeddings = compute_mean_embeddings_batch(batch_tweets, bertweet_model=bertweet_model, tokenizer=tokenizer, device=device, max_length=MAX_LENGTH)  # Shape: (batch_size_rows * MAX_TWEETS, EMBEDDING_DIM)
            
            embeddings = embeddings.reshape((end_idx - start_idx), MAX_TWEETS, EMBEDDING_DIM)
            
            masks = np.array(batch_masks, dtype='bool')
            
            emb_dataset[start_idx:end_idx] = embeddings
            mask_dataset[start_idx:end_idx] = masks

            del embeddings, masks, batch_tweets, batch_masks
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
    MODEL_NAME = 'vinai/bertweet-base' # 'cardiffnlp/twitter-roberta-base-2022-154m' # 
    model_suffix = MODEL_NAME.split('/')[-1]

    DATA_DIR = '/Data/tlh45/'
    TRAINING_DIR = os.path.join(DATA_DIR, 'challenge_data/train_tweets')
    AUGMENTED_DF_PATH = os.path.join(DATA_DIR, f"NumLenPeriod-{model_suffix}.parquet")
    EMBEDDINGS_PATH = os.path.join(DATA_DIR, f"NumLenPeriod_embeddings-{model_suffix}.h5")
    MASKS_PATH = os.path.join(DATA_DIR, f"NumLenPeriod_masks-{model_suffix}.h5")
    DATASET_NAME = f"period_level_agg_embds_NumLenPeriod-{model_suffix}"

    MAX_LENGTH = 128  
    MAX_TWEETS = 1500  
    EMBEDDING_DIM = 768 
    BATCH_SIZE_EMBEDDINGS = 512 
    BATCH_SIZE = 32 
    NUM_WORKERS = 4 
    EPOCHS = 80 
    EARLY_STOPPING_PATIENCE = 10
    PERCENTAGE_VAL_DATA = 0.1 

    CHECKPOINT = None  # e.g., "checkpoint-795"
    WANDB_RUN_ID = None  # e.g., "rmhj5ijn"

    PREPROCESSING1 = False 
    PREPROCESSING2 = True 
    EXCLUDED_MATCH_IDS = None

    sia = SentimentIntensityAnalyzer()

    ablation_scenarios = [
        'all',
        'exclude_numtweets',
        'exclude_avgtweetlength',
        'exclude_normalizedperiod',
        'exclude_normalizedsentiment',
        'exclude_all'
    ]

    if PREPROCESSING1:
        print("Loading and processing data in chunks...")
        df_original = load_and_process_data(TRAINING_DIR)
        
        print("Grouping data by time periods...")
        df_time_periods = group_time_periods(df_original)

        del df_original
        gc.collect()

        print("Computing and saving embeddings and masks...")
        compute_and_save_embeddings(df_time_periods, EMBEDDINGS_PATH, MASKS_PATH)
        
        print("Saving augmented DataFrame...")
        save_augmented_dataframe(df_time_periods, AUGMENTED_DF_PATH)
        
        del df_time_periods
        gc.collect()
        torch.cuda.empty_cache()
        print("Preprocessing1 completed successfully!")

    for scenario in ablation_scenarios:
        print(f"\n--- Starting Ablation Scenario: {scenario} ---")
        
        selected_features, additional_features_dim = get_additional_features_scenario(scenario)
        print(f"Selected additional features: {selected_features}")

        if PREPROCESSING2:
            print("Loading augmented DataFrame...")
            df_time_periods = pd.read_parquet(AUGMENTED_DF_PATH)

            print("Loading precomputed embeddings and masks...")
            with h5py.File(EMBEDDINGS_PATH, 'r') as emb_h5, h5py.File(MASKS_PATH, 'r') as mask_h5:
                embeddings = emb_h5['embeddings'][:] 
                masks = mask_h5['masks'][:] 

            print("Normalizing features based on scenario...")
            df_time_periods, normalization_params_path = normalize_features(df_time_periods, selected_features, scenario)

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
            if selected_features:
                additional_features = df_time_periods[selected_features].values
            else:
                additional_features = np.empty((len(df_time_periods), 0), dtype=np.float32)

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

            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )

            with open(normalization_params_path, 'r') as f:
                normalization_params = json.load(f)

            class_weights = class_weights.astype(np.float32)
            normalization_params['class_weights'] = class_weights.tolist()

            with open(normalization_params_path, 'w') as f:
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
                train_embeddings, train_masks, train_additional_features, train_labels, chunk_size=500
            )
            del train_embeddings, train_masks, train_labels, train_additional_features
            gc.collect()

            print("Creating HF validation dataset in chunks...")
            val_dataset = create_hf_dataset(
                val_embeddings, val_masks, val_additional_features, val_labels, chunk_size=500
            )
            del val_embeddings, val_masks, val_labels, val_additional_features
            gc.collect()

            dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

            processed_data_dir = os.path.join(DATA_DIR, f"{DATASET_NAME}_{scenario}")
            dataset_dict.save_to_disk(processed_data_dir)
            print(f"Processed DatasetDict for scenario '{scenario}' saved.")

        if not PREPROCESSING2:
            print("Loading processed dataset from disk...")
            processed_data_dir = os.path.join(DATA_DIR, DATASET_NAME)
            dataset_dict = load_from_disk(processed_data_dir)
            print("Dataset loaded successfully!")

        with open(normalization_params_path, 'r') as f:
            normalization_params = json.load(f)


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

        print("Initializing WandB...")
        wandb.login()

        run_name = f"{MODEL_NAME}-attention-FROZEN-{scenario}"
        wandb.init(
            project="ml-data-challenge-24",
            entity="ml-data-challenge-24",
            dir=os.path.join(DATA_DIR, "wandb"),
            resume="must" if CHECKPOINT is not None else "allow",
            id=WANDB_RUN_ID,
            name=run_name
        )

        with open(normalization_params_path, 'r') as f:
            normalization_params = json.load(f)
        wandb.config.update(normalization_params)

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
            additional_features_dim=additional_features_dim 
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        class_weights = torch.tensor(normalization_params['class_weights'], dtype=torch.float32).to(device)
        model = EnhancedPeriodClassifier(config, class_weights=class_weights)
        model.to(device)

        class WandbCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
                wandb.save(os.path.join(checkpoint_dir, "*"), base_path=args.output_dir)

        total_steps = (len(dataset_dict['train']) // BATCH_SIZE) * EPOCHS
        warmup_steps = int(0.1 * total_steps)
        print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

        training_args = TrainingArguments(
            output_dir=os.path.join(DATA_DIR, "MLDataChallenge2024"),
            evaluation_strategy="epoch",
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

        def custom_data_collator(features, additional_features_dim):
            embeddings = torch.tensor([f['embeddings'] for f in features], dtype=torch.float32) 
            masks = torch.tensor([f['masks'] for f in features], dtype=torch.float32)
            
            if additional_features_dim > 0:
                additional_features = torch.tensor([f['additional_features'] for f in features], dtype=torch.float32)
            else:
                additional_features = torch.empty((len(features), 0), dtype=torch.float32)
            
            labels = torch.tensor([f['labels'] for f in features], dtype=torch.long) 
            
            return {
                'embeddings': embeddings,
                'masks': masks,
                'additional_features': additional_features,
                'labels': labels
            }

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        callbacks = [WandbCheckpointCallback(), early_stopping_callback]

        def collator_wrapper(features):
            return custom_data_collator(features, additional_features_dim)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            compute_metrics=compute_metrics,
            data_collator=collator_wrapper,
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