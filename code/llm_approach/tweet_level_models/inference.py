from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import wandb

# Define constants
DATA_DIR = '/Data/tlh45/'
os.environ["HF_HOME"] = DATA_DIR
wandb_project = "ml-data-challenge-24"
wandb_entity = "ml-data-challenge-24"
wandb_run_id = "hekmj1mm"
checkpoint_name = "checkpoint-2500"  # Name of the checkpoint
best_model_dir = "MLDataChallenge2024/BestModel"  # Folder for the best model
MAX_LENGTH = 512
batch_size = 512
eval_dir = f"{DATA_DIR}/challenge_data/eval_tweets"
output_dir = "../predictions" 

# Define mode: "best_model" or "checkpoint"
mode = "best_model" 

if mode == "best_model":
    download_dir = os.path.join(DATA_DIR, f"BestModel-{wandb_run_id}")
    model_folder = best_model_dir
elif mode == "checkpoint":
    download_dir = os.path.join(DATA_DIR, f"Checkpoints-{wandb_run_id}")
    model_folder = checkpoint_name
else:
    raise ValueError("Invalid mode. Use 'best_model' or 'checkpoint'.")

checkpoint_dir = os.path.join(download_dir, model_folder)

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
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

print(f"Model loaded successfully from {checkpoint_dir}")

os.makedirs(output_dir, exist_ok=True)
def predict_event_type(texts, period_ids):
    texts_with_context = [f"Period: {pid}. {text}" for pid, text in zip(period_ids, texts)]
    inputs = tokenizer(
        texts_with_context,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    return predicted_classes.cpu().numpy(), probabilities.cpu().numpy()

for eval_file in tqdm(os.listdir(eval_dir), desc="Processing files"):
    if eval_file.endswith(".csv"):
        eval_path = os.path.join(eval_dir, eval_file)
        df = pd.read_csv(eval_path)

        predicted_classes = []
        probabilities = []
        for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {eval_file}"):
            batch_tweets = df['Tweet'][i:i + batch_size]
            batch_periods = df['PeriodID'][i:i + batch_size]
            batch_classes, batch_probs = predict_event_type(batch_tweets, batch_periods)
            predicted_classes.extend(batch_classes)
            probabilities.extend(batch_probs[:, 1]) 

        df['EventType'] = predicted_classes
        df['Probability_Class1'] = probabilities

        output_path = os.path.join(output_dir, f"predictions_{eval_file}")
        df[['ID', 'EventType']].to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")

predictions_dir = "../predictions"

def load_predictions(directory):
    all_data = []
    for filename in tqdm(os.listdir(directory), desc="Loading prediction files"):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

df_predictions = load_predictions(predictions_dir)
output_path = os.path.join(predictions_dir, "predictions.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_predictions.to_csv(output_path, index=False)

print(f"Combined predictions saved to {output_path}")
print(df_predictions['EventType'].value_counts())