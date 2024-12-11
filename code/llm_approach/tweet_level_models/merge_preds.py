import pandas as pd
from collections import Counter
from tqdm import tqdm
import os

# script to combine predictions into a single file
predictions_dir = "../predictions"
input_file = '../predictions/predictions.csv'
output_file = '../predictions/majority_vote_predictions.csv' 

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

data = pd.read_csv(input_file)

def majority_vote(event_types):
    count = Counter(event_types)
    return count.most_common(1)[0][0]

majority_votes = data.groupby('ID')['EventType'].apply(majority_vote).reset_index()

majority_votes[['FirstPart', 'SecondPart']] = majority_votes['ID'].str.split('_', expand=True)
majority_votes['FirstPart'] = majority_votes['FirstPart'].astype(int)
majority_votes['SecondPart'] = majority_votes['SecondPart'].astype(int)

majority_votes = majority_votes.sort_values(by=['FirstPart', 'SecondPart'])
majority_votes = majority_votes.drop(columns=['FirstPart', 'SecondPart'])
majority_votes.to_csv(output_file, index=False)

print(f"Majority vote predictions saved to {output_file}")