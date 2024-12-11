from torch.utils.data import Dataset
import pandas as pd
import torch

# Dataset class for BertTweet mpodel
class TimePeriodDataset(Dataset):
    def __init__(self, tweets_list, labels, tokenizer, max_length=128, max_tweets=None):
        self.tweets_list = tweets_list 
        self.labels = labels.reset_index(drop=True).tolist()
        assert len(self.tweets_list) == len(self.labels), "Mismatch in tweets_list and labels lengths."
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_tweets = max_tweets 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        try:
            tweets = self.tweets_list[index]
            label = self.labels[index]

            if pd.isnull(label):
                raise ValueError(f"Label at index {index} is NaN or None.")
            label_tensor = torch.tensor(label, dtype=torch.long)
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
                input_ids = encoding['input_ids']
                attention_masks = encoding['attention_mask']
            else:
                encodings = [self.tokenizer(
                    tweet,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ) for tweet in tweets]
                input_ids = torch.stack([e['input_ids'].squeeze(0) for e in encodings])
                attention_masks = torch.stack([e['attention_mask'].squeeze(0) for e in encodings])

            return {
                'input_ids': input_ids,
                'attention_mask': attention_masks,
                'labels': label_tensor
            }
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            return {
                'input_ids': torch.zeros((1, self.max_length), dtype=torch.long),
                'attention_mask': torch.zeros((1, self.max_length), dtype=torch.long),
                'labels': torch.tensor(0, dtype=torch.long)
            }

class LongformerDataset(Dataset):
    def __init__(self, tweets_list, labels, tokenizer, max_length=4096):
        self.texts = [' '.join(tweets) for tweets in tweets_list]
        self.labels = labels.reset_index(drop=True).tolist()
        assert len(self.texts) == len(self.labels), "Mismatch in texts and labels lengths."
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }