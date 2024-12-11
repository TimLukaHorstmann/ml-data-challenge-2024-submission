from transformers import AutoModel, PreTrainedModel, PretrainedConfig
import torch.nn as nn
import torch

class PeriodClassifierConfig(PretrainedConfig):
    model_type = "period_classifier"

    def __init__(self, model_name="vinai/bertweet-base", hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_size = hidden_size

class PeriodClassifier(PreTrainedModel):
    config_class = PeriodClassifierConfig

    def __init__(self, config: PeriodClassifierConfig, class_weights=None):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = config.hidden_size
        self.attention_layer = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 2)
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask, tweet_mask=None, labels=None):
        batch_size, num_tweets, seq_length = input_ids.size()
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(batch_size, num_tweets, self.hidden_size)

        attention_weights = nn.functional.softmax(
            self.attention_layer(cls_embeddings).squeeze(-1), dim=1
        )  

        if tweet_mask is not None:
            attention_weights = attention_weights * tweet_mask 
            epsilon = 1e-8 
            attention_weights = attention_weights / (
                attention_weights.sum(dim=1, keepdim=True) + epsilon
            ) 

        weighted_embeddings = torch.bmm(
            attention_weights.unsqueeze(1), cls_embeddings
        ).squeeze(1)
        x = self.dropout(weighted_embeddings)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        logits = self.classifier(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}