from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn
import torch

class EnhancedPeriodClassifierConfig(PretrainedConfig):
    model_type = "enhanced_period_classifier"
    
    def __init__(
        self,
        embedding_dim=768,
        num_classes=2,
        num_heads=8,
        dropout_prob=0.3,
        additional_features_dim=4,
        fc1_dim=2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.additional_features_dim = additional_features_dim
        self.fc1_dim = fc1_dim

class EnhancedPeriodClassifier(PreTrainedModel):
    config_class = EnhancedPeriodClassifierConfig

    def __init__(self, config: EnhancedPeriodClassifierConfig, class_weights=None):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.additional_features_dim = config.additional_features_dim
        self.class_weights = class_weights

        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of attention heads: {config.num_heads}")
        print(f"Dropout probability: {config.dropout_prob}")
        print(f"Additional features dimension: {self.additional_features_dim}")
        print(f"FC1 dimension: {config.fc1_dim}")

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.num_heads,
            batch_first=True
        )
        
        self.attention_pool = nn.Linear(self.embedding_dim, 1)
        
        combined_dim = self.embedding_dim + self.additional_features_dim
        
        self.dropout = nn.Dropout(p=config.dropout_prob)
        

        self.fc1 = nn.Linear(combined_dim, config.fc1_dim)
        self.ln1 = nn.LayerNorm(config.fc1_dim)

        self.fc2 = nn.Linear(config.fc1_dim, int(config.fc1_dim/2))
        self.ln2 = nn.LayerNorm(int(config.fc1_dim/2))

        self.fc3 = nn.Linear(int(config.fc1_dim/2), self.num_classes)

    def forward(self, embeddings, masks, additional_features=None, labels=None):
        key_padding_mask = ~masks.bool()

        attn_output, _ = self.attention(
            embeddings,  # query
            embeddings,  # key
            embeddings,  # value
            key_padding_mask=key_padding_mask
        ) 

        attn_weights = torch.softmax(self.attention_pool(attn_output).squeeze(-1), dim=1)
        
        weighted_sum = torch.sum(attn_weights.unsqueeze(-1) * attn_output, dim=1) 

        if additional_features is not None:
            combined_embedding = torch.cat([weighted_sum, additional_features], dim=-1)
        else:
            combined_embedding = weighted_sum 

        x = self.dropout(combined_embedding)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x) 

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}