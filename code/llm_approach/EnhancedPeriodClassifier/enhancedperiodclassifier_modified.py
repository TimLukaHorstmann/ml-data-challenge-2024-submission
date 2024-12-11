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
        dropout_prob=0.5,
        additional_features_dim=3,
        fc1_dim=4096,
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

        combined_dim = (self.embedding_dim * 3) + self.additional_features_dim

        self.dropout = nn.Dropout(p=config.dropout_prob)

        self.fc1 = nn.Linear(combined_dim, config.fc1_dim)
        self.ln1 = nn.LayerNorm(config.fc1_dim)
        self.dropout1 = nn.Dropout(p=config.dropout_prob)
        
        self.fc2 = nn.Linear(config.fc1_dim, config.fc1_dim)
        self.ln2 = nn.LayerNorm(config.fc1_dim)
        self.dropout2 = nn.Dropout(p=config.dropout_prob)
        
        self.fc3 = nn.Linear(config.fc1_dim, int(config.fc1_dim / 2))
        self.ln3 = nn.LayerNorm(int(config.fc1_dim / 2))
        self.dropout3 = nn.Dropout(p=config.dropout_prob)
        
        self.fc4 = nn.Linear(int(config.fc1_dim / 2), self.num_classes)

        self.activation = nn.GELU()

    def forward(self, embeddings, masks, additional_features=None, labels=None):
        key_padding_mask = ~masks.bool()
        attn_output, _ = self.attention(
            embeddings,  # query
            embeddings,  # key
            embeddings,  # value
            key_padding_mask=key_padding_mask
        )
        attn_output = attn_output + embeddings

        # Poolings
        attn_weights = torch.softmax(self.attention_pool(attn_output).squeeze(-1), dim=1)
        weighted_sum = torch.sum(attn_weights.unsqueeze(-1) * attn_output, dim=1)  

        mean_pool = torch.mean(attn_output, dim=1) 

        max_pool, _ = torch.max(attn_output, dim=1)  

        pooled = torch.cat([weighted_sum, mean_pool, max_pool], dim=-1)

        if additional_features is not None:
            combined_embedding = torch.cat([pooled, additional_features], dim=-1)  
        else:
            combined_embedding = pooled 

        x = self.dropout(combined_embedding)

        fc1_out = self.fc1(x)
        fc1_out = self.ln1(fc1_out)
        fc1_out = self.activation(fc1_out)
        fc1_out = self.dropout1(fc1_out)

        fc2_out = self.fc2(fc1_out)
        fc2_out = self.ln2(fc2_out)
        fc2_out = self.activation(fc2_out)
        fc2_out = self.dropout2(fc2_out)

        x = fc1_out + fc2_out 

        fc3_out = self.fc3(x)
        fc3_out = self.ln3(fc3_out)
        fc3_out = self.activation(fc3_out)
        fc3_out = self.dropout3(fc3_out)

        logits = self.fc4(fc3_out)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}