"""
Custom pooling models for HISEMOTIONS multi-label classification.

Three strategies:
  1. MeanPoolingClassifier   — average all token hidden states
  2. AttentionPoolingClassifier — learnable weighted average
  3. Sentence-Pair uses standard AutoModelForSequenceClassification(num_labels=2)
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class MeanPoolingClassifier(nn.Module):
    """BERT + Mean Pooling + Multi-label classifier.
    
    Instead of using only [CLS], averages ALL token hidden states
    (excluding padding) for a richer sentence representation.
    """

    def __init__(self, model_name: str, num_labels: int = 7, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def mean_pool(self, last_hidden_state, attention_mask):
        """Average hidden states, ignoring padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()       # (B, seq_len, 1)
        summed = (last_hidden_state * mask).sum(dim=1)    # (B, hidden)
        counts = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
        return summed / counts                            # (B, hidden)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits, "loss": torch.tensor(0.0, device=logits.device)}


class AttentionPoolingClassifier(nn.Module):
    """BERT + Multi-head Attention Pooling + Multi-label classifier.

    Architecture:
    1. Run BERT → hidden states (B, seq, 768)
    2. Use [CLS] as query, all tokens as key/value → MHA → attended vector (B, 768)
    3. Residual: attended + mean_pool → richer representation
    4. LayerNorm → 2-layer MLP → logits

    Why better than single-head / CLS+Mean concat:
    - [CLS] query learns WHICH tokens are emotion-relevant (task-specific)
    - Multi-head allows attending to multiple aspects simultaneously (8 heads)
    - Residual mean_pool acts as a safety net: if attention underfits, mean keeps signal
    - LayerNorm stabilizes training
    """

    def __init__(self, model_name: str, num_labels: int = 7, dropout: float = 0.1,
                 num_heads: int = 8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )
        self.num_labels = num_labels

    def mean_pool(self, last_hidden_state, attention_mask):
        """Average hidden states, ignoring padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # (B, hidden)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state                           # (B, seq, 768)
        cls = h[:, :1, :]                                       # (B, 1, 768) — query
        key_padding_mask = (attention_mask == 0)                # True = ignore padding

        # MHA: [CLS] attends over all tokens
        attended, _ = self.mha(
            query=cls,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
        )
        attended = attended.squeeze(1)                          # (B, 768)

        # Residual: add mean pool for stability
        mean_out = self.mean_pool(h, attention_mask)            # (B, 768)
        pooled = self.layer_norm(attended + mean_out)           # (B, 768)

        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits, "loss": torch.tensor(0.0, device=logits.device)}

