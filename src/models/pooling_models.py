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
    """BERT + CLS & Mean Pooling Concatenation + Multi-label classifier.

    Combines two complementary representations:
    - [CLS] token: global sentence-level summary (what BERT was trained to encode)
    - Mean pool: token-level average (local/compositional information)

    Concatenated → 1536-dim → 2-layer MLP → logits.
    This is typically more stable and richer than single-head attention pooling.
    """

    def __init__(self, model_name: str, num_labels: int = 7, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        # Input: [CLS (768) || Mean (768)] = 1536
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.num_labels = num_labels

    def mean_pool(self, last_hidden_state, attention_mask):
        """Average hidden states, ignoring padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()       # (B, seq_len, 1)
        summed = (last_hidden_state * mask).sum(dim=1)    # (B, hidden)
        counts = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
        return summed / counts                            # (B, hidden)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state                          # (B, seq, 768)
        cls_out = h[:, 0, :]                                   # (B, 768)
        mean_out = self.mean_pool(h, attention_mask)           # (B, 768)
        combined = torch.cat([cls_out, mean_out], dim=-1)      # (B, 1536)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return {"logits": logits, "loss": torch.tensor(0.0, device=logits.device)}
