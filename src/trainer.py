from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

from src.data_deps.loader import HisemotionSampler

class HisemotionTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, labels_array=None, sampler_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn      = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.labels_array = labels_array
        self.sampler_cfg  = sampler_cfg or {}
        self._sampler: Optional[HisemotionSampler] = None
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        loss    = self.loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
 
    def get_train_dataloader(self) -> DataLoader:
        if self.labels_array is None:
            return super().get_train_dataloader()
        if self._sampler is None:
            self._sampler = HisemotionSampler(
                labels_array   = self.labels_array,
                batch_size     = self.args.per_device_train_batch_size,
                rare_threshold = self.sampler_cfg.get("rare_threshold", 300),
                rare_slots     = self.sampler_cfg.get("rare_slots", None),
                max_repeat     = self.sampler_cfg.get("max_repeat", 5),
                seed           = self.sampler_cfg.get("seed", 42),
            )
        if hasattr(self, "state") and self.state is not None:
            self._sampler.set_epoch(int(self.state.epoch or 0))
        return DataLoader(
            self.train_dataset,
            batch_size  = self.args.per_device_train_batch_size,
            sampler     = self._sampler,
            collate_fn  = self.data_collator,
            drop_last   = self.args.dataloader_drop_last,
            num_workers = self.args.dataloader_num_workers,
            pin_memory  = self.args.dataloader_pin_memory,
        )