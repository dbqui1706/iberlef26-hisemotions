from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

from src.data_deps.loader import HisemotionSampler
class FGM:
    """Perturb word embeddings along gradient direction to create adversarial examples.
    
    Usage:
        fgm = FGM(model, epsilon=1.0)
        # after loss.backward():
        fgm.attack()           # perturb embeddings
        loss_adv = forward()   # compute adversarial loss
        loss_adv.backward()    # accumulate adversarial gradients
        fgm.restore()          # restore original embeddings
    """
    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class HisemotionTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, labels_array=None, sampler_cfg=None,
                 use_fgm=False, fgm_epsilon=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn      = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.labels_array = labels_array
        self.sampler_cfg  = sampler_cfg or {}
        self._sampler: Optional[HisemotionSampler] = None
        self.use_fgm      = use_fgm
        self.fgm_epsilon   = fgm_epsilon
        self._fgm: Optional[FGM] = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        loss    = self.loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to add FGM adversarial attack."""
        if not self.use_fgm:
            return super().training_step(model, inputs, num_items_in_batch)

        # Lazy init FGM
        if self._fgm is None:
            self._fgm = FGM(model, epsilon=self.fgm_epsilon)
            print(f"  FGM adversarial training enabled (epsilon={self.fgm_epsilon})")

        model.train()
        inputs = self._prepare_inputs(inputs)

        # Step 1: Normal forward + backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)

        # Step 2: Adversarial forward + backward
        self._fgm.attack()
        with self.compute_loss_context_manager():
            loss_adv = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_adv = loss_adv.mean()
        self.accelerator.backward(loss_adv)
        self._fgm.restore()

        return loss.detach() / self.args.gradient_accumulation_steps

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