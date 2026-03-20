import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import numpy as np

from transformers import TrainingArguments

from src.trainer import HisemotionTrainer
from src.data_deps.loader import HisemotionSampler

class DummyModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        
    def forward(self, **kwargs):
        return {"logits": self.logits}



class TestHisemotionTrainer(unittest.TestCase):
    def setUp(self):
        self.dummy_args = TrainingArguments(output_dir="tmp/test_dir", per_device_train_batch_size=4)

    def test_compute_loss(self):
        mock_model = DummyModel(logits=torch.tensor([[0.5, 0.5]]))
        
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = torch.tensor(0.25)
        
        trainer = HisemotionTrainer(model=mock_model, args=self.dummy_args, loss_fn=mock_loss_fn)
        
        inputs = {"labels": torch.tensor([[1, 0]])}
        loss = trainer.compute_loss(mock_model, inputs)
        
        self.assertEqual(loss.item(), 0.25)
        
    def test_get_train_dataloader_with_sampler(self):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        
        labels_array = np.zeros((10, 7))
        labels_array[:, 0] = 1 
        
        trainer = HisemotionTrainer(
            model=DummyModel(logits=torch.tensor([[0.0]])),
            args=self.dummy_args,
            train_dataset=mock_dataset,
            data_collator=MagicMock(),
            labels_array=labels_array,
            sampler_cfg={"rare_threshold": 300, "rare_slots": 2, "max_repeat": 5}
        )
        
        dataloader = trainer.get_train_dataloader()
        
        self.assertIsNotNone(dataloader)
        self.assertIsInstance(dataloader.sampler, HisemotionSampler)
        self.assertEqual(dataloader.batch_size, 4)

    def test_get_train_dataloader_fallback(self):
        trainer = HisemotionTrainer(model=DummyModel(logits=torch.tensor([[0.0]])), args=self.dummy_args, train_dataset=MagicMock())
        trainer._get_train_sampler = MagicMock(return_value=None)
        
        dataloader = trainer.get_train_dataloader()
        self.assertIsNotNone(dataloader)
        self.assertNotIsInstance(dataloader.sampler, HisemotionSampler)

if __name__ == "__main__":
    unittest.main()
