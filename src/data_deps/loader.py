import math
import random
from typing import Dict, Iterator, List, Optional
 
import numpy as np
import torch
from torch.utils.data import Sampler
 
LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]

class HisemotionSampler(Sampler):
    def __init__(
        self,
        labels_array: np.ndarray,
        batch_size: int = 16,
        rare_threshold: int = 300,
        rare_slots: Optional[int] = None,
        max_repeat: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.labels_array   = labels_array
        self.batch_size     = batch_size
        self.rare_threshold = rare_threshold
        self.max_repeat     = max_repeat
        self.seed           = seed
        self.epoch          = 0
 
        self._build_pools()
        self.rare_slots   = rare_slots if rare_slots is not None else len(self.rare_labels)
        self.common_slots = batch_size - self.rare_slots
        self._compute_rounds()
        self._log()
 
    def _build_pools(self):
        self.pools: Dict[str, List[int]] = {}
        self.raw_sizes: Dict[str, int]   = {}
        n_cols = self.labels_array.shape[1]
        self.label_names = LABEL_COLS[:n_cols]  # Support 6 or 7 columns
        for i, label in enumerate(self.label_names):
            indices = np.where(self.labels_array[:, i] == 1)[0].tolist()
            self.pools[label]     = indices
            self.raw_sizes[label] = len(indices)
        self.rare_labels   = [l for l in self.label_names if self.raw_sizes[l] < self.rare_threshold]
        self.common_labels = [l for l in self.label_names if self.raw_sizes[l] >= self.rare_threshold]
 
    def _compute_rounds(self):
        max_raw = max(self.raw_sizes.values())
        self.target_sizes: Dict[str, int] = {}
        for label in self.rare_labels:
            raw    = self.raw_sizes[label]
            target = int(math.sqrt(max_raw * raw))
            target = min(target, raw * self.max_repeat)
            self.target_sizes[label] = target
        self.rounds     = min(self.target_sizes.values()) if self.target_sizes else 0
        self.total_size = self.rounds * self.batch_size
 
    def _log(self):
        print(f"\n{self.__class__.__name__}  rare_threshold={self.rare_threshold}  "
              f"rare_slots={self.rare_slots}  common_slots={self.common_slots}")
        print(f"  {'Label':<12} {'Type':<8} {'Raw':>6} {'Target':>8} {'Repeat':>8}")
        for label in self.label_names:
            raw = self.raw_sizes[label]
            if label in self.rare_labels:
                tgt = self.target_sizes[label]
                repeat_str = f"{tgt/raw:>7.1f}x" if raw > 0 else "    0.0x"
                print(f"  {label:<12} {'RARE':<8} {raw:>6} {tgt:>8} {repeat_str}")
            else:
                print(f"  {label:<12} {'COMMON':<8} {raw:>6} {'—':>8} {'random':>8}")
        print(f"  rounds={self.rounds}  total/epoch={self.total_size}\n")
 
    def set_epoch(self, epoch: int):
        self.epoch = epoch
 
    def __len__(self) -> int:
        return self.total_size
 
    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
 
        # ── Phase 1: Scale + shuffle rare pools ──────────────────────
        scaled: Dict[str, List[int]] = {}
        for label in self.rare_labels:
            pool, target = self.pools[label], self.target_sizes[label]
            repeated: List[int] = []
            while len(repeated) < target:
                chunk = pool.copy()
                rng.shuffle(chunk)
                repeated.extend(chunk)
            scaled[label] = repeated[:target]
 
        # ── Phase 2: Tính slots mỗi common label theo tỷ lệ ─────────
        total_common_raw = sum(self.raw_sizes[l] for l in self.common_labels)
        common_slots_per: Dict[str, int] = {}
        remaining = self.common_slots
        for i, label in enumerate(self.common_labels):
            if i == len(self.common_labels) - 1:
                common_slots_per[label] = max(1, remaining)
            else:
                slots = max(1, round(self.common_slots * self.raw_sizes[label] / total_common_raw))
                common_slots_per[label] = slots
                remaining -= slots
 
        # Chuẩn bị common pools đủ cho rounds × slots
        common_pools: Dict[str, List[int]] = {}
        for label in self.common_labels:
            needed = self.rounds * common_slots_per[label]
            pool   = self.pools[label].copy()
            rng.shuffle(pool)
            while len(pool) < needed:
                extra = self.pools[label].copy()
                rng.shuffle(extra)
                pool.extend(extra)
            common_pools[label] = pool[:needed]
 
        # ── Ghép 2 phase theo từng round ─────────────────────────────
        rare_ptrs   = {l: 0 for l in self.rare_labels}
        common_ptrs = {l: 0 for l in self.common_labels}
        result: List[int] = []
 
        for _ in range(self.rounds):
            round_batch: List[int] = []
            used: set = set()
 
            # Phase 1
            for label in self.rare_labels:
                ptr, pool = rare_ptrs[label], scaled[label]
                chosen = None
                for _ in range(5):
                    if ptr >= len(pool): break
                    c = pool[ptr]; ptr += 1
                    if c not in used: chosen = c; used.add(c); break
                if chosen is None and ptr > 0:
                    chosen = pool[ptr - 1]
                if chosen is not None:
                    round_batch.append(chosen)
                rare_ptrs[label] = ptr
 
            # Phase 2
            for label in self.common_labels:
                for _ in range(common_slots_per[label]):
                    ptr = common_ptrs[label]
                    if ptr < len(common_pools[label]):
                        round_batch.append(common_pools[label][ptr])
                        common_ptrs[label] += 1
 
            rng.shuffle(round_batch)
            result.extend(round_batch)
 
        return iter(result)