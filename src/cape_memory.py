import random

import torch


class ReplayBuffer:
    def __init__(self, capacity=1024, seed=0):
        self.capacity = capacity
        self.items = []
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def add_batch(self, batch):
        batch_size = batch["labels"].shape[0]
        for idx in range(batch_size):
            self.seen += 1
            item = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    item[key] = value[idx].detach().cpu()
                elif isinstance(value, list):
                    item[key] = value[idx]
            if len(self.items) < self.capacity:
                self.items.append(item)
                continue
            replacement = self.rng.randrange(self.seen)
            if replacement < self.capacity:
                self.items[replacement] = item

    def sample(self, n):
        if not self.items:
            return None
        chosen = self.rng.sample(self.items, min(n, len(self.items)))
        out = {}
        for key in chosen[0]:
            values = [item[key] for item in chosen]
            out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
        return out


class FeatureReservoir:
    """Keyed reservoir for fixed-capacity calibration features.

    The reservoir stores one immutable CPU copy per sample id. Re-inserting a
    sample that has already been observed is a no-op, which prevents earlier
    validation stages from being over-counted when calibration is refitted.
    """

    def __init__(self, capacity, seed=0):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("FeatureReservoir capacity must be positive")
        self.capacity = capacity
        self.items = []
        self.seen = 0
        self.sample_ids = set()
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _freeze(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        return value

    def add(self, sample_id, **features):
        sample_id = str(sample_id)
        if sample_id in self.sample_ids:
            return False

        self.seen += 1
        item = {"sample_id": sample_id}
        item.update({key: self._freeze(value) for key, value in features.items()})

        if len(self.items) < self.capacity:
            self.items.append(item)
            self.sample_ids.add(sample_id)
            return True

        replacement = self.rng.randrange(self.seen)
        if replacement >= self.capacity:
            return False

        evicted_id = self.items[replacement]["sample_id"]
        self.sample_ids.remove(evicted_id)
        self.items[replacement] = item
        self.sample_ids.add(sample_id)
        return True

    def batches(self, batch_size):
        batch_size = max(1, int(batch_size))
        for start in range(0, len(self.items), batch_size):
            chunk = self.items[start : start + batch_size]
            out = {"sample_id": [item["sample_id"] for item in chunk]}
            keys = [key for key in chunk[0] if key != "sample_id"] if chunk else []
            for key in keys:
                values = [item[key] for item in chunk]
                out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
            yield out
