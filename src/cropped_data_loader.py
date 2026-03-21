import torch
import time
import os
import glob
import bisect
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChunkedTensorDataset(Dataset):
    """
    Dataset for chunked .pt files, where each file contains a tensor
    of shape [N, 3, 224, 224].

    Efficient behavior:
    - only one chunk is loaded at a time
    - global indexing is mapped to (chunk_id, local_idx)
    """

    def __init__(self, data_dir, pattern="*.pt", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if len(self.chunk_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {data_dir} matching {pattern}")

        self.chunk_lengths = []
        self.cumulative_sizes = []

        total = 0
        for f in self.chunk_files:
            x = torch.load(f, map_location="cpu")
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{f} does not contain a tensor")
            if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
                raise ValueError(f"{f} has shape {tuple(x.shape)}, expected [N, 3, 224, 224]")

            n = x.shape[0]
            self.chunk_lengths.append(n)
            total += n
            self.cumulative_sizes.append(total)

        self.total_len = total
        self._cached_chunk_idx = None
        self._cached_chunk = None

        print(f"[Dataset] Found {len(self.chunk_files)} chunk files")
        print(f"[Dataset] Total samples: {self.total_len}")

    def __len__(self):
        return self.total_len

    def _load_chunk(self, chunk_idx):
        if self._cached_chunk_idx != chunk_idx:
            self._cached_chunk = torch.load(
                self.chunk_files[chunk_idx],
                map_location="cpu"
            )
            self._cached_chunk_idx = chunk_idx
        return self._cached_chunk

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_len}")

        chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        prev_cum = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1]
        local_idx = idx - prev_cum

        chunk = self._load_chunk(chunk_idx)
        x = chunk[local_idx]   # [3,224,224], float32

        # used specifically for simclr
        if self.transform is not None:
            x1, x2 = self.transform(x)
            return x1, x2

        return x


def build_dataloader(
    data_dir,
    pattern="*.pt",
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    transform=None,
):
    dataset = ChunkedTensorDataset(
        data_dir=data_dir,
        pattern=pattern,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    return dataset, loader



if __name__ == "__main__":

    start = time.time()

    dataset, loader = build_dataloader(
        data_dir="Data/cropped/chunked",
        pattern="X_unlabeled_root_01_*.pt",   # only root 1
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    end = time.time()
    print(f"Total data loading time: {end - start:.3f} s")
    print("DataLoader built.")

    for i, batch in enumerate(loader):
        print(i, batch.shape, batch.dtype)
        batch = batch.to(DEVICE, non_blocking=True)
        if i == 4:
            break