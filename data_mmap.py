#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory-mapped dataset for Megatron-LM preprocessed .bin/.idx shards.

Replaces data.py (streaming .jsonl.zst) when using pre-tokenized Dolma shards.
Supports multi-rank, multi-worker sharding for FSDP training.
"""
import glob
import os
import struct
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset


# Megatron .idx dtype codes (from megatron/core/datasets/indexed_dataset.py)
_MEGATRON_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}
_KNOWN_MAGICS = {
    b'\x00MMIDX\x00\x00': 9,   # older Megatron-LM
    b'MMIDIDX\x00\x00': 9,     # newer Megatron-LM / ReMoE fork
}


def _read_dtype_from_idx(idx_path: str) -> np.dtype:
    """Read the token dtype from a Megatron .idx header."""
    with open(idx_path, 'rb') as f:
        magic = f.read(9)
        if magic not in _KNOWN_MAGICS:
            raise ValueError(f"Bad .idx magic in {idx_path}: {magic!r}")
        _version = struct.unpack('<Q', f.read(8))[0]
        dtype_code = struct.unpack('B', f.read(1))[0]
    return _MEGATRON_DTYPES.get(dtype_code, np.uint16)


def _load_shard_tokens(bin_path: str, idx_path: str) -> np.ndarray:
    """Load all tokens from a single .bin shard as int64 numpy array."""
    dtype = _read_dtype_from_idx(idx_path)
    data = np.fromfile(bin_path, dtype=dtype)
    return data.astype(np.int64)


def discover_shards(data_root: str):
    """Find all .bin/.idx pairs in data_root. Returns sorted list of .bin paths."""
    bins = sorted(glob.glob(os.path.join(data_root, "*_text_document.bin")))
    # Verify each .bin has a matching .idx
    valid = []
    for b in bins:
        idx = b.replace(".bin", ".idx")
        if os.path.isfile(idx):
            valid.append(b)
    return valid


class MMapTokenDataset(IterableDataset):
    """
    Infinite streaming dataset over pre-tokenized Megatron .bin/.idx shards.

    Each shard is loaded fully into memory (typically ~2-3 MB each),
    tokens are packed into fixed-length seq_len chunks.
    Shards are distributed across (rank, worker) pairs for parallel I/O.
    """

    def __init__(
        self,
        data_root: str,
        seq_len: int = 1024,
        rank: int = 0,
        world_size: int = 1,
        infinite: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = data_root
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.infinite = infinite
        self.seed = seed

        self.all_shards = discover_shards(data_root)
        assert len(self.all_shards) > 0, f"No *_text_document.bin found in {data_root}"

    def _split_for_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid, num_workers = 0, 1
        else:
            wid, num_workers = worker_info.id, worker_info.num_workers
        global_id = self.rank * num_workers + wid
        global_count = self.world_size * num_workers
        my_shards = self.all_shards[global_id::global_count]
        return my_shards, global_id

    def __iter__(self):
        my_shards, gid = self._split_for_worker()
        if not my_shards:
            return

        import random
        rng = random.Random(self.seed + gid)
        seq_len = self.seq_len
        buf = []

        while True:
            order = list(my_shards)
            rng.shuffle(order)

            for bin_path in order:
                idx_path = bin_path.replace(".bin", ".idx")
                try:
                    tokens = _load_shard_tokens(bin_path, idx_path)
                except Exception as e:
                    print(f"[data_mmap] WARNING: skipping {bin_path}: {e}")
                    continue

                buf.extend(tokens.tolist())

                while len(buf) >= seq_len:
                    chunk = buf[:seq_len]
                    del buf[:seq_len]
                    t = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": t, "labels": t.clone()}

            if not self.infinite:
                break