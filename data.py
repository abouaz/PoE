#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming dolma3 tokenization dataset for multi-node FSDP training.

Streams .jsonl.zst files directly -> tokenize -> pack into seq_len chunks.
No offline preprocessing needed.
"""
import glob
import io
import json
import os
import sys
from typing import Optional, Sequence, Iterator

import torch
from torch.utils.data import IterableDataset

import zstandard as zstd


DEFAULT_ROOT = "/apdcephfs_hldy/share_304318596/nlperyin/dolma3_dolmino_mix-100B-1025"
_DEFAULT_KEEP = ("text",)


def _iter_shard(path: str, keep_keys=_DEFAULT_KEEP) -> Iterator[dict]:
    """Yield records from a single .jsonl.zst shard."""
    dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        text_stream = io.TextIOWrapper(r, encoding="utf-8")
        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if keep_keys is not None:
                obj = {k: obj[k] for k in keep_keys if k in obj}
            yield obj


def _iter_subset(subset_dir: str, **kwargs) -> Iterator[dict]:
    shards = sorted(glob.glob(os.path.join(subset_dir, "*.jsonl.zst")))
    for sh in shards:
        yield from _iter_shard(sh, **kwargs)


def list_subsets(root: str = DEFAULT_ROOT) -> list:
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


class StreamingTokenDataset(IterableDataset):
    """
    Infinite streaming token dataset with cross-(rank, worker) sharding.

    Each document is tokenized on-the-fly, separated by EOS, and packed
    into fixed-length seq_len chunks.
    """

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        tokenizer=None,
        seq_len: int = 2048,
        rank: int = 0,
        world_size: int = 1,
        subsets: Optional[Sequence[str]] = None,
        infinite: bool = True,
        shuffle_subsets_seed: Optional[int] = 42,
    ):
        super().__init__()
        assert tokenizer is not None, "tokenizer is required"
        self.root = root
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.infinite = infinite
        self.shuffle_subsets_seed = shuffle_subsets_seed
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            self.eos_id = tokenizer.pad_token_id
        assert self.eos_id is not None, "tokenizer needs eos_token_id or pad_token_id"

        all_subsets = list_subsets(root)
        if subsets:
            self.subsets = [s for s in subsets if s in all_subsets]
        else:
            self.subsets = all_subsets

    def _split_for_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid, num_workers = 0, 1
        else:
            wid, num_workers = worker_info.id, worker_info.num_workers
        global_id = self.rank * num_workers + wid
        global_count = self.world_size * num_workers
        return self.subsets[global_id::global_count], global_id

    def __iter__(self):
        my_subsets, gid = self._split_for_worker()
        if not my_subsets:
            return
        seq_len = self.seq_len
        eos = self.eos_id

        rng = None
        if self.shuffle_subsets_seed is not None:
            import random
            rng = random.Random(self.shuffle_subsets_seed + gid)

        buf = []
        while True:
            order = list(my_subsets)
            if rng is not None:
                rng.shuffle(order)
            for subset_name in order:
                subset_dir = os.path.join(self.root, "data", subset_name)
                if not os.path.isdir(subset_dir):
                    continue
                for ex in _iter_subset(subset_dir, keep_keys=_DEFAULT_KEEP):
                    text = ex.get("text")
                    if not text:
                        continue
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    buf.extend(ids)
                    buf.append(eos)
                    while len(buf) >= seq_len:
                        chunk = buf[:seq_len]
                        del buf[:seq_len]
                        t = torch.tensor(chunk, dtype=torch.long)
                        yield {"input_ids": t, "labels": t.clone()}
            if not self.infinite:
                break
