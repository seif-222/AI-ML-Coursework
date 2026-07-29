import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

local_dir = '../edu_fineweb10B'
remote_name = 'sample-10BT'
shard_size = int(1e8) # 100M Tokens

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

hf_data = load_dataset('HuggingFaceFW/fineweb-edu', name=remote_name, split='train', streaming=True)


enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>'] # Getting the id of the eot token

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text'])) # encode_ordinary encodes but treats the special tokens as normal tokens and igonre the special tokens | ['text'] is to index into the HF data dictionary
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'Dictionary is too large for uint16'
    return tokens_np.astype(np.uint16)



def write_datafile(filename, token_np):
    np.save(filename, token_np)


nprocs = max(1, os.cpu_count()//2)

# mp.Pool(nprocs) — creates a pool of nprocs worker processes, each a full independent Python interpreter
# with ... as pool — context manager, automatically terminates and cleans up the worker processes when the block exits, even if an error occurs

with mp.Pool(processes=nprocs) as pool:
    # np.empty((shard_size,), dtype=np.uint16) — allocates a 1D array of 100M uint16 slots WITHOUT initializing the values. Faster than p.zeros because it skips zeroing memory. The values are garbage until you write into them
    # (shard_size,) — a tuple with one element, defining the shape as 1-dimensional with shard_size elements
    shard_idx = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None


    # pool.imap(function, iterable, chunksize=16) — distributes elements of fw across worker processes, each worker calls tokenize(doc) on its assigned docs.
    # -> Returns results to the main process in the original order (unlike imap_unordered which returns as soon as each finishes regardless of order)
    # chunksize=16 — each worker grabs 16 documents at a time before asking for more, reducing the overhead of constant inter-process communication
    # Each iteration, tokens is one numpy uint16 array returned by tokenize, the tokens for one document

    for tokens in pool.imap(tokenize, hf_data, chunksize=16):

        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count: token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None: progress_bar = tqdm(total=shard_size, unit='Tokens', desc=f'shard {shard_idx}')
            progress_bar.update(len(tokens))
        else:
            split = 'Val' if shard_idx == 0 else 'Train'
            filename = os.path.join(DATA_CACHE_DIR, f'Shard_{split}-{shard_idx:05d}')
            remainder = shard_size - token_count
            all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
            progress_bar.update(remainder)
            write_datafile(filename, all_tokens_np)
            shard_idx += 1
            progress_bar = None
            if shard_idx >= 2: break
            # Populate the next shard with the leftovers
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remainings as last shard
    if token_count != 0:
        split = "Val" if shard_idx == 0 else "Train"
        filename = os.path.join(DATA_CACHE_DIR, f"Shard_{split}-{shard_idx:05d}")
        write_datafile(filename, all_tokens_np[:token_count])






