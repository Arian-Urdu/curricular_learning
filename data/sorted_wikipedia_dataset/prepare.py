import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

# Number of workers in .map() call NOT USED
num_proc = 8

# Number of workers in load_dataset() call
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("wikipedia", "20220301.simple")

    # Splitting the dataset
    split_dataset = dataset["train"].train_test_split(test_size=0.5, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename the test split to val

    # Function to calculate the length of the text
    def calculate_text_length(example):
        return len(example['text'])

    # Sort the training dataset based on the length of the 'text'
    sorted_train_dataset = sorted(split_dataset['train'], key=calculate_text_length)
    sorted_val_dataset = sorted(split_dataset['val'], key=calculate_text_length)

    # Convert the sorted lists back to dataset objects
    sorted_train_dataset = Dataset.from_dict({key: [example[key] for example in sorted_train_dataset] for key in split_dataset['train'].column_names})
    sorted_val_dataset = Dataset.from_dict({key: [example[key] for example in sorted_val_dataset] for key in split_dataset['val'].column_names})

    # Define the number of shards
    num_shards = 5

    # Calculate the number of examples per shard
    examples_per_shard = len(sorted_train_dataset) // num_shards

    # Tokenization using the GPT-2 BPE encoding
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize each shard and save
    for shard_index in range(num_shards):
        start_idx = shard_index * examples_per_shard
        end_idx = (shard_index + 1) * examples_per_shard if shard_index < num_shards - 1 else len(sorted_train_dataset)

        shard_dataset = sorted_train_dataset.select(range(start_idx, end_idx))
        tokenized_shard = shard_dataset.map(
            process,
            remove_columns=['text'],
            desc=f"Tokenizing shard {shard_index + 1}"
        )

        # Concatenate all the ids in the shard into one large file
        arr_len = np.sum(tokenized_shard['len'], dtype=np.uint64)
        filename = f'curriculum_shard_{shard_index + 1}.bin'
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            batch = tokenized_shard.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Tokenize the validation set and save
    tokenized_val = sorted_val_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing validation set"
    )

    # Concatenate all the ids in the validation set into one large file
    arr_len_val = np.sum(tokenized_val['len'], dtype=np.uint64)
    filename_val = 'sorted_wikipedia_val_dataset.bin'
    dtype_val = np.uint16
    arr_val = np.memmap(filename_val, dtype=dtype_val, mode='w+', shape=(arr_len_val,))
    total_batches_val = 1024

    idx_val = 0
    for batch_idx in tqdm(range(total_batches_val), desc=f'Writing {filename_val}'):
        batch_val = tokenized_val.shard(num_shards=total_batches_val, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch_val = np.concatenate(batch_val['ids'])
        arr_val[idx_val: idx_val + len(arr_batch_val)] = arr_batch_val
        idx_val += len(arr_batch_val)
    arr_val.flush()
