from itertools import chain
import pickle

import transformers
from datasets import load_dataset


tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-base")
datasets = load_dataset("c4", "realnewslike", cache_dir="/home/kzhao/.cache/huggingface")


def tokenize_function(examples):
    return tokenizer(examples["text"], return_attention_mask=False, truncation=True)


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=16,
    remove_columns=['text', 'timestamp', 'url'],
)

expanded_inputs_length, targets_length = 141, 29  # TODO: don't hardcode


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= expanded_inputs_length:
        total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
        for k, t in concatenated_examples.items()
    }
    return result


processed_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=16,
)

with open("processed_realnewslike_bart.pkl", "wb") as f:
    pickle.dump(processed_datasets, f)
