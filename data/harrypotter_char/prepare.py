"""
Prepare the Harry Potter dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import argparse
from functools import reduce
import math

parser = argparse.ArgumentParser(description="A script that demonstrates argparse")
parser.add_argument("--training_type", type=str, default="scratch", help="argument to change parse method based on finetuning or training from scratch")
args = parser.parse_args()

input_files = ['Harry Potter and the Sorcerer\'s Stone.txt','Harry Potter and the Chamber of Secrets.txt',
               'Harry Potter and the Prisoner of Azkaban .txt','Harry Potter and the Goblet of Fire.txt',
               'Harry Potter and the Order of the Phoenix.txt','Harry Potter and The Half-Blood Prince.txt',
               'Harry Potter and the Deathly Hallows .txt']

data = []
for file_name in input_files:
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, "r") as file:
        file_content = file.read()
        data.append(file_content)

# Combine the content by joining the list with a separator (e.g., newline)
data = "\n".join(data)

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data_shakespeare = f.read()

if(args.training_type == "finetuning"):
    chars_shakespeare = sorted(list(set(data_shakespeare)))
    vocab_size_shakespeare = len(chars_shakespeare)
    data = list(filter(lambda x: x in chars_shakespeare, data))

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)

print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

chars_prob_list = []
for char in chars:
    chars_prob_list.append((1.0*data.count(char))/n)

entropy_of_training_set = reduce(lambda x, y: -x*math.log(x) - y*math.log(y), chars_prob_list)

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"entropy of training distribution is {entropy_of_training_set}")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
