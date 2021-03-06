import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import random
import numpy as np

import Skipgram
import custom_utils as utils

import os
import sys

import json

os.environ["CUDA_VISIBLE_DEVICES"]="3"
# constants
data_dir = sys.argv[1]
output_dir = sys.argv[2]
window_size = 8
embedding_size = 100
# batch_size
batch_size = 250

if torch.cuda.is_available():
    print('using gpu')
    device = 'cuda'
else:
    print('using cpu')
    device = 'cpu'

# get dataset
# data_set, vocabs, w2i, i2w = get_dataset(data_dir, min_frequency=8, threshhold=5000)
print("Loading dataset")
data_set, vocabs, w2i, i2w = utils.get_dataset(data_dir)
print("Loading successfully")

print("Preparing data for training")
training_data = utils.get_training_data(data_set, vocabs, w2i, i2w, window_size)
print("Prepare sucessfully")

print("Initialize model")
# training
losses = []
loss_function = nn.NLLLoss()
model = Skipgram.Model(len(vocabs), embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch = 20

print("Start training")
for e in range(epoch):
    total_loss = 0
    for center, context in utils.get_batch(training_data, batch_size):
        input_ = torch.LongTensor(center).to(device)
        label = torch.LongTensor(context).to(device)
        log_probs = model(input_)
        loss = loss_function(log_probs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss
    print('{}/{}'.format(e, epoch))
    losses.append(total_loss)

print(losses)
test_word = list(vocabs.keys())[1]
print(test_word)
test_word = w2i[test_word]
test_word = torch.tensor(test_word, dtype=torch.long).to(device)
print(test_word)
print(model.get_embedding(test_word))

# save vocabs
with open('vocabs.json', 'w') as f:
    json.dump(vocabs, f)
# save models

print("saving trained model")
torch.save(model.state_dict(), output_dir)
print("save successfully at {}".format(output_dir))
#torch.save(model, model_dir)
