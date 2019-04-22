from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

### Preparing the data
all_letters = string.ascii_letters + " .,;'-" # Collection of ASCII characters
n_letters = len(all_letters) + 1 # Add one for EOS marker

def find_files(path):
    return glob.glob(path)

def unicode_to_ascii(s):
    ascii = []

    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn' and c in all_letters:
            ascii.append(c)

# Read file, split into lines
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

# Build category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
data_glob = 'data/names/*.txt'
for filename in find_files(data_glob):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found.')

print('Num. categories:', n_categories, all_categories)


### Creating the network
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # Input, hidden, output
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, category, inputs, hidden):
        input_combined = torch.cat((category, inputs, hidden), 1)
        hidden = self.i2h(input_combined)
        i2o = self.i2o(input_combined)
        output_combined = torch.cat((hidden, i2o), 1)
        o2o = self.o2o(output_combined)
        dropout = self.dropout(o2o)
        output = self.softmax(dropout)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


### Prepare for training
import random

def random_choice(l):
    # Return random item from list
    rand_idx = random.randint(0, len(l)-1)
    return l[rand_idx]

def random_training_pair():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line

"""
For each timestep, the inputs of the network contains:
    category, current letter, hidden state

and the output consists of:
    next letter, next hidden state
"""

# One-hot vector for category
def category_tensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1

    return tensor

# One-hot matrix of first to last letters, excluding EOS, for input
def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1

    return tensor

# LongTensor of second letter to end (EOS) for target
def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS

    return torch.LongTensor(letter_indexes)


# Helper: create a random training example
def random_training_example():
    category, line = random_training_pair()
    rand_category_tensor = category_tensor(category)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)

    return rand_category_tensor, input_line_tensor, target_line_tensor


### Training the network
criterion = nn.NLLLoss() # Negative log likelihood loss
# TODO: replace w MSELoss for single category ?

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

# Helper: timekeep
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' % (m, s)

### Training
rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

print('Training model...')
for iter in range(1, n_iters + 1):
    output, loss = train(*random_training_example())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

### Save model after training
torch.save(rnn, 'language_names.pt')

### Generating samples
max_length = 50

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():
        sample_category_tensor = category_tensor(category)
        sample_input = input_tensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(sample_category_tensor, sample_input[0], hidden)
            top_v, top_i = output.topk(1)
            top_i = top_i[0][0]

            if top_i == n_letters-1: # If EOS
                break
            else: # Else, append next letter to output
                letter = all_letters[top_i]
                output_name += letter

        return output_name

# Generate multiple samples
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')
