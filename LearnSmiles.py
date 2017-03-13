from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers.core import Lambda, Dropout
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os

from shuffle_smiles import shuffle_lines, fisherYatesShuffle

RUN_KEY = 'Test'
SHUFFLE = True
SEQ_LENGTH = 64

# Locate and read smiles strings from file
SMILES_PATH = 'smiles.txt'
smiles_list = list(set(open(SMILES_PATH).readlines()))

# Prepare one-hot char representation
smiles_text = [item for sublist in smiles_list for item in sublist]
chars = sorted(list(set(smiles_text)))
alphabet_size = len(chars)
char_indices = dict((char, idx) for idx, char in enumerate(chars))
indices_char = dict((idx, char) for idx, char in enumerate(chars))

# Vectorize
def one_hot_vectorize(smiles_text, char_indices, seq_length=25):
    """Take the text, predetermined char_indices and sequence length and return a list or vectors of the appropriate one-hot representation"""
    p = 0
    inputs = []
    targets = []
    while p < len(smiles_text) + seq_length:
        inputs += [char_indices[char] for char in smiles_text[p:p+seq_length]]
        targets += [char_indices[char] for char in char_indices[p+1:p+seq_length+1]]
        p += seq_length
    return inputs, targets


# Model setup (based on paper from MHS Segler Generating Focussed Molecule Libraries...)
model = Sequential()
model.add(LSTM(1024, input_shape=(1, alphabet_size)))
model.add(Dropout(0.2))
model.add(LSTM(1024, input_shape=(1, alphabet_size)))
model.add(Dropout(0.2))
model.add(LSTM(1024, input_shape=(1, alphabet_size)))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# Use clipping norm of 5 to avoid exploding gradients
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5)
model.compile(loss='cross_entropy')

for iteration in range(2):
    model.fit(list(smiles_text), [0] + list(smiles_text), batch_size=128, nb_epoch=1)
