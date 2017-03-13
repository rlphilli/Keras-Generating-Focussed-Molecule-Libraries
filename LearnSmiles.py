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

from shuffle_smiles import 
RUN_KEY = 'Test'
SHUFFLE = True

# Locate and read smiles strings from file
SMILES_PATH = 'smiles.txt'
smiles_text = open(SMILES_PATH).read()

# Prepare one-hot char representation
chars = sorted(list(set(smiles_text)))
alphabet_size = len(chars)
char_indices = dict((c, np.identity(alphabet_size)[i, :]) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Model setup (based on paper from MHS Segler Generating Focussed Molecule Libraries...)
model = Sequential()
Lambda(lambda x : char_indices[x], output_shape=(1, alphabet_size), arguments=None)
model.add(LSTM(1024, input_shape=(1, alphabet_size)))
model.add(Dropout(0.2))
model.add(LSTM(1024, input_shape=(1, alphabet_size)))
model.add(Dropout(0.2))
model.add(LSTM(1024, input_shape=(1, alphabet_size)))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# Use clipping norm of 5 to avoid exploding gradients
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


for i in range(100):