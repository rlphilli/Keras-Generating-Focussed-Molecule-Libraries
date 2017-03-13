from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

# Locate and read smiles strings from file
SMILES_PATH = 'smiles.txt'
smiles_text = open(SMILES_PATH).read()

# Prepare one-hot char representation
chars = sorted(list(set(smiles_text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Model setup (based on paper from MHS Segler Generating Focussed Molecule Libraries...)