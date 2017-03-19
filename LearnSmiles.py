from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dropout
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import numpy as np
from random import shuffle


from shuffle_smiles import shuffle_lines, fisherYatesShuffle

RUN_KEY = 'Model_Saves/Test'
SHUFFLE = True
SEQ_LENGTH = 64
BATCH_SIZE = 128

# Locate and read smiles strings from file
SMILES_PATH = 'smiles.txt'
smiles_list = list(set(open(SMILES_PATH).readlines()[1:600000]))

# Prepare one-hot char representation
smiles_text = [item for sublist in smiles_list for item in sublist]
chars = sorted(list(set(smiles_text)))
alphabet_size = len(chars)
char_indices = dict((char,idx) for idx, char in enumerate(chars))
char_vectors = dict((char,np.identity(alphabet_size)[idx]) for idx, char in enumerate(chars))
indices_char = dict((idx, char) for idx, char in enumerate(chars))

# Vectorize
def one_hot_vectorize(smiles_text, char_indices, seq_length=25):
    """Take the text, predetermined char_indices and sequence length and return a list or vectors of the appropriate one-hot representation"""
    inputs = []
    targets = []
    # while p < len(smiles_text) + seq_length:
    #     inputs.append([char_indices[char] for char in smiles_text[p:p+seq_length]])
    #     targets.append([char_indices[char] for char in smiles_text[p+1:p+seq_length+1]])
    #     p += seq_length

    for p in range(0, len(smiles_text) - seq_length):
        input_seq = smiles_text[p:p + seq_length]
        output_seq = smiles_text[p + seq_length]
        inputs.append([char_indices[char] for char in input_seq])
        targets.append([char_indices[char] for char in output_seq])

    X = pad_sequences(inputs, maxlen=seq_length)
    inputs = np.reshape(inputs,(X.shape[0], seq_length, 1))

    return inputs, np_utils.to_categorical(targets)


## TENSORBOARD
tb_callback = TensorBoard(log_dir='./{}/logs'.format(RUN_KEY), histogram_freq=1, write_graph=True, write_images=False)
## END TENSORBOARD

# Model setup (based on paper from MHS Segler Generating Focussed Molecule Libraries...)
# TODO WHEN USING TENSORFLOW THE MODEL IS ALWAYS UNROLLED
model = Sequential()
model.add(LSTM(1024, return_sequences=True,
               stateful=True,
               batch_input_shape=(BATCH_SIZE, SEQ_LENGTH, alphabet_size), unroll=True))
model.add(Dropout(0.2))
model.add(LSTM(1024, return_sequences=True,
               stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(1024, stateful=True))
model.add(Dropout(0.2))

# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))

model.add(Dense(alphabet_size, activation='softmax'))

# Use clipping norm of 5 to avoid exploding gradients
# Paper recommends default params, recorded here for clarity
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)

from random import randint
def one_hot_vector_gen(smiles_text, char_indices, seq_length=25):
    """Take the text, predetermined char_indices and sequence length and return a list or vectors of the appropriate one-hot representation"""
    inputs = []
    targets = []
    for p in range(0, len(smiles_text) - seq_length):
        input_seq = smiles_text[p:p + seq_length]
        output_seq = smiles_text[p + seq_length]
        inputs.append([char_vectors[char] for char in input_seq])
        targets.append(char_vectors[output_seq])
        if len(targets) >= BATCH_SIZE and len(targets) % BATCH_SIZE == 0:
            yield (np.reshape(np.array(inputs, dtype=np.float32), (BATCH_SIZE, SEQ_LENGTH, alphabet_size)),np.array(targets, dtype=np.int8))
            inputs, targets = [], []
        # yield (np.array([char_vectors[char] for char in input_seq], dtype=int), char_indices[output_seq])


if not SHUFFLE:
    inputs, targets = one_hot_vectorize(smiles_text, char_indices, seq_length=SEQ_LENGTH)
else:
    inputs, targets = None, None

if SHUFFLE:
    # shuffle(smiles_list)
    pass

one_hot_generator =  one_hot_vector_gen(smiles_text, char_indices, SEQ_LENGTH)

for iteration in range(25):


    checkpoint_callback = ModelCheckpoint('./{}/'.format(RUN_KEY) + '{}.hdf5'.format(str(iteration)), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    if iteration % 1 == 0 and (iteration < 5 or iteration % 4 == 0):
        model.fit_generator(one_hot_generator, samples_per_epoch=BATCH_SIZE,
                  nb_epoch=1000,
                  callbacks=[checkpoint_callback, tb_callback],
                  verbose=1,
                  initial_epoch=iteration)
    else:
        model.fit_generator(one_hot_generator, samples_per_epoch=BATCH_SIZE,
                  nb_epoch=1000,
                  callbacks=[tb_callback],
                  verbose=1,
                  initial_epoch=iteration)
