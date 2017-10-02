from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers.core import Dropout
from keras.optimizers import Adam
import numpy as np


RUN_KEY = 'Model_Saves/Test'
SHUFFLE = True
SEQ_LENGTH = 64
BATCH_SIZE = 128

# Locate and read smiles strings from file
SMILES_PATH = 'smiles.txt'
# Will converge well before this point, so only read first 600000
smiles_list = list(set(open(SMILES_PATH).readlines()[0:600000]))

# Prepare one-hot char representation
smiles_text = [item for sublist in smiles_list for item in sublist]
chars = sorted(list(set(smiles_text)))
alphabet_size = len(chars)
char_indices = dict((char,idx) for idx, char in enumerate(chars))
char_vectors = dict((char,np.identity(alphabet_size)[idx]) for idx, char in enumerate(chars))
indices_char = dict((idx, char) for idx, char in enumerate(chars))

## TENSORBOARD
tb_callback = TensorBoard(log_dir='./{}/logs'.format(RUN_KEY), histogram_freq=1, write_graph=True, write_images=False)
## END TENSORBOARD

# Model setup (based on paper from MHS Segler Generating Focussed Molecule Libraries...)
# WHEN USING TENSORFLOW THE MODEL IS ALWAYS UNROLLED
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

model.add(Dense(alphabet_size, activation='softmax'))

# Use clipping norm of 5 to avoid exploding gradients
# Paper recommends default params, recorded here for clarity
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)


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

one_hot_generator =  one_hot_vector_gen(smiles_text, char_indices, SEQ_LENGTH)

# Generate validation generator from last third of data
validation_generator = one_hot_vector_gen(smiles_text[int(-1 * len(smiles_text)/3.0):-1], char_indices, SEQ_LENGTH)
validation_data = [next(validation_generator) for i in range(5120)]


for iteration in range(60):

    checkpoint_callback = ModelCheckpoint('./{}/'.format(RUN_KEY) + '{}.hdf5'.format(str(iteration)), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    if iteration % 5 and iteration >= 40:
        model.fit_generator(one_hot_generator, samples_per_epoch=BATCH_SIZE,
                            nb_epoch=1000,
                            callbacks=[checkpoint_callback],
                            verbose=1,
                            initial_epoch=iteration,
                            validation_data = validation_data)
    else:
        model.fit_generator(one_hot_generator, samples_per_epoch=BATCH_SIZE,
                            nb_epoch=1000,
                            verbose=0,
                            initial_epoch=iteration)
#                            validation_data = validation_data)
