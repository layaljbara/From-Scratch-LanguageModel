# LSTMs for Next-Word Prediction in Cooking Recipes
# SciNet DAT112: Neural Network Programming
# Layal Jbara
#
# This script trains LSTMs to predict the next word
# in a corpus of cooking recipes. The model is trained using sequences of words
# as input and the following word as the target.

#####################################################################################################

"""
Train_Recipe_LSTM.py

This script implements the following main tasks:

1. **Data Preprocessing**:
   - Reads a text file containing cooking recipes.
   - Tokenizes the text while handling punctuation and special characters.
   - Converts text to lowercase and removes infrequent words.
   - Encodes words into one-hot vectors.
   - Prepares sequences of fixed length for RNN training.

2. **Model Definition**:
   - Defines a Sequential LSTM model for next-word prediction.
   - Uses a Dense output layer with softmax activation.

3. **Training & Saving**:
   - Trains the model for a specified number of epochs.
   - Saves the processed data, metadata, and the trained model for reuse.


"""

#####################################################################################################
# Import required libraries
#####################################################################################################
import os
import shelve
import numpy as np
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl

#####################################################################################################
# File and directory configuration
#####################################################################################################
datafile0   = 'allrecipes.txt'                      # Raw text file with recipes
datafile    = 'recipes.data'                        # Processed data filename prefix
shelvefile  = 'recipes.metadata.shelve'             # Metadata storage filename
modelfile   = 'recipes.model.keras'                 # Saved Keras model filename

#####################################################################################################
# Data preprocessing
#####################################################################################################

# If metadata doesn't already exist, process the raw data
if not os.path.isfile('data/' + shelvefile + '.dat'):

    # Load and read the raw text file
    print('Reading data.')
    with open(os.path.expanduser(datafile0), encoding="ISO-8859-1") as f:
        corpus0 = f.read()

    # Handle punctuation and special formatting
    corpus0 = corpus0.replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ')
    corpus0 = corpus0.replace('.', ' . ').replace(';', ' ; ').replace(':', ' : ')
    corpus0 = corpus0.replace('!', ' ! ').replace('?', ' ? ')
    corpus0 = corpus0.replace('\r\n', ' \n \n ').replace('\n', ' \n ')

    # Preserve multi-dash sequences
    corpus0 = corpus0.replace('--------------------------------------------------------', 'replaceme5')
    corpus0 = corpus0.replace('--------------------------------\n', 'replaceme4')
    corpus0 = corpus0.replace('------------', 'replaceme3')
    corpus0 = corpus0.replace('--------', 'replaceme2')
    corpus0 = corpus0.replace('--', 'replaceme1')

    # Separate single dashes
    corpus0 = corpus0.replace('-', ' - ')

    # Restore multi-dash sequences
    corpus0 = corpus0.replace('replaceme1', '--').replace('replaceme2', '--------')
    corpus0 = corpus0.replace('replaceme3', '------------')
    corpus0 = corpus0.replace('replaceme4', '--------------------------------')
    corpus0 = corpus0.replace('replaceme5', '--------------------------------------------------------')

    # Convert text to lowercase
    corpus0 = corpus0.lower()

    # Limit to first 700,000 words
    corpus0 = corpus0.split(' ')[0:700000]

    # Remove empty entries
    corpus = list(filter(lambda x: x != '', corpus0))
    print('Length of corpus is', len(corpus))

    # Identify unique words
    words = sorted(list(set(corpus)))
    num_words = len(words)
    print('We have', num_words, 'different words.')

    # Remove rare words (appear < 5 times)
    for word in words:
        if corpus.count(word) < 5:
            corpus = list(filter(lambda x: x != word, corpus))

    words = sorted(list(set(corpus)))
    num_words = len(words)
    print('We now have', num_words, 'different words.')

    # Encode words
    encoding = {w: i for i, w in enumerate(words)}
    decoding = {i: w for i, w in enumerate(words)}

    # Prepare sequences
    print('Processing data.')
    sentence_length = 50
    x_data, y_data = [], []
    for i in range(0, len(corpus) - sentence_length):
        sentence = corpus[i: i + sentence_length]
        next_word = corpus[i + sentence_length]
        x_data.append([encoding[word] for word in sentence])
        y_data.append(encoding[next_word])

    num_sentences = len(x_data)
    print('We have', num_sentences, 'sentences.')

    # One-hot encode data
    x = np.zeros((num_sentences, sentence_length, num_words), dtype=bool)
    y = np.zeros((num_sentences, num_words), dtype=bool)
    print('Encoding data.')
    for i, sentence in enumerate(x_data):
        for t, encoded_word in enumerate(sentence):
            x[i, t, encoded_word] = 1
        y[i, y_data[i]] = 1

    # Save processed data
    print('Saving processed data.')
    np.save('data/' + datafile + '.x.npy', x)
    np.save('data/' + datafile + '.y.npy', y)

    # Save metadata
    print('Creating metadata shelve file.')
    with shelve.open('data/' + shelvefile) as g:
        g['sentence_length'] = sentence_length
        g['num_words'] = num_words
        g['encoding'] = encoding
        g['decoding'] = decoding

else:
    # Load existing processed data
    print('Reading metadata shelve file.')
    with shelve.open('data/' + shelvefile, flag='r') as g:
        sentence_length = g['sentence_length']
        num_words = g['num_words']

    print('Reading processed data.')
    x = np.load('data/' + datafile + '.x.npy')
    y = np.load('data/' + datafile + '.y.npy')

#####################################################################################################
# Model definition
#####################################################################################################

if not os.path.isfile('data/' + modelfile):
    print('Building network.')
    model = km.Sequential()
    model.add(kl.LSTM(256, input_shape=(sentence_length, num_words)))
    model.add(kl.Dense(num_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    print('Reading model file.')
    model = km.load_model('data/' + modelfile)

#####################################################################################################
# Model training
#####################################################################################################

print('Beginning fit.')
fit = model.fit(x, y, epochs=200, batch_size=128, verbose=2)

# Save trained model
model.save('data/' + modelfile)


