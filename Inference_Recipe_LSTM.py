# LSTMs for Next-Word Prediction in Cooking Recipes
# SciNet DAT112: Neural Network Programming 
# Layal Jbara
#
# This script loads a trained LSTM model for next-word prediction
# on cooking recipes, seeds it with a random starting sequence, and
# generates a sequence of words to form a recipe-like text.


#####################################################################################################

"""
Inference_Recipe_LSTM.py

Main tasks:
1) Load trained model and metadata
   - Model architecture and weights from Keras `.keras` file
   - Vocabulary encoding/decoding and sentence length from shelve

2) Generate text
   - Seed with a random sequence of `sentence_length` words
   - Iteratively predict the most probable next word
   - Append predicted word to sequence, drop oldest word to keep length constant
   - Repeat for desired number of generated words

3) Output generated recipe text to console
"""

#####################################################################################################
# Import required libraries
#####################################################################################################

import os
import random
import shelve
import numpy as np
import tensorflow.keras.models as km

#####################################################################################################
# File configuration
#####################################################################################################

SHELVE_FILE = 'recipes.metadata.shelve'
MODEL_FILE  = 'recipes.model.keras'
DATA_DIR    = 'data'

#####################################################################################################
# Load model and metadata
#####################################################################################################

print('Loading trained model...')
model = km.load_model(os.path.join(DATA_DIR, MODEL_FILE))

print('Loading metadata...')
with shelve.open(os.path.join(DATA_DIR, SHELVE_FILE), flag='r') as g:
    sentence_length = g['sentence_length']
    num_words       = g['num_words']
    encoding        = g['encoding']
    decoding        = g['decoding']

#####################################################################################################
# Create random seed sequence
#####################################################################################################

seed_words = [decoding[random.randint(0, num_words - 1)] for _ in range(sentence_length)]

# One-hot encode the seed
x = np.zeros((1, sentence_length, num_words), dtype=bool)
for i, word in enumerate(seed_words):
    x[0, i, encoding[word]] = 1

#####################################################################################################
# Generate text
#####################################################################################################

generated_text = []
num_generated_words = 2000

for _ in range(num_generated_words):
    # Predict next word ID
    pred_id = np.argmax(model.predict(x, verbose=0))

    # Append predicted word to output
    generated_text.append(decoding[pred_id])

    # One-hot encode the predicted word
    next_word_vec = np.zeros((1, 1, num_words), dtype=bool)
    next_word_vec[0, 0, pred_id] = 1

    # Slide window: drop first word, append new word
    x = np.concatenate((x[:, 1:, :], next_word_vec), axis=1)

#####################################################################################################
# Display generated recipe
#####################################################################################################

print("\nGenerated Recipe:")
print(' '.join(generated_text))