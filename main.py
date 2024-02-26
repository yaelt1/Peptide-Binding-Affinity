import pandas as pd
from sklearn.model_selection import KFold
import torch
import keras
from torch.nn import functional
import torch.nn.functional as F
from keras import Sequential
from keras import layers
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Reshape
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

AMINO='ADEFGHKLNPQRSVWYX'

def convert_sequences_to_tokens(sequences, max_length, padding='right'):
        """Convert sequence strings to token representation

        Args:
            sequences (series): Amino acid peptide sequences
            max_length (int): Longest peptide sequence length
            padding (str, optional): Side to pad sequences to max length. Defaults to 'right'.

        Returns:
            tensor: Token representation of sequences
        """

        # Pad sequences with X to max length
        sequences_padded = sequences.str.pad(width=max_length, side=padding, fillchar='X')

        # Tokenize sequences
        amino_acid_indices = {aa: i for i, aa in enumerate(AMINO)}
        sequences_tokenized = sequences_padded.apply(lambda x: [amino_acid_indices[aa] for aa in x]).to_list()

        return torch.tensor(sequences_tokenized)
        
def convert_tokens_to_one_hot(tokens):
        """Convert sequence tokens to one-hot representation

        Args:
            tokens (tensor): Token representation of sequences

        Returns:
            tensor: One-hot representation of sequences
        """
      
        return functional.one_hot(tokens, num_classes=len(AMINO)).float()

def find_max_length_sequence(series):
    max_length = 0
    for sequence in series:
        length = len(sequence)
        if length > max_length:
            max_length = length
    return max_length

def keras_model():
    N=5
    model = Sequential()
    model.add(Input([16,17]))
    model.add(Dense(N, activation=None))
    model.add(Reshape((16*N,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='MSE', metrics=['MeanAbsoluteError'])
    return model
  
  
def cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes for classification
    ])
    model.compile(optimizer='adam',
              loss='mse',  # Assuming you have integer labels
             metrics=['MeanAbsoluteError'])
    return model
    
def fit_model(model_name, one_hot_sequences, labels):
    if model_name.lower()=="keras":
        model = keras_model()
    elif model_name.lower()=="cnn":
        model = cnn_model()
        one_hot_sequences = np.expand_dims(one_hot_sequences, axis=-1)       
    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(one_hot_sequences)):
        print(f"Fold {fold + 1}")
        X_train, X_test = one_hot_sequences[train_index], one_hot_sequences[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        history = model.fit(X_train,y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)




def main(filepath):
    df = pd.read_csv(filepath)
    sequences = df.iloc[:, 0]
    scores =  df.iloc[:, 1]
    scores = np.array(scores)
    scores += 100
    scores =np.log10(scores)
    scores = pd.Series(scores)
    max_length = find_max_length_sequence(sequences)
    tokens = convert_sequences_to_tokens(sequences=sequences, max_length=max_length)
    one_hot_sequences = convert_tokens_to_one_hot(tokens)
    model = cnn_model()
    fit_model("keras", one_hot_sequences, scores)
    
            
    
        
       
    
if __name__=="__main__":
    main("Diaphorase.csv")