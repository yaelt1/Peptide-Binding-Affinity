import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import torch
import keras
from torch.nn import functional
import torch.nn.functional as F
from tensorflow.keras.optimizers import Adam
from keras import Sequential
from keras import layers
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Reshape
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras.backend as K_backend
import tensorflow.keras.backend as K
import os
import json


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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='MSE', metrics=[pearson_correlation_coefficient])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    return model
  


def pearson_correlation_coefficient(y_true, y_pred):
    """
    Calculate Pearson correlation coefficient as a metric.
    
    Arguments:
    y_true -- true labels
    y_pred -- predicted labels
    
    Returns:
    Pearson correlation coefficient value
    """

    y_true -= K.mean(y_true)
    y_pred -= K.mean(y_pred)
    
    y_true /= K.std(y_true) + K.epsilon()
    y_pred /= K.std(y_pred) + K.epsilon()

    pearson_r = K.mean(y_true * y_pred)

    return pearson_r

def cnn_model():
    model = tf.keras.Sequential([
        model.add(keras.layers.Masking(mask_value=0)),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation=None) 
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
             metrics=[pearson_correlation_coefficient])
    return model
    
def fit_model(model_name, one_hot_sequences, labels):   
    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(one_hot_sequences)):
        if model_name.lower()=="keras":
            model = keras_model()
        elif model_name.lower()=="cnn":
            model = cnn_model()  
        print(f"Fold {fold + 1}")
        X_train, X_test = one_hot_sequences[train_index], one_hot_sequences[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        history = model.fit(X_train,y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    return model


def get_model_data(filepath,max_length ):
    df = pd.read_csv(filepath)
    sequences = df.iloc[:, 0]
    scores =  df.iloc[:, 1]
    scores = np.array(scores)
    scores += 100
    scores =np.log10(scores)
    mean = scores.mean()
    scores -= mean
    scores = pd.Series(scores)
    # max_length = find_max_length_sequence(sequences)
    tokens = convert_sequences_to_tokens(sequences=sequences, max_length=max_length)
    one_hot_sequences = convert_tokens_to_one_hot(tokens)
    return one_hot_sequences, scores
    
    
    
def fit_on_data(train_filepath, max_length):
    K_backend.clear_session()
    train_one_hot_sequences, train_scores = get_model_data(train_filepath, max_length)
    model = fit_model("keras", train_one_hot_sequences, train_scores)
    return model

def predict_on_data(model, predict_filepath, max_length):
    test_one_hot_sequences, test_scores = get_model_data(predict_filepath, max_length)
    predictions = model.predict(test_one_hot_sequences, batch_size=None, verbose="auto", steps=None, callbacks=None)
    np_scores = test_scores.to_numpy()
    predictions = np.squeeze(predictions)
    pearson_coefficient, _ = pearsonr(np_scores, predictions)
    print(pearson_coefficient)
    return pearson_coefficient
    
            
    
        
       
    
if __name__=="__main__":
    # train_one_hot_sequences, scores = get_model_data("Diaphorase.csv")
    # plt.hist(scores, bins=100)
    # plt.show()
    max_length = 16
    directory ="/Users/yaeltzur/Desktop/uni/third_yaer/סדנה/proteins"
    # for file in os.listdir(directory):
    #     file_path = os.path.join(directory, file)
    #     df = pd.read_csv(file_path)
    #     cur_max = find_max_length_sequence(df.iloc[:, 0])
    #     max_length = max(max_length, cur_max)
    
    fit_filepath = os.path.join(directory, "Diaphorase.csv")
    model = fit_on_data(fit_filepath, max_length)
    # model = keras.models.load_model("/Users/yaeltzur/Desktop/uni/third_yaer/סדנה/keras_dia_model.keras")
    pearson_coef = {}
    pearson_coef["Diaphorase"] = []
    for file in os.listdir(directory):
        if file != "Diaphorase.csv":
            file_path = os.path.join(directory, file)
            print("Predicting on ",file)
            pearson_coef_pred = predict_on_data(model, file_path, max_length)
            pearson_coef["Diaphorase"].append(pearson_coef_pred)
    with open("pearson_coef_dict", "w") as json_file:
    
        json.dump(pearson_coef, json_file)

    