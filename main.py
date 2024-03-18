import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import keras
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Reshape, Masking
from keras import Sequential
from keras import layers
import keras.backend as K_backend
import torch
from torch.nn import functional
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os


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
    """Find the maximum length of peptide sequences in a series.

    Args:
        series: Series containing peptide sequences.

    Returns:
        int: Maximum length of peptide sequences.
    """
    max_length = 0
    for sequence in series:
        length = len(sequence)
        if length > max_length:
            max_length = length
    return max_length

  


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

def fnn_model():
    """Builds a feedforward neural network (FNN) model for peptide binding prediction.

    Returns:
        keras.Model: FNN model.
    """
    N=5
    model = Sequential()
    model.add(Input([16,17]))
    model.add(Dense(N, activation=None))
    model.add(Reshape((16*N,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation=None))
    model.add(Masking(mask_value=0))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='MSE', metrics=[pearson_correlation_coefficient])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    return model

def cnn_model():
    """Builds a convolutional neural network (CNN) model for peptide binding prediction.

    Returns:
        keras.Model: CNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation=None) 
    ])
    model.add(tf.keras.layers.Masking(mask_value=0))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='MSE',
             metrics=[pearson_correlation_coefficient])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    return model
    
def fit_model(model_name, one_hot_sequences, labels):   
    """Fits the specified machine learning model on the data.

    Args:
        model_name (str): Name of the model ('fnn' or 'cnn').
        one_hot_sequences (array-like): One-hot encoded peptide sequences.
        labels (array-like): Labels for the sequences.

    Returns:
        keras.Model: Trained machine learning model.
    """
    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(one_hot_sequences)):
        if model_name.lower()=="fnn":
            model = fnn_model()
        elif model_name.lower()=="cnn":
            model = cnn_model()  
        print(f"Fold {fold + 1}")
        X_train, X_test = one_hot_sequences[train_index], one_hot_sequences[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        history = model.fit(X_train,y_train, batch_size=32, epochs=7, validation_data=(X_test, y_test))
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print(score)
    return model


def get_model_data(filepath,max_length):
    """Reads data from a CSV file and preprocesses it for model training.

    Args:
        filepath (str): Path to the CSV file containing data.
        max_length (int): Maximum length of peptide sequences.

    Returns:
        tuple: Tuple containing one-hot encoded sequences and corresponding labels.
    """
    df = pd.read_csv(filepath)
    sequences = df.iloc[:, 0]
    scores =  df.iloc[:, 1]
    scores = np.array(scores)
    scores += 100
    scores =np.log10(scores)
    mean = scores.mean()
    scores -= mean
    scores = pd.Series(scores)
    tokens = convert_sequences_to_tokens(sequences=sequences, max_length=max_length)
    one_hot_sequences = convert_tokens_to_one_hot(tokens)
    return one_hot_sequences, scores
    
    
def fit_on_data(train_filepath, max_length, model="fnn"):
    """Fits the specified model on the training data.

    Args:
        train_filepath (str): Path to the CSV file containing training data.
        max_length (int): Maximum length of peptide sequences.
        model (str, optional): Name of the model ('fnn' or 'cnn'). Defaults to 'fnn'.

    Returns:
        keras.Model: Trained machine learning model.
    """
    K_backend.clear_session()
    train_one_hot_sequences, train_scores = get_model_data(train_filepath, max_length)
    model = fit_model(model, train_one_hot_sequences, train_scores)
    return model

def predict_on_data(model, predict_filepath, max_length):
    """Makes predictions using the trained model on the test data.

    Args:
        model (keras.Model): Trained machine learning model.
        predict_filepath (str): Path to the CSV file containing test data.
        max_length (int): Maximum length of peptide sequences.

    Returns:
        tuple: Tuple containing Pearson correlation coefficient and predicted values.
    """
    test_one_hot_sequences, test_scores = get_model_data(predict_filepath, max_length)
    predictions = model.predict(test_one_hot_sequences, batch_size=None, verbose="auto", steps=None, callbacks=None)
    np_scores = test_scores.to_numpy()
    predictions = np.squeeze(predictions)
    pearson_coefficient, _ = pearsonr(np_scores, predictions)
    print(pearson_coefficient)
    return pearson_coefficient, predictions
    
            
    
        
def train_pred_all_pairs(model_name, proteins_dir, max_length=16):
    """Trains the specified model on all pairs of protein files in the directory and saves results.

    Args:
        model_name (str): Name of the model ('fnn' or 'cnn').
        proteins_dir (str): Directory containing protein CSV files.
        max_length (int, optional): Maximum length of peptide sequences. Defaults to 16.
    """
    files = [file for file in os.listdir(proteins_dir)]
    df = pd.DataFrame(columns= files)
    for train_file in files:
        fit_filepath = os.path.join(proteins_dir, train_file)
        model = fit_on_data(fit_filepath, max_length, model_name)
        pearson_coef_preds = []
        for predict_file in files:
            predict_file_path = os.path.join(proteins_dir, predict_file)
            print("Predicting on ",predict_file_path)
            pearson_coef_pred,predictions = predict_on_data(model, predict_file_path, max_length)
            pearson_coef_preds.append(pearson_coef_pred)
        df[train_file] = pd.Series(pearson_coef_preds)
    
    df.index = files
    df.to_csv(f"results_{model_name}.csv")

         
def scatter_test_pred(fit_filepath, predict_file_path, model_name,max_length=16): 
    """Plots a scatter plot of true vs predicted scores for a specified model.

    Args:
        fit_filepath (str): Path to the CSV file containing training data.
        predict_file_path (str): Path to the CSV file containing test data.
        model_name (str): Name of the model ('fnn' or 'cnn').
        max_length (int, optional): Maximum length of peptide sequences. Defaults to 16.
    """
    test_one_hot_sequences, test_scores = get_model_data(predict_file_path, max_length)
    model = fit_on_data(fit_filepath, max_length, model_name)
    pearson_coef_pred,predictions = predict_on_data(model, predict_file_path, max_length)
    x = test_scores
    y = np.squeeze(predictions)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.title('True vs Predicted Scores')
    plt.xlabel('True Scores')
    plt.ylabel('Predicted Scores')
    plt.grid(True)
    plt.show()
    
def heat_map(filepath):
      """Generate a heatmap of Pearson correlation coefficients from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing Pearson correlation coefficients.

    Returns:
        None
    """
    df = pd.read_csv(filepath, index_col=0)
    df_sorted_rows = df.sort_index()
    df_sorted_both = df_sorted_rows[sorted(df_sorted_rows.columns)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_sorted_both, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Pearson Correlation Coefficients CNN')
    plt.show()
    
    
if __name__=="__main__":
    train_pred_all_pairs("cnn",proteins_dir="/Users/yaeltzur/Desktop/uni/third_yaer/סדנה/proteins")
    