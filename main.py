import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import datetime
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import keras


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
    """
    Find the maximum length sequence from a series of sequences.

    Args:
    - series: Pandas Series containing sequences

    Returns:
    - max_sequence: Maximum length sequence found in the series
    """
    max_length = 0
    for sequence in series:
        length = len(sequence)
        if length > max_length:
            max_length = length
    return max_length



def main(filepath):
    df = pd.read_csv(filepath)
    sequences = df.iloc[:, 0]
    max_length = find_max_length_sequence(sequences)
    tokens = convert_sequences_to_tokens(sequences=sequences, max_length=max_length)
    hot_encoding = convert_tokens_to_one_hot(tokens)
    
if __name__=="__main__":
    main("Diaphorase.csv")