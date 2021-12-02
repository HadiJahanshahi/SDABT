import argparse
import ast
import copy
# import winsound
import datetime
import json
import math
import os
import pickle
import random
import re
import statistics
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gensim import corpora, matutils, models, similarities
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gurobipy import GRB, Model, quicksum
from IPython.display import clear_output
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import stats
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.layers import (LSTM, Activation, Bidirectional, Dense,
                                     Embedding, Flatten)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


def isNaN(num):
    return num != num

def isnotNaN(inp):
    return inp == inp

def convert_string_to_array(string_):
    """ it converts "[1,2,3]" to [1,2,3]
    Args:
        string_ ([str]): a list in the string format

    Returns:
        [list]: a numpy array 
    """
    array_string = ''.join(string_.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def string_to_time(string, format_ = '%Y-%m-%dT%H:%M:%SZ'):
    """
    Converting a string to time
    Args:
        string ([str]): a string that needs to be converted to time format (in a proper format)
        format_ ([str], optional): [time format]. Defaults to '%Y-%m-%dT%H:%M:%SZ'.

    Returns:
        [datetime]: date time formatted from the library datetime
    """
    return datetime.strptime(string, format_)
    
def mean_list (x):
    return np.mean(list(x))
