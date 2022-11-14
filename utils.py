import io
import os
import json
import time
import random
import datetime
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)



def exist_make(file):
    if not os.path.exists(file):
        os.makedirs(file, exist_ok=True)


def read_txtdata(path):
    with open(path, 'r') as fp:
        eigenvalues = fp.readlines()

    eigenvalues = np.array([float(d.split('\n')[0]) for d in eigenvalues])

    return eigenvalues

def normal_data(data):
    data = data / sum(data)

    data = np.cumsum(data)
    data = np.concatenate([[0], data])

    return data
