
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from gensim.models import word2vec

import preprocess

path_prefix = './'
def train_word2vec(x):
    model = word2vec.Word2Vec(x,vector_size = 250, window=5,min_count= 5, workers=12, epochs=10,sg=1)
    return model


if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = preprocess.load_training_data('training_label.txt')
    
    print("loading testing data.....")
    test_x = preprocess.load_testing_data('testing_data.txt')

    model = train_word2vec(train_x + test_x)

    print("saving model ...")

    model.save(os.path.join(path_prefix, 'w2v_all.model'))

