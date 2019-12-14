import numpy as np
import sys
import io
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import os
os.chdir('./5-1/')
from shakespeare_utils import *

model.fit(x, y, batch_size=128, epochs=1)

generate_output()


