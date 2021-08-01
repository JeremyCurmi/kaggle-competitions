import os
import sys
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append('../../')
sys.path.append('../../source_code')
from source_code.utils import current_timestamp

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.DEBUG)

BATCH_SIZE = 20
FILE_NAME = 'cnn_model_08-01-2021_16-15-19' + '.h5'

data_path = '/Users/jeremy/data/'
file_name = 'seti_breakthrough_listen_et'
data_path = data_path + file_name
TARGET = 'target'
user = 'jeremy'
root = f'/Users/{user}/data/seti_breakthrough_listen_et/test/'

all_files = []
for path, sub_dirs, files in os.walk(root):
    for name in files:
        all_files.append((name.split('.')[0], path + "/" + name))
test = pd.DataFrame(all_files, columns=['id', 'file_path'])
test_size = test.shape[0]
logging.info(f'Found {test_size} test files ✅')

test = test[30000:40000]

test_examples = []
test_indexes = []

logging.info("Fetching Examples")
for i in tqdm(test.index[:test_size].to_list()):
    test_examples.append(np.load(test.loc[i, 'file_path']).reshape(6, 273, 256))
    test_indexes.append(test.loc[i, 'id'])

logging.info("Test Examples Fetched ✅")

logging.info("Creating Dataset object")
test_dataset = tf.data.Dataset.from_tensor_slices(test_examples)
test_dataset = test_dataset.batch(BATCH_SIZE)
logging.info("Dataset object created ✅")

logging.info("Loading model")
model = load_model(FILE_NAME)
logging.info(f'Model {FILE_NAME} loaded ✅')

logging.info("predictions in progress")
submission_pred = model.predict(test_dataset).flatten()

current_timestamp_ = current_timestamp()
submission = pd.DataFrame([test_indexes, submission_pred], index=['id', 'target']).T
submission.to_csv(f'submission_{current_timestamp_}.csv')
