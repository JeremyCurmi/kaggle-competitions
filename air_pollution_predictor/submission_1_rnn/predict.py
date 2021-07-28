import glob
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model


test_df = pd.read_csv('data/test.csv')
test_df['date_time'] = pd.to_datetime(test_df['date_time'])

# Feature engineering
test_df['year'] = test_df['date_time'].dt.year
test_df['month'] = test_df['date_time'].dt.month
test_df['hour'] = test_df['date_time'].dt.hour
test_df['day'] = test_df['date_time'].dt.day

test = test_df.set_index("date_time").copy()

feature_columns = test.columns.tolist()

ARTIFACT_PATH = 'model_artifacts/'

BATCH_SIZE = 1
LENGTH = 48

scaler = pickle.load(open(ARTIFACT_PATH + 'scaler.pkl', 'rb'))
test_scaled = scaler.transform(test)

test_generator = tf.keras.preprocessing.timeseries_dataset_from_array(
    test,
    None,
    sequence_length=BATCH_SIZE,
    batch_size=1)

models = glob.glob(f"{ARTIFACT_PATH}*.h5")

submission_df = pd.DataFrame(index=test.index)
for model_file_name in models:
    target = model_file_name.split('/')[-1].split('_model')[0]
    model = load_model(model_file_name)
    submission_df[target] = model.predict(test_generator)

submission_df.reset_index().to_csv('submission.csv', index=False)