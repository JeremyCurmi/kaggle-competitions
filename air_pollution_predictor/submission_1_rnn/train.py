import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


logging.basicConfig(level=logging.INFO, format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%H:%M:%S')


def rnn_model():
    model = Sequential()
    model.add(LSTM(LENGTH, activation='relu', input_shape=(LENGTH, N_FEATURES)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


train_df = pd.read_csv('data/train.csv')
train_df['date_time'] = pd.to_datetime(train_df['date_time'])


# Feature engineering
train_df['year'] = train_df['date_time'].dt.year
train_df['month'] = train_df['date_time'].dt.month
train_df['hour'] = train_df['date_time'].dt.hour
train_df['day'] = train_df['date_time'].dt.day

train = train_df.set_index("date_time").copy()

target_columns = [col for col in train.columns if col.startswith('target')]
feature_columns = [col for col in train.columns if col not in target_columns]

ARTIFACT_PATH = 'model_artifacts/'
TEST_PERCENT = 0.1
test_point = np.round(len(train)*TEST_PERCENT)
test_idx = int(len(train) - test_point)

TARGET_IDX = 0
BATCH_SIZE = 1
LENGTH = 48

X_train = train.drop(target_columns, axis=1).iloc[:test_idx,:]
X_test = train.drop(target_columns, axis=1).iloc[test_idx:,:]

y_train = train[target_columns[TARGET_IDX]].iloc[:test_idx]
y_test = train[target_columns[TARGET_IDX]].iloc[test_idx:]

N_FEATURES = X_train.shape[1]

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
pickle.dump(scaler, open(ARTIFACT_PATH + 'scaler.pkl', 'wb'))

train_generator = TimeseriesGenerator(
    data=X_train_scaled,
    targets=y_train,
    length=LENGTH,
    batch_size=BATCH_SIZE,
)

test_generator = TimeseriesGenerator(
    data=X_test_scaled,
    targets=y_test,
    length=LENGTH,
    batch_size=BATCH_SIZE,
)

tf.random.set_seed(1)
early_stop = EarlyStopping(monitor='val_loss', patience=2)

for target in target_columns:
    logging.info(f"Training model for {target}")
    model = rnn_model()
    logging.info(model.summary())
    model.fit_generator(train_generator, epochs=10, validation_data=test_generator, callbacks=[early_stop])
    losses = pd.DataFrame(model.history.history)
    print(losses[['loss', 'val_loss']])
    model.save(ARTIFACT_PATH + f'{target}_model.h5')  # HDF5