#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[17]:

import numpy as np
import pandas as pd
import tqdm as tqdm
import logging
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from random import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import sys
sys.path.append('../../')
sys.path.append('../../source_code')
from source_code.utils import current_timestamp
logging.info("importing libraries")


# In[18]:


data_path = '/Users/jeremy/data/'
file_name = 'seti_breakthrough_listen_et'
data_path = data_path + file_name


# In[19]:


TARGET = 'target'


# In[20]:


def get_train_file_path(image_id):
    return f"{data_path}/train/{image_id[0]}/{image_id}.npy"

def get_test_file_path(image_id):
    return f"{data_path}/test/{image_id[0]}/{image_id}.npy"


# In[21]:

logging.info("loading data")
train = pd.read_csv(data_path + '/train_labels.csv')
train['file_path'] = train['id'].apply(get_train_file_path)
train['file_path'] = train['file_path']#.str.split(prefix).str[-1]

# test = pd.read_csv(data_path + '/sample_submission.csv')
# test['file_path'] = test['id'].apply(get_test_file_path)
# test['file_path'] = test['file_path']#.str.split(prefix).str[-1]


# In[22]:


train_df, validation_df = train_test_split(train, test_size=0.1)


# In[23]:


train_size = int(train_df.shape[0]/2)
validation_size = int(validation_df.shape[0]/3)
# test_size = int(test.shape[0])
print(train_size, validation_size) #, test_size)


# In[ ]:


train_examples = []
validation_examples = []
# test_examples = []

train_labels = []
validation_labels = []
# test_indexes = []

for i in tqdm(train_df.index[:train_size].to_list()):
    train_examples.append(np.load(train_df.loc[i,'file_path']).reshape(6,273,256))
    train_labels.append(train_df.loc[i,TARGET])

for i in tqdm(validation_df.index[:validation_size].to_list()):
    validation_examples.append(np.load(validation_df.loc[i,'file_path']).reshape(6,273,256))
    validation_labels.append(validation_df.loc[i,TARGET])

# for i in tqdm(test.index[:test_size].to_list()):
#     test_examples.append(np.load(test.loc[i,'file_path']).reshape(6,273,256))
#     test_indexes.append(test.loc[i, 'id'])


# In[ ]:


train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_examples, validation_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices(test_examples)


# In[ ]:


BATCH_SIZE = 20
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 30

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)
class_weights = class_weight.compute_class_weight('balanced', np.unique(train['target'].values),
                                                  train['target'].values)
class_weights = dict(enumerate(class_weights))
class_weights[1] = class_weights[1]*0.5
class_weights


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import save_model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(6, 273, 256)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC()])


# In[ ]:


callbacks = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True,)
model.fit(train_dataset, epochs=EPOCHS, validation_data = validation_dataset, callbacks=[callbacks], class_weight=class_weights)


# In[ ]:


# save model
current_timestamp_ = current_timestamp()
save_model(model, f'cnn_model_{current_timestamp_}.h5')


# In[ ]:


losses = pd.DataFrame(model.history.history)
losses


# In[ ]:


losses[['auc','val_auc']].plot()


# In[ ]:


losses[['loss','val_loss']].plot()


# In[ ]:


preds = model.predict(validation_dataset).flatten()


# In[ ]:


binary_preds = (preds > 0.5).astype(int)
confusion_matrix(validation_labels, binary_preds)


# In[ ]:


print(classification_report(validation_labels, binary_preds))


# In[ ]:


# print(validation_examples[0].shape, test_examples[0].shape)
# print(len(validation_examples), len(test_examples))


# In[ ]:


# submission_pred = model.predict(test_dataset).flatten()


# In[ ]:


# submission = pd.DataFrame([test_indexes, submission_pred], index=['id','target']).T
# submission.to_csv(f'submission_{current_timestamp_}.csv')
# submission


# In[ ]:




