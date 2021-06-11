import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import keras
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import keras.optimizers
from keras import backend as K

df = pd.read_csv('spam_or_not_spam.csv')

def get_params(emails):
    emails = emails.dropna()
    unique = []
    vocab = 0
    max_len = 0
    for j in emails:
        j = j.split(' ')

        for i in j:

            if i not in unique:

                unique.append(i)
                vocab += 1
                if len(i) > max_len:
                    max_len = len(i)

    return emails,max_len,vocab




def preprocessing(emails,max_len,vocab):


    encoded = []
    for i in emails:
        encoded.append(one_hot(i, vocab))

    encoded = pad_sequences(encoded, maxlen=max_len, padding='post')
    return encoded


def spam_classifier():

    inputs = layers.Input(shape=max_len)
    layer = layers.Embedding(vocab, 150, input_length=max_len)(inputs)
    layer = layers.LSTM(64)(layer)
    layer = layers.Dense(256)(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Dropout(0.5)(layer)
    layer = layers.Dense(1)(layer)
    layer = layers.Activation('sigmoid')(layer)

    model = keras.Model(inputs=inputs, outputs=layer)
    return model


def rec(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def pre(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = pre(y_true, y_pred)
    recall = rec(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


emails , max_len , vocab = get_params(df['email'])
encoded = preprocessing(emails,max_len,vocab)


model = spam_classifier()
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy',f1,pre,rec])


labels = df['label'].to_numpy().reshape((len(df['label']),1))
model.fit(encoded,labels,epochs = 30,batch_size=64,validation_split=0.25)