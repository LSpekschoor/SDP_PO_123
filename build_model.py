import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
import re

path = os.getcwd() + "/data/Google-Playstore-Modified.parquet"
df = pd.read_parquet(path, engine='fastparquet')

# Note, requires following order: train, val, test & always input at least train
def prepare_inputs(encoder_type, *args):
    encoded_sets = range(0,len(args))
    try:
        ohe = encoder_type
        ohe.fit(args[0])
        encoded_sets = []
        for type_set in args:
            encoded_sets.append(ohe.transform(type_set))
        # print(encoded_sets)
        return tuple(encoded_sets)
    except:
        print("Please try to input only train, val or test set")

#x = None # drop App id
#y = None

x = df[['Rating Count', 'Maximum Installs', 'Price', 'Free', 'Editors Choice', 'Editors Choice', 'In App Purchases']].values
y = df['Rating Bin'].values

if x.any():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=123)
    X_train, X_val, X_test = prepare_inputs(sklearn.preprocessing.OneHotEncoder(), x_train, x_val, x_test)

print(X_train)
# print(prepare_inputs(sklearn.preprocessing.OneHotEncoder(), df))

class Network:
    def __init__(self, name):
        self.name = name
        self.init_model = None  # model to be build & trained
        self.model = None       # trained model
        pass

    def build(self, activation_, optimizer_, loss_, n_features_input, n_features_output):
        self.init_model = Sequential()
        self.init_model.add(layers.Dense(n_features_input, activation=activation_))
        self.init_model.add(layers.Dense(10, activation=activation_))
        self.init_model.add(layers.Dense(n_features_output, activation=activation_))
        self.init_model.compile(optimizer=optimizer_, loss=loss_)

    def train(self, train_set, epochs_, verbose_, checkpoint_path):
        # to implement: tf.keras.callbacks.ModelCheckpoint, this lets you save the model during &
        # after training

        # create directory if it does not exist
        if not os.path.exists('training_logs'):
            os.makedirs('training_logs')

        # callback helps to save model during & after training
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        # train the model                                         
        self.model = self.init_model.fit(x=x_train, y=y_train, epochs=epochs_, verbose=verbose_, 
            callbacks=[cp_callback])

    def test(self, test_set):
        pass

    def eval(self, true_labels, predicted_labels):
        pass

    def get_loss(self, model):
        return pd.DataFrame(self.model.history.history) 

    # Or save as pickle file??
    def save_model(self):
        self.model.save(str(self.name + '.h5'))

    def load_model(self):
        self.model.load(str(self.name + '.h5'))

# optimizer = keras.optimizers.Adam(learning_rate=0.01) # perform grid search for multiple learning rates
