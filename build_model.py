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
from sklearn.tree import DecisionTreeClassifier
import re

path = os.getcwd() + "/data/Google-Playstore-Modified.parquet"
df = pd.read_parquet(path, engine='fastparquet')


# x = df.drop(['Bad App Yo', 'Moderate', 'Superb', 'App Id'], axis=1).values #'Rating Bin'
# y = df[['Bad App Yo', 'Moderate', 'Superb']].values
x = df.drop(['Rating Bin', 'App Id'], axis=1).values #'Rating Bin'
y = df['Rating Bin'].values

if x.any():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=123)

# print(x_train)

scaler = sklearn.preprocessing.StandardScaler().fit(x_train)
scaled_X_train = scaler.transform(x_train)
scaled_X_test = scaler.transform(x_val)

baseline = False
if baseline:

    # hc = DecisionTreeClassifier().fit(x_train, y_train)
    c = LogisticRegression(multi_class='ovr').fit(x_train, y_train)

    preds = c.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, preds)
    print('Accuracy score is {}'.format(accuracy))

    cf = pd.DataFrame(metrics.confusion_matrix(y_test, preds), index=['Bad App', 'Alrighty', 'Superb'], 
                    columns=['Bad App', 'Alrighty', 'Superb'])
    plt.figure(figsize=(3,3))
    sns.heatmap(cf) #annot=True)
    # plt.show()



#if not os.path.exists('pred_logs'):
 #           os.makedirs('pred_logs')

class Network:
    def __init__(self, name):
        self.name = name
        self.init_model = None  # model to be build & trained
        self.model = None       # trained model
        self.train_data = None
        self.test_data = None
        self.predictions = None
        pass

    def build(self, activation_, optimizer_, loss_, output_activation, n_features_input, n_features_output):
        self.init_model = Sequential()
        self.init_model.add(layers.Dense(n_features_input, activation=activation_))
        self.init_model.add(layers.Dense(10, activation=activation_))
        self.init_model.add(layers.Dense(n_features_output, activation=output_activation))
        self.init_model.compile(optimizer=optimizer_, loss=loss_)

    def train(self, train_set, train_labels, epochs_, verbose_, checkpoint_path):
        # to implement: tf.keras.callbacks.ModelCheckpoint, this lets you save the model during &
        # after training

        # create directory if it does not exist
        if not os.path.exists('training_logs'):
            os.makedirs('training_logs')

        # callback helps to save model during & after training
        call_back = False
        if call_back:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
            # train the model                                         
            self.model = self.init_model.fit(x=train_set, y=train_labels, epochs=epochs_, verbose=verbose_, 
                callbacks=[cp_callback])
        else:
            self.model = self.init_model
            self.model.fit(x=train_set, y=train_labels, epochs=epochs_, verbose=verbose_)

    def get_loss(self):
        # does not work yet
        return pd.DataFrame(self.model.history.history)

    def test(self, test_set):
        # Preds = pd.Series(preds.reshape(300,)
        self.predictions=self.model.predict(test_set)

    def eval(self, true_labels):
        accuracy = metrics.accuracy_score(true_labels, self.predictions)
        print(f'Accuracy score is {accuracy}')
        return accuracy 

    # Or save as pickle file??
    def save_model(self):
        self.model.save(str(self.name + '.h5'))

    def load_model(self):
        self.model.load(str(self.name + '.h5'))

build_NN = True
if build_NN:
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    x_val = np.asarray(x_val).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)
    optimizer = keras.optimizers.Adam(learning_rate=0.01) # perform grid search for multiple learning rates
    model = Network(name='Bram')
    model.build(activation_='relu', optimizer_=optimizer, loss_='categorical_crossentropy', output_activation='softmax', 
                n_features_input=10, n_features_output=1)
    model.train(train_set=x_train, train_labels=y_train, epochs_=1, verbose_=1, checkpoint_path=None)
    # model.get_loss()
    model.test(x_val)
    ev = model.eval(y_val)