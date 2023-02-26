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


baseline = True
build_NN = False


class Data:
    def __init__(self, drop_variables, target_variable):
        self.drop_variables = drop_variables
        self.target_variable = target_variable

    def get_data(self):
        path = os.getcwd() + "/data/Google-Playstore-Modified.parquet"
        df = pd.read_parquet(path, engine='fastparquet')
        x = df.drop(self.drop_variables, axis=1).values #'Rating Bin'
        y = df[self.target_variable].values
        if x.any():
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=123)
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def scaler(self, scaler_type, x_train, x_val, x_test, y_train, y_val, y_test):
        scaler = scaler_type.fit(x_train)
        x_train_ = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        y_train = scaler.transform(y_train)
        y_val = scaler.transform(y_val)
        y_test = scaler.transform(y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test


class Baseline:
    def __init__(self, type_model, problem_type):
        self.type_model = type_model
        self.problem_type = problem_type
        self.model = None
        self.preds = None
    
    def train(self, train_data, train_labels):
        self.model = self.type_model.fit(train_data, train_labels)
    
    def test(self, test_data):
        self.preds = self.model.predict(test_data)
    
    def eval(self, test_labels):
        if self.problem_type == 'Regression':
            rmse = metrics.mean_squared_error(y_true=test_labels,y_pred=self.preds,squared=False)

        elif self.problem_type == 'Classification':
            accuracy = metrics.accuracy_score(test_labels, self.preds)
            print('Accuracy score is {}'.format(accuracy))

            cf = pd.DataFrame(metrics.confusion_matrix(test_labels, self.preds), index=['Bad App', 'Alrighty', 'Superb'], 
                            columns=['Bad App', 'Alrighty', 'Superb'])
            plt.figure(figsize=(3,3))
            sns.heatmap(cf) #annot=True)
            plt.show()
            return accuracy, cf
        else:
            return 0


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


data = Data(drop_variables=['Rating Bin', 'App Id'], target_variable='Rating Bin')
x_train, x_val, x_test, y_train, y_val, y_test = data.get_data()
#x_train, x_val, x_test, y_train, y_val, y_test = data.scaler(sklearn.preprocessing.StandardScaler(), x_train, x_val, 
 #                                                            x_test, y_train, y_val, y_test)

if baseline:
    b = Baseline(LogisticRegression(), 'Classification')
    b.train(x_train, y_train)
    b.test(x_val)
    b.eval(y_val)

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