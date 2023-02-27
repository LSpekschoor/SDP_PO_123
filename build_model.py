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


baseline = False
build_NN = True
save_model = False
load_model = False


class Data:
    def __init__(self, drop_variables, target_variable):
        self.drop_variables = drop_variables
        self.target_variable = target_variable

    def get_data(self):
        path = os.getcwd() + "/data/Google-Playstore-Modified_w_ohe_y.parquet"
        df = pd.read_parquet(path, engine='fastparquet')
        x = df.drop(self.drop_variables, axis=1).values #'Rating Bin'
        y = df[self.target_variable].values
        if x.any():
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=123)
        x_train = np.asarray(x_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        x_val = np.asarray(x_val).astype(np.float32)
        y_val = np.asarray(y_val).astype(np.float32)
        x_test = np.asarray(x_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def scaler(self, scaler_type, x_train, x_val, x_test):
        scaler = scaler_type.fit(x_train)
        x_train_ = scaler.transform(x_train)
        x_val_ = scaler.transform(x_val)
        x_test_ = scaler.transform(x_test)
        return x_train_, x_val_, x_test_


class Baseline:
    def __init__(self, type_model, problem_type):
        self.type_model = type_model
        self.problem_type = problem_type
        self.model = None
        self.preds = None
    
    def train(self, train_data, train_labels):
        train_labels = np.argmax(train_labels,axis=1)
        self.model = self.type_model.fit(train_data, train_labels)
    
    def test(self, test_data):
        self.preds = self.model.predict(test_data)
    
    def eval(self, test_labels):
        if self.problem_type == 'Regression':
            rmse = metrics.mean_squared_error(y_true=test_labels,y_pred=self.preds,squared=False)

        elif self.problem_type == 'Classification':
            test_labels = np.argmax(test_labels,axis=1)
            accuracy = metrics.accuracy_score(test_labels, self.preds)
            print(f'Accuracy score for {str(self.type_model)} is: {accuracy}%')

            cf = pd.DataFrame(metrics.confusion_matrix(test_labels, self.preds), index=['Bad App', 'Alrighty', 'Superb'], 
                            columns=['Bad App', 'Alrighty', 'Superb'])
            plt.figure(figsize=(3,3))
            sns.heatmap(cf, annot=True) #annot=True)
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
        #self.init_model.add(layers.Dropout(0.5))
        self.init_model.add(layers.Dense(10, activation=activation_))
        self.init_model.add(layers.Dropout(0.5))
        self.init_model.add(layers.Dense(10, activation=activation_))
        self.init_model.add(layers.Dropout(0.5))
        self.init_model.add(layers.Dense(10, activation=activation_))
        self.init_model.add(layers.Dropout(0.2))
        self.init_model.add(layers.Dense(n_features_output, activation=output_activation))
        self.init_model.compile(optimizer=optimizer_, loss=loss_)

    def train(self, train_set, train_labels, epochs_, verbose_, val_set, val_labels, checkpoint_path):
        # create directory if it does not exist
        if not os.path.exists('outputs/training_logs'):
            os.makedirs('outputs/training_logs')

        # callback helps to save model during & after training
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
        # train the model
        self.model = self.init_model                                    
        self.model.fit(x=train_set, y=train_labels, epochs=epochs_, verbose=verbose_, 
                           validation_data=[val_set, val_labels], callbacks=[cp_callback])
   
    def get_loss(self):
        # plot loss for training & val
        hist = pd.DataFrame(self.model.history.history)
        hist.plot()
        plt.show()

    def test(self, test_set):
        self.predictions=self.model.predict(test_set)
        self.predictions = np.argmax(self.predictions,axis=1)

    def eval(self, true_labels):
        true_labels = np.argmax(true_labels,axis=1)
        accuracy = metrics.accuracy_score(true_labels, self.predictions)
        print(f'Accuracy score is {accuracy}')

        # to create confusion matrix
        cf = pd.DataFrame(metrics.confusion_matrix(true_labels, self.predictions), index=['Bad App', 'Alrighty', 'Superb'], 
                            columns=['Bad App', 'Alrighty', 'Superb'])
        cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(3,3))
        sns.heatmap(cf, annot=True, fmt='.2f') #annot=True)
        plt.show()
        return accuracy 

    def save_model(self):
        self.model.save('outputs/model/' + str(self.name + '.h5'))

    def load_model(self, name_model):
        tf.keras.models.load('outputs/model/' + str(name_model + '.h5'))
        # self.model = tf.keras.models.load


# data = Data(drop_variables=['Rating Bin', 'App Id'], target_variable='Rating Bin')
d_v = ['App Id', 'Bad App Yo', 'Moderate', 'Superb'] # or ['Rating Bin', 'App Id']
t_v = ['Bad App Yo', 'Moderate', 'Superb'] # or 'Rating Bin'
t_v_len = 3
data = Data(drop_variables=d_v, target_variable=t_v)
x_train, x_val, x_test, y_train, y_val, y_test = data.get_data()
x_train, x_val, x_test= data.scaler(sklearn.preprocessing.StandardScaler(), x_train, x_val, x_test)

if baseline:
    baselines = [LogisticRegression(), DecisionTreeClassifier()]
    for classifier in baselines:
        b = Baseline(classifier, 'Classification')
        b.train(x_train, y_train)
        b.test(x_val)
        b.eval(y_val)

if build_NN:
    optimizer = keras.optimizers.Adam(learning_rate=0.01) # perform grid search for multiple learning rates
    model = Network(name='Bram')
    model.build(activation_='relu', optimizer_=optimizer, loss_='categorical_crossentropy', 
                output_activation='softmax', n_features_input=10, n_features_output=t_v_len)
    model.train(train_set=x_train, train_labels=y_train, epochs_=10, verbose_=1, val_set=x_val, val_labels=y_val, 
                checkpoint_path='outputs/training_logs/' + model.name)
    model.get_loss()
    model.test(x_val)
    ev = model.eval(y_val)
    if save_model:
        model.save_model()

if load_model:
    model = Network(name='Bram')
    model.load_model(name_model='Bram.h5')



    