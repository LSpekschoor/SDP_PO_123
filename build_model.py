import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import metrics
# from ann_visualizer.visualize import ann_viz
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Run configuraties
baseline = False
build_NN = True
save_model = False
load_model = False
only_vis = True

class Data:
    def __init__(self, drop_variables, target_variable):
        self.drop_variables = drop_variables
        self.target_variable = target_variable

    def get_data(self):
        path = os.getcwd() + "/data/Google-Playstore-Modified_L.parquet"
        df = pd.read_parquet(path, engine='fastparquet')
        df_features = df.drop(self.drop_variables, axis=1)
        x = np.asarray(df_features.values).astype(np.float32) #'Rating Bin'
        y = np.asarray(df[self.target_variable].values).astype(np.float32)
        if x.any():
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.11, random_state=123)
        return x_train, x_val, x_test, y_train, y_val, y_test, len(df_features.columns)
    
    def scaler(self, scaler_type, x_train, x_val, x_test):
        scaler = scaler_type.fit(x_train)
        x_train_ = scaler.transform(x_train)
        x_val_ = scaler.transform(x_val)
        x_test_ = scaler.transform(x_test)
        return x_train_, x_val_, x_test_


class Baseline:
    def __init__(self, type_model, problem_type):
        self.name = str(type_model)[:-2]
        self.type_model = type_model
        self.problem_type = problem_type
        self.model = None
        self.preds = None
    
    def train(self, train_data, train_labels):
        # np argmax voor de kolom
        train_labels = np.argmax(train_labels,axis=1)
        self.model = self.type_model.fit(train_data, train_labels)
    
    def test(self, test_data):
        self.preds = self.model.predict(test_data)
    
    def eval(self, test_labels, set, save):
        if self.problem_type == 'Regression':
            rmse = metrics.mean_squared_error(y_true=test_labels,y_pred=self.preds,squared=False)

        elif self.problem_type == 'Classification':
            test_labels = np.argmax(test_labels,axis=1)
            accuracy = metrics.accuracy_score(test_labels, self.preds)
            precision = metrics.precision_score(test_labels, self.preds, average='macro')
            recall = metrics.recall_score(test_labels, self.preds, average='macro')
            print('\n' + self.name)
            print(f'Accuracy score is {accuracy}')
            print(f'Precision score is {precision}')
            print(f'Recall score is {recall}')

            cf = pd.DataFrame(metrics.confusion_matrix(test_labels, self.preds), index=['Bad App', 'Alrighty', 'Superb'], 
                            columns=['Bad App', 'Alrighty', 'Superb'])
            
            if save:
                plt.figure(figsize=(10,10))
                sns.heatmap(cf, annot=True, fmt='g') #annot=True)

                path = 'outputs/' + self.name
                if not os.path.exists(path):
                    os.makedirs(path)
                
                plt.savefig(path + '/confusion_matrix.png')
                lines = [f'Accuracy score is {accuracy}', f'Precision score is {precision}', f'Recall score is {recall}']
                with open(path + '/' + set + '_scores.txt', 'w') as f:
                    f.writelines('\n'.join(lines))
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
        ## Clever Hans
        self.init_model = Sequential()
        # self.init_model.add(layers.Dense(n_features_input, input_shape=(11, ))) #, activation=activation_))
        self.init_model.add(layers.Dense(128, activation=activation_)) #, input_shape=(11,)))
        # self.init_model.add(layers.BatchNormalization(axis=-1))
        self.init_model.add(layers.Dense(16, activation=activation_, kernel_regularizer='l2'))
        self.init_model.add(layers.Dense(n_features_output, activation=output_activation))
        self.init_model.compile(optimizer=optimizer_, loss=loss_)
        a = self.init_model
        return a

    def train(self, train_set, train_labels, epochs_, verbose_, val_set, val_labels, batch_size_):
        # create directory if it does not exist
        path = 'outputs/' + self.name
        if not os.path.exists(path):
            os.makedirs(path)

        # introduce early stopping
        cp_callback = tf.keras.callbacks.EarlyStopping(start_from_epoch=5, patience=5, restore_best_weights=True)
        #logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #cp_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        self.model = self.init_model                                    
        self.model.fit(x=train_set, y=train_labels, epochs=epochs_, verbose=verbose_, 
                        validation_data=[val_set, val_labels], batch_size=batch_size_, callbacks=[cp_callback])
   
    def get_loss(self):
        # plot loss for training & val
        hist = pd.DataFrame(self.model.history.history)
        hist.plot()
        plt.savefig('outputs/' +str(self.name) + '/loss_plot.png')

    def test(self, test_set):
        self.predictions=self.model.predict(test_set)
        self.predictions = np.argmax(self.predictions,axis=1)

    def eval(self, true_labels, set):
        true_labels = np.argmax(true_labels,axis=1)
        accuracy = metrics.accuracy_score(true_labels, self.predictions)
        precision = metrics.precision_score(true_labels, self.predictions, average='macro')
        recall = metrics.recall_score(true_labels, self.predictions, average='macro')
        print(f'Accuracy score is {accuracy}')
        print(f'Precision score is {precision}')
        print(f'Recall score is {recall}')

        # to create confusion matrix
        cf = pd.DataFrame(metrics.confusion_matrix(true_labels, self.predictions), index=['Bad App', 'Alrighty', 'Superb'], 
                            columns=['Bad App', 'Alrighty', 'Superb'])
        cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10,10))
        sns.heatmap(cf, annot=True, fmt='.2f') #annot=True)
        plt.savefig('outputs/' +str(self.name) + '/confusion_matrix.png')
        lines = [f'Accuracy score is {accuracy}', f'Precision score is {precision}', f'Recall score is {recall}']
        with open('outputs/' + str(self.name) + '/' + set + '_scores.txt', 'w') as f:
            f.writelines('\n'.join(lines))
        return accuracy 

    def save_model(self):
        self.model.save('outputs/' +str(self.name) + '/' + str(self.name + '.h5'))

    def load_model(self, name_model):
        self.model = tf.keras.models.load_model('outputs/' +str(self.name) + '/' + str(name_model + '.h5'))


# data = Data(drop_variables=['Rating Bin', 'App Id'], target_variable='Rating Bin')
d_v = ['App Id', 'Bad App Yo', 'Moderate', 'Superb','Size Cat', 'App Name'] #, 'log_Maximum Installs', 'Rating'] # or ['Rating Bin', 'App Id', 'log_Maximum Installs']
t_v = ['Bad App Yo', 'Moderate', 'Superb'] # or 'Rating Bin'
t_v_len = 3
data = Data(drop_variables=d_v, target_variable=t_v)
x_train, x_val, x_test, y_train, y_val, y_test, n_input_features = data.get_data()
x_train, x_val, x_test= data.scaler(sklearn.preprocessing.MinMaxScaler(), x_train, x_val, x_test)

if baseline:
    baselines = [LogisticRegression(), DecisionTreeClassifier(), GradientBoostingClassifier()]
    for classifier in baselines:
        b = Baseline(classifier, 'Classification')
        b.train(x_train, y_train)
        b.test(x_val)
        b.eval(y_val, 'val', True)
        b.test(x_test)
        b.eval(y_test, 'test', True)

if build_NN:
    optimizer = keras.optimizers.Adam(learning_rate=0.001) # perform grid search for multiple learning rates
    model = Network(name='CleverHans4')

    mod_ = model.build(activation_='relu', optimizer_=optimizer, loss_='categorical_crossentropy', 
                output_activation='softmax', n_features_input=n_input_features, n_features_output=t_v_len)
    if only_vis:
        # ann_viz(mod_, view=True, filename='construct_model', title='Clever Hans')
        pass
    else:
        model.train(train_set=x_train, train_labels=y_train, epochs_=15, verbose_=1, val_set=x_val, val_labels=y_val, 
                    batch_size_=64)
        model.get_loss()
        model.test(x_val)
        ev = model.eval(y_val, 'val')
        model.test(x_test)
        ev = model.eval(y_test, 'test')
        if save_model:
            model.save_model()

if load_model:
    model = Network(name='CleverHans10')
    model.load_model(name_model='CleverHans')
    model.test(x_val)
    model.eval(y_val)
    #model.test(x_test)
    #model.eval(y_test)
    