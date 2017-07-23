'''
Working on Neural Networks!
Data set from Kaggle: <link>
Accuracy for the other models is around 75%, but the neural network model has an accuracy of 35%
Have I built the model wrong, or does it have something to do with the dataset?
'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.api import OLS
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, rmsprop


def load_data():
    df = pd.read_csv('train.csv')
    y = df.pop('type')
    df = df.join(pd.get_dummies(df.color))
    df.pop('color')
    return df, y

def model_fit(model, X, y):
    score = np.mean(cross_val_score(estimator= model,X= X, y= y, scoring='accuracy', cv= 5 ))
    model.fit(X,y)
    return score, model

def create_model(X, y, nodes, unique_labels, optimizer):
    ''' nodes - nodes in each layer except the last layer'''
    model = Sequential()
    for idx, node_value in enumerate(nodes):
        if idx == 0:
            model.add(Dense(node_value, activation = 'relu', input_dim = X.shape[1]))
        else:
            model.add(Dense(node_value, activation = 'relu'))
    model.add(Dense(unique_labels, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics= ['accuracy'])
    model.fit(X, y)
    return model

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X.values,y.values)
    logreg = LogisticRegression(solver='newton-cg')
    dt = DecisionTreeClassifier(max_depth= 5)
    rf = RandomForestClassifier(max_depth=4, max_features=6, n_estimators= 1000)
    gbc = GradientBoostingClassifier(learning_rate= 0.0001, n_estimators= 1000, max_depth=4)
    abc = AdaBoostClassifier(learning_rate= 0.001, n_estimators = 10000)
    # Printing cross validation score for all models above
    for i in [logreg, dt, rf, gbc, abc]:
        print(i.__class__.__name__ +':'+'{}'.format(model_fit(i,X_train, y_train)[0]))

    nodes = np.ones(2, dtype='int64') * 64
    sgd = SGD(lr=0.0001)
    nn = create_model(X_train, pd.get_dummies(y_train).values, nodes, 3,  sgd)
    sc_nn = nn.evaluate(X_test, pd.get_dummies(y_test).values)
    print('\n'+nn.__class__.__name__+':'+'{}'.format(sc_nn[1]))
