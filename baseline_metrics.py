import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def baseline_classifiers(csv_in):
    
    df = pd.read_csv(csv_in)
    x = df.drop(['label', 'filename'], axis=1)
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=45, stratify=y)

    print("*** SUPPORT VECTOR MACHINE ***")
    svclassifier = SVC()
    svclassifier.fit(x_train, y_train)
    train_accuracy = svclassifier.score(x_train, y_train)
    test_accuracy = svclassifier.score(x_test, y_test)
    print("Train accuracy:",train_accuracy)
    print("Test accuracy:", test_accuracy)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    print("*** POLYNOMIAL KERNEL ***")
    svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(x_train, y_train)
    train_accuracy = svclassifier.score(x_train, y_train)
    test_accuracy = svclassifier.score(x_test, y_test)
    print("Train accuracy:",train_accuracy)
    print("Test accuracy:", test_accuracy)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("*** GAUSSIAN KERNEL ***")
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(x_train, y_train)
    train_accuracy = svclassifier.score(x_train, y_train)
    test_accuracy = svclassifier.score(x_test, y_test)
    print("Train accuracy:",train_accuracy)
    print("Test accuracy:", test_accuracy)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("*** SIGMOID KERNEL ***")
    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(x_train, y_train)
    train_accuracy = svclassifier.score(x_train, y_train)
    test_accuracy = svclassifier.score(x_test, y_test)
    print("Train accuracy:",train_accuracy)
    print("Test accuracy:", test_accuracy)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    print("*** K NEAREST NEIGHBOR ***")
    y = y.replace(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(x_train, y_train)
    train_accuracy = knn.score(x_train, y_train)
    test_accuracy = knn.score(x_test, y_test)
    print("Train accuracy:",train_accuracy)
    print("Test accuracy:", test_accuracy)
    y_pred = knn.predict(x_test)
    print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print(classification_report(y_test, y_pred))

    print("*** MULTI LAYERED PERCEPTRON ***")
    #shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    mlp.fit(X_train, y_train)
    train_accuracy = mlp.score(X_train, y_train)
    test_accuracy = mlp.score(X_test, y_test)
    print("Train accuracy:",train_accuracy)
    print("Test accuracy:", test_accuracy)
    predictions = mlp.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))



baseline_classifiers("data/features_30_sec.csv")
