import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def baseline_classifiers(csv_in):
    
    df = pd.read_csv(csv_in)
    x = df.drop(['label', 'filename'], axis=1)
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=45, stratify=y)

    print("*** SUPPORT VECTOR MACHINE ***")
    svclassifier = SVC()
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    print("*** POLYNOMIAL KERNEL ***")
    svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("*** GAUSSIAN KERNEL ***")
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("*** SIGMOID KERNEL ***")
    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(x_train, y_train)
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




baseline_classifiers("features_30_sec.csv")
