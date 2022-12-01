import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# read data
df = pd.read_csv("data/features_30_sec.csv")
X = df.iloc[:,1:-1].to_numpy()
y = df['label']
y = y.replace(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# split train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

print("Train accuracy:",train_accuracy)
print("Test accuracy:", test_accuracy)

y_pred = knn.predict(X_test)
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print(classification_report(y_test, y_pred))
