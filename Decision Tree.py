import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from six import StringIO
from six import StringIO
from IPython.display import Image
import pydotplus
import graphviz
df=pd.read_csv('iris.csv')
print(df.head())
print(df.describe())
print(df.info)
df=df.drop(["Id"],axis=1)
print(df.head())
X=df.drop(["Species"],axis=1)
Y=df["Species"]
print(X)
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
dot_data = StringIO()
fig = plt.subplots(figsize=(12, 8))
tree.plot_tree(clf,
                           feature_names=df.iloc[:, :-1].columns,
                           class_names=['0','1','2'],
                           filled=True, rounded=True)
plt.show()



