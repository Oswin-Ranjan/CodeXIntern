import pandas as pd
import seaborn as sns
import numpy as nd
import matplotlib.pyplot as plt

#Loading the dataset and exploring it visually(Scatter plots)

df = pd.read_table(r'C:\Users\ranja\OneDrive\Desktop\CODEXINTERN\Iris Flower Classification/iris.data',sep=',')
df.columns = ['Sepal length(cm)', 'Sepal width(cm)', 'Petal length(cm)', 'Petal width(cm)', 'Class']
print(df.head(150))

sns.FacetGrid(df, hue="Class") \
   .map(plt.scatter, "Sepal length(cm)", "Sepal width(cm)") \
   .add_legend()
plt.show()

sns.FacetGrid(df, hue="Class") \
   .map(plt.scatter, "Petal length(cm)", "Petal width(cm)") \
   .add_legend()
plt.show()

sns.FacetGrid(df, hue="Class") \
   .map(plt.scatter, "Sepal width(cm)", "Sepal length(cm)") \
   .add_legend()
plt.show()

sns.FacetGrid(df, hue="Class") \
   .map(plt.scatter, "Petal width(cm)", "Petal length(cm)") \
   .add_legend()
plt.show()

#Splitting the dataset 

from sklearn.datasets import load_iris
iris=load_iris()
for keys in iris.keys() :
    print(keys)
    
X=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(type(iris.data))
print(iris.data)
print(y)

iris['feature_names']
print(len(iris.data))
print(len(iris.target))

#Data is already clean and preprocessed

#Training the model and evaluating it
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.preprocessing import StandardScaler

lrc = LogisticRegression(solver='liblinear', penalty='l1')
knc = KNeighborsClassifier()
dtc = DecisionTreeClassifier(max_depth=5)

clfs = {
    'LRC':lrc,
    'KNC':knc,
    'DTC':dtc
}

def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average='macro')
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy,precision,confusion_mat
  
accuracy_scores = []
precision_scores = []
confusion_scores = []

for name,clf in clfs.items():

    current_accuracy,current_precision, current_confusion = train_classifier(clf, X_train,y_train,X_test,y_test)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    confusion_scores.append(current_confusion)
    
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores,'Confusion Matrix':confusion_scores}).sort_values('Accuracy',ascending=False)
print(performance_df)

##Feature Scaling and retraining Logistic Regression

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#Predicting a sample(User Input)

sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

sample = [[sepal_length, sepal_width, petal_length, petal_width]]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print("Predicted flower type:", iris.target_names[prediction[0]])
