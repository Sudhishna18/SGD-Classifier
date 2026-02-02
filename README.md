# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Data
2. Initialize the Model
3. Train the Model
4. Test and Evaluate

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SUDHISHNA P
RegisterNumber:  212224040336
*/
```
```
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
iris=load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
clf=SGDClassifier(loss="log_loss",max_iter=1000,tol=1e-3,random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

```

## Output:
<img width="333" height="164" alt="exp7 op" src="https://github.com/user-attachments/assets/631f3703-6de4-4748-ac26-73489c24ba62" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
