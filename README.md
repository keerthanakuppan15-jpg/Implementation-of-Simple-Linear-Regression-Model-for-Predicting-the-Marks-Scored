# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph
6.Compare the graphs and hence we obtained the linear regression for the given datas.
  
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KEERTHANA K
RegisterNumber: 212225230137 
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  
Y = np.array([35, 40, 50, 55, 65])           
model = LinearRegression()
model.fit(X, Y)
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

Y_pred = model.predict(X)

hours = np.array([[6]])
predicted_marks = model.predict(hours)
print("Predicted marks for 6 hours of study:", predicted_marks[0])
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression Model")
plt.legend()
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="865" height="670" alt="image" src="https://github.com/user-attachments/assets/cefb35e5-4f39-475f-a086-b009189bf8db" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
