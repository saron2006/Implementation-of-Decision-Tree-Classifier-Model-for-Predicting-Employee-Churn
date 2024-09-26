# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: SARON XAVIER A

RegisterNumber:212223230197
```python
import pandas as pd


data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = df.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])
```

## Output:
![Image-1](https://github.com/user-attachments/assets/7a9d26dd-ae76-4835-90df-0e8579483194)

![Image-2](https://github.com/user-attachments/assets/941e1c4f-7707-41d2-92fa-c907668565f0)

![Image-3](https://github.com/user-attachments/assets/b4837a26-d0a0-4450-89ba-e021d94b794d)

![Image-4](https://github.com/user-attachments/assets/bd78dea0-8230-4ebe-a4c2-7aa47f3a4fdc)

![Image-5](https://github.com/user-attachments/assets/09228d96-1f7a-4ab6-a1eb-f097cfe3946a)

![Image-6](https://github.com/user-attachments/assets/c4364ae1-5e56-4db3-a0f8-89cf2a43fe70)

![Image-7](https://github.com/user-attachments/assets/a12adcdf-1e1a-4354-8243-f26234da9f53)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
