#prediction of percentage of an student based on no. of study hours

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv("student_scores - student_scores.csv")
print(df.head())

X = df.iloc[:,:1].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(X, line)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

y_pred = regressor.predict(X_test)
acc_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(acc_df)
#or
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print("\n")

target_hours = 9.25
pred_score = regressor.predict([[target_hours]])
print("Predicted Score = ",pred_score[0])
