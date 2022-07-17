
#####################################################################
#                                                                   #
#            LOGISTIC REGRESSION MODEL                              #                                                       #
#                                                                   #





#The python libraries required for this specific task are shown below
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


#This is how to open the data using pandas module and usecols is used to select specific columns you want to use.
df = pd.read_csv('C:\\Users\PC\\Downloads\\diabetes.csv', usecols = ['Glucose','Age', 'Outcome','BloodPressure'])



print(df.head)    #Displays the first 5 and last 5 rows and columns.
print(df.shape)   #Displays the number of rows and columns in row x column format
print(df.columns) #Displays all the column names


            
#     Uncommenting the code below allows you to visualize the last column "Outcome"
#     with matplotlib module and confirms that it has binary values thus fitting a logistic curve would be appropriate
#     for this prediction
#df[df.columns[-1]].hist(alpha=0.8, bins= 20, density=True)
#plt.show()


# The .dropna() method is used to drop all the values with empty values before fitting the model 
df= df.dropna( subset=['Glucose','Age', 'Outcome','BloodPressure'])


#this block splits the data into features(X) and Target(Y)
X = df[df.columns[:-1]].values
Y = df[[df.columns[-1]]].values

#instatiating the standard scaler
scale = StandardScaler()

#overcome overfitting and get more accurate results.
scaled_x = scale.fit_transform(X)


# this line splits the the data into train and test sets
x_test, x_train, y_test, y_train = train_test_split(scaled_x, Y, random_state=0, test_size = .75)


#instatiating logistic regression
model = LogisticRegression() 


# .ravel method changes the shape of y to remove data conversion anormalies
model.fit(x_train,y_train.ravel())


#prints out the coefficients in the model and the intercept
print("Evaluated Model coefficients:",model.coef_)
print("Evaluated Model intercepts:",model.intercept_)




#using the model to predict given our test data feature
predict_test = model.predict(x_test)
print("Evaluated test_predictions:", predict_test)

#Determining an accuracy score of the model given our test target
accuracy_test = accuracy_score(y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)




