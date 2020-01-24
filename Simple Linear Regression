# Simple Linear  Regression


# dataset  Delivery Time
# predict delivery time using sorting time
# Y = delivery time (Dt), X = sorting time (St)
# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
delt =pd.read_csv("file:///C:/Users/new/Downloads/LinearRegression Assignment/delivery_time.csv")
delt.columns

plt.hist(delt.Dt)
plt.boxplot(delt.Dt,0,"rs",0)


plt.hist(delt.St)
plt.boxplot(delt.St)

plt.plot(delt.Dt,delt.St,"bo");plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")


 # # correlation value between X and Y
np.corrcoef(delt.Dt,delt.St)
# 0.82 moderate corellation between Dt and st consumed

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("delt.Dt~delt.St",data=delt).fit()

# For getting coefficients of the varibles used in equation
model.params
# linear equation is Dt= 6.58+ 1.64 St

# P-values for the variables and R-squared value for prepared model
model.summary()
# P- valve =0   and      R-squared is 0.68 weak

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(pd.DataFrame(delt.Dt)) # Predicted values of cal_consumed using the model
pred
# Visualization of regresion line over the scatter plot of Dt and St
# For visualization we need to import matplotlib.pyplot

import matplotlib.pylab as plt
plt.scatter(x=delt.St,y=delt.Dt ,color='red');plt.plot(delt.Dt,pred,color='black');plt.xlabel('Sorting Time');plt.ylabel('Delivery Time')

pred.corr(delt.Dt)
# r = 0.82 moderately strong correlation (with predicted values)

# Transforming variables for accuracy
model2 = smf.ols('delt.Dt~np.log(delt.St)',data=delt).fit()
model2.params
model2.summary()         # R squared 0.69 weak
print(model2.conf_int(0.01)) # 99% confidence level

pred2 = model2.predict(pd.DataFrame(np.log(delt.St)))

pred2.corr(delt.Dt) # r =0.83 very strong correlation
# pred2 = model2.predict(wtcal.iloc[:,0])

pred2

#Done Till 

plt.scatter(x=np.log(delt.St),y=delt.Dt,color='green');plt.plot(np.log(delt.St),pred2,color='blue');plt.xlabel('lnSTime)');plt.ylabel('Delivery Time)



# Exponential transformation
model3 = smf.ols('np.log(delt.Dt)~delt.St',data=delt).fit()
model3.params
# new equation  Dt = 2.12 + 0.105 St

model3.summary()   # R - squared 0.71  slightly improved

print(model3.conf_int(0.01)) # 99% confidence level

pred_log = model3.predict(pd.DataFrame(delt.St))
pred_log
pred3=np.exp(pred_log)  # as we have used log(Dt) in preparing model so we need to convert it back
pred3
pred3.corr(delt.Dt) # 0.80  moderate strong r correlation


plt.scatter(x=delt.St ,y=delt.Dt ,color='green');plt.plot(delt.St,np.exp(pred_log),color='blue');plt.xlabel('SortingTime');plt.ylabel('DeliveryTime')


resid_3 = pred3- delt.Dt

# so we will consider the model having highest R-Squared value which is the first model- model
# getting residuals of the entire data set

student_resid = model.resid_pearson 
student_resid

plt.plot(model.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred,y=delt.St);plt.xlabel("Predicted");plt.ylabel("Actual")


# done till this ok tested


# Quadratic model

delt["St_Sq"] = delt.St*delt.St

model_quad = smf.ols("delt.Dt ~ delt.St + delt.St*delt.St" ,data=delt).fit()
model_quad.params
model_quad.summary()  # R Squared 0.68 low
pred_quad = model_quad.predict(delt.St)

model_quad.conf_int(0.05) 

plt.scatter(delt.St,delt.Dt,c="b");plt.plot(delt.St,pred_quad,"r")

plt.scatter(np.arange(20),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 

# ok done tested till this

############################### Implementing the Linear Regression model from sklearn library

from sklearn.linear_model import LinearRegression
import numpy as np

plt.scatter(delt.St,delt.Dt)
model1 = LinearRegression()
model1.fit(delt.St.values.reshape(-1,1),delt.Dt)
pred1 = model1.predict(delt.St.values.reshape(-1,1))

# Adjusted R-Squared value

model1.score(delt.St.values.reshape(-1,1),delt.Dt) #  0.68 weak 
rmse1 = np.sqrt(np.mean((pred1-delt.Dt)**2))  
model1.coef_ 
model1.intercept_ #6.58

#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-delt.Dt),c="r")
plt.hlines(y=0,xmin=0,xmax=300) 
# checking normal distribution for residual
plt.hist(pred1-delt.Dt)

#Done till date

### Fitting Quadratic Regression 
delt["St_Sq"] = delt.St*delt.St

model2 = LinearRegression()
model2.fit(X =delt.iloc[:,[0,2]],y=delt.Dt)
pred2 = model2.predict(delt.iloc[:,[0,2]])

# Adjusted R-Squared value
model2.score(delt.iloc[:,[0,2]],delt.Dt)  # 1 very high
rmse2 = np.sqrt(np.mean((pred2-delt.Dt)**2))  # 2.3578785256152632e-15= 0 nearly
model2.coef_
model2.intercept_

#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-delt.Dt),c="r")
plt.hlines(y=0,xmin=0,xmax=200)  

# Checking normal distribution
plt.hist(pred2- delt.Dt)
import pylab
import scipy.stats as st
st.probplot(pred2-delt.Dt,dist="norm",plot=pylab)

# Let us prepare a model by applying transformation on dependent variable
delt["delt.Dt_sqrt"] = np.sqrt(delt.Dt)

model3 = LinearRegression()
model3.fit(X = delt.iloc[:,[0,2]],y=delt["delt.Dt_sqrt"])
pred3 = model3.predict(delt.iloc[:,[0,2]])

# Adjusted R-Squared value
model3.score(delt.iloc[:,[0,2]],delt["delt.Dt_sqrt"] ) # 0.99
rmse3 = np.sqrt(np.mean(((pred3)**2-delt.Dt)**2)) 
model3.coef_   
model3.intercept_

#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred3)**2,((pred3)**2-delt.Dt),c="r")
plt.hlines(y=0,xmin=0,xmax=300)  
# checking normal distribution for residuals 



plt.hist((pred3)**2-delt.Dt )
st.probplot((pred3)**2-delt.Dt ,dist="norm",plot=pylab)


# Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
model4 = LinearRegression()
model4.fit(X = delt.St.values.reshape(-1,1),y=delt['delt.Dt_sqrt'])
pred4 = model4.predict(delt.St.values.reshape(-1,1))
pred4

# Adjusted R-Squared value
model4.score(delt.St.values.reshape(-1,1),y=delt['delt.Dt_sqrt']) #0.70 
rmse4 = np.sqrt(np.mean(((pred4)**2)-(delt.Dt)**2))
model4.coef_
model4.intercept_


#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred4)**2,((pred4)**2-delt.Dt),c="r")
plt.hlines(y=0,xmin=0,xmax=300)  

st.probplot((pred4)**2-delt.Dt,dist="norm",plot=pylab)

# Checking normal distribution for residuals 
plt.hist((pred4)**2- delt.Dt)

