import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
le = LabelEncoder()

veriler1 = pd.read_excel("merc_prediction/merc.xlsx")



veriler3 = veriler1.apply(le.fit_transform)

b = veriler3.iloc[:,-5].to_frame()

b = ohe.fit_transform(b).toarray()

Transmission = pd.DataFrame(data=b, columns = ["a","s","m","o"])


lastver = pd.concat([Transmission , veriler1], axis=1)

lastver = lastver.drop(columns="transmission")

lastver = lastver.sort_values("price", ascending=False).iloc[131:-1]

# time to p val
 

#import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

price = lastver.iloc[:,5:6]
left = lastver.iloc[:,:5]
right = lastver.iloc[:,6:]
data = pd.concat([left,right],axis=1)

x_train, x_test, y_train, y_test = train_test_split(data,price,test_size=0.33,random_state=10)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#print(x_train.shape)

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error

model.fit(x=x_train, y = y_train,validation_data=(x_test,y_test),batch_size=125,epochs=1000)
#epochs ve batch size'ı değiştirerek mae'u düşürmek ?

# loss and prediction sequence

lossData = pd.DataFrame(model.history.history)
print(lossData.head())

prediction_sequence = model.predict(x_test)


print(mean_absolute_error(y_test,prediction_sequence))
print(mean_absolute_percentage_error(y_test,prediction_sequence))
plt.scatter(y_test,prediction_sequence)
plt.plot(y_test,y_test,"r-*")
plt.show()




"""
X = np.append(arr = np.ones((12987,1)).astype(int), values=data, axis=1 )

X_l = data.iloc[:,[0,1,2,3,4,5,6,7,8]].values

X_l = np.array(X_l,dtype=float)

model = sm.OLS(price,X_l).fit()


print(model.summary())
"""
