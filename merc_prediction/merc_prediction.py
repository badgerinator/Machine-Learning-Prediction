import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_excel("merc.xlsx")

editedDF = dataFrame.sort_values("price",ascending = False).iloc[131:]

dataFrame = editedDF

# outlier değerlerden kurtulmak için df'nin price column'undaki 1 percent kısmını silmek ?

editedDF = editedDF[editedDF.year != 1970]

# Transmission'ı numerik verilere çevirmeyi deniyorum

"""
editedDF["transmission"].replace({"Automatic":"2","Semi-Auto":"3","Manual":"1"}, inplace = True)
try:
    pd.to_numeric(editedDF["transmission"])
except:
    pass

"""
# galiba başardım

# BAŞARAMADIK ABİĞĞĞ

# bu yüzden transmission column'unu siliyoruz :)

editedDF = editedDF.drop("transmission",axis=1)

# şimdi de x,y belirleyip train test split'i ayarlamak ?

x = editedDF.drop("price",axis=1).values
y = editedDF["price"].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#scaler ile değerler ayarlanıldı, 0.33-0.66 train test split set

# model oluşturuluyor...

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print(x_train.shape)

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

from sklearn.metrics import mean_squared_error, mean_absolute_error

model.fit(x=x_train, y = y_train,validation_data=(x_test,y_test),batch_size=275,epochs=300)

# loss and prediction sequence

lossData = pd.DataFrame(model.history.history)
print(lossData.head())

prediction_sequence = model.predict(x_test)


print(mean_absolute_error(y_test,prediction_sequence))
plt.scatter(y_test,prediction_sequence)
plt.plot(y_test,y_test,"r-*")
plt.show()

print(editedDF)

# This DUMB BAD CODE DOESN'T MAKE ANY SENSE
#       BUT IT WORKS FINE FOR NOW

#   ERROR PERCENTAGE KINDA HIGH BUT;
#       WHO CARES ??!!11!
