import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

data = pd.read_csv("adult.csv")

#blank the epty spaces of the data ?? doesnt it creates huge bias
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

#give the bianry value to the results
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'native-country']
data = pd.get_dummies(data, columns=categorical_cols)

#the num f people this ceratin class represnt ı dont thik it is important
data.drop('fnlwgt', axis=1, inplace=True)

#target and features
X = data.drop('income', axis=1)
y = data['income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create the model like in iris mdoel
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")
