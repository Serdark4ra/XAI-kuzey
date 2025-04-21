import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def load_crop_data(file_path):

    df = pd.read_csv(file_path)

    df = pd.get_dummies(df, columns=['season']) # 0,0,1 ya da 0,1,0 ya da 1,0,0 a çeviriyor

    # x burada feature matrix
    # .values array'e dönüştürüyor
    X = df.drop('label', axis=1).values # başlıkları silip sadece datalara dönüşütüyor

    #label yani sonuçları integer yapıyor
    encoder = LabelEncoder()
    # çevirmeyi burada yapıyot "label" columnu üzerinde
    y = encoder.fit_transform(df['label'])

    # convert int to one-hot vector
    # probability için VE SOFTMAX LAYER KULLANIYORSAN ZORUNLU**
    y_one_hot = tf.keras.utils.to_categorical(y)

    # store original names for label classes like rice wheat etc
    oldnames = encoder.classes_

    return X, y_one_hot, oldnames

def build_model(input_shape, num_classes):

    # input_shape = num of features
    # num classes = num of output classes
    model = Sequential()

    # input shape must be specif,ad in first layer
    model.add(Dense(16, activation='sigmoid', input_shape=(input_shape,)))
    model.add(Dense(12, activation='relu'))

    # also for prevent overfitting adding dropout is necessary but in this case no need

    # final layer with 5 neurons
    # softmax turns results to 0-1 probability
    model.add(Dense(num_classes, activation='softmax'))

    # loss='categorical_crossentropy' is used as I also used one-hot encoding
    # optimizer='adam' ne anlamadım ??????
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# X train input
# Y train one-hot encoded labels
# epochs kaç kere sampleların üstüdnen geçileceği (override!)
# batch size anlamadım???
def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # validation data helps to check overfitting (memorization) also used to check how well model trained
    return history

def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.show()

def main(file_path):

    # x for feature matrix
    # y for one-hot encoding
    # target names for remembering label names
    X, y, target_names = load_crop_data(file_path)

    # split data %80 goes for train. %20 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # standardization, make every gap same bc we do not want to confuse neurons, increases accuracy
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # build and train model
    model = build_model(X_train.shape[1], y.shape[1])
    print("Model Summary:")
    model.summary()

    history = train_model(model, X_train, y_train, X_test, y_test, epochs=100)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc:.4f}")
    plot_history(history)

    # Make an example prediction taken from GPT
    example = X_test[0:1]
    prediction = model.predict(example)
    predicted_class = np.argmax(prediction)
    print("\nExample Prediction:")
    print(f"Predicted Crop: {target_names[predicted_class]}")
    print(f"Probability Distribution: {prediction[0]}")

    # Save the model
    model.save("crop_model.h5")
    print("\nModel saved to 'crop_model.h5'")

# === Entry point ===
if __name__ == "__main__":
    main("Crop_recommendation.csv")
