import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_iris_data(file_path=None):
    if file_path:
        try:
            # Load from custom CSV
            df = pd.read_csv(file_path)
            X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

            # Encode species to numerical values
            encoder = LabelEncoder()
            y = encoder.fit_transform(df['species'])
            target_names = encoder.classes_

        except Exception as e:
            print(f"Error loading custom CSV: {e}")
            print("Falling back to built-in dataset...")
            return load_iris_data(None)
    else:
        # Use the built-in dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        target_names = iris.target_names

    # We need one-hot encoded labels for softmax output
    y_one_hot = tf.keras.utils.to_categorical(y)

    return X, y_one_hot, target_names


def build_model(input_shape, num_classes):
    model = Sequential()
    # First dense layer
    model.add(Dense(10, activation='relu', input_shape=(input_shape,)))
    # Second dense layer
    model.add(Dense(8, activation='relu'))
    # Output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Train the model and return history
def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history


# Main execution function
def main(file_path=None):
    # Load data
    X, y, target_names = load_iris_data(file_path)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.18, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build model
    num_classes = y.shape[1]
    model = build_model(X_train.shape[1], num_classes)

    # Print model summary
    print("Model Architecture:")
    model.summary()

    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=100)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Example prediction
    example = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example Iris measurement
    example_scaled = scaler.transform(example)
    prediction = model.predict(example_scaled)
    predicted_class = np.argmax(prediction)

    print("\nExample Prediction:")
    print(f"Input: {example[0]}")
    print(f"Predicted class: {target_names[predicted_class]}")
    print(f"Probability distribution: {prediction[0]}")


if __name__ == "__main__":
<<<<<<< Updated upstream:irisDataset-model/a.py
    main("/Users/serdarkara/Desktop/Python/XAI-kuzey/XAI-kuzey/iris.csv")
=======
    # Change to your CSV file path or leave as None to use built-in dataset
    file_path = None  # "iris.csv"
    main("iris.csv")
>>>>>>> Stashed changes:a.py
