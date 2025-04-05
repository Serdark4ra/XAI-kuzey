import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt


# Function to load the Iris dataset (either from CSV or use the built-in dataset)
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


# Build the model from scratch using TensorFlow
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


# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()


# Visualize predictions on a 2D plot
def plot_decision_boundary(model, X, y_encoded, target_names):
    # We'll visualize using the first two features
    h = 0.02  # step size

    # Scale features to help with visualization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create meshgrid for the first two features
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Create a grid with the mean values for the remaining features
    grid = np.zeros((xx.ravel().shape[0], X.shape[1]))
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()
    # Fill the remaining features with mean values
    for i in range(2, X.shape[1]):
        grid[:, i] = np.mean(X_scaled[:, i])

    # Get predictions for the grid
    Z = model.predict(grid)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    # Get the original classes from one-hot encoded values
    y_classes = np.argmax(y_encoded, axis=1)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)

    # Plot training points
    for i, c in enumerate(np.unique(y_classes)):
        idx = np.where(y_classes == c)
        plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1],
                    label=target_names[c], edgecolor='k', s=40)

    plt.xlabel('Sepal Length (scaled)')
    plt.ylabel('Sepal Width (scaled)')
    plt.title('Decision Boundaries on First Two Features')
    plt.legend()
    plt.show()


# Main execution function
def main(file_path=None):
    # Load data
    X, y, target_names = load_iris_data(file_path)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    # Plot training history
    plot_history(history)

    # Visualize decision boundaries
    plot_decision_boundary(model, X, y, target_names)

    # Example prediction
    example = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example Iris measurement
    example_scaled = scaler.transform(example)
    prediction = model.predict(example_scaled)
    predicted_class = np.argmax(prediction)

    print("\nExample Prediction:")
    print(f"Input: {example[0]}")
    print(f"Predicted class: {target_names[predicted_class]}")
    print(f"Probability distribution: {prediction[0]}")

    # Save the model
    model.save("iris_model.h5")
    print("\nModel saved to 'iris_model.h5'")


# Run the main function if script is executed directly
if __name__ == "__main__":
    # Change to your CSV file path or leave as None to use built-in dataset
    file_path = None  # "iris.csv"
    main("/Users/serdarkara/Desktop/Python/XAI-kuzey/XAI-kuzey/iris.csv")