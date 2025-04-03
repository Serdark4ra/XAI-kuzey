import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Function to load your custom iris.csv file
def load_custom_iris(file_path):
    """Load iris dataset from a custom CSV file"""
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract features and target
    # Assuming your CSV has columns: sepal_length, sepal_width, petal_length, petal_width, species
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

    # Convert species to numerical values
    species_mapping = {name: i for i, name in enumerate(df['species'].unique())}
    y = df['species'].map(species_mapping).values

    # Store target names for later reference
    target_names = list(species_mapping.keys())

    return X, y, target_names


class SimpleIrisClassifier:
    def __init__(self):
        # Initialize parameters for each class (means and counts)
        self.class_means = {}
        self.class_counts = {}
        self.classes = None

    def fit(self, X, y):
        """Train the model using training data"""
        self.classes = np.unique(y)

        # Calculate mean feature values for each class
        for c in self.classes:
            # Get all samples from this class
            class_samples = X[y == c]
            # Calculate mean for each feature
            self.class_means[c] = np.mean(class_samples, axis=0)
            # Count samples in this class
            self.class_counts[c] = len(class_samples)

        return self

    def predict(self, X):
        """Predict the class for each sample in X"""
        predictions = []

        for sample in X:
            # For each sample, calculate Euclidean distance to each class mean
            distances = {}
            for c in self.classes:
                # Euclidean distance between sample and class mean
                distances[c] = np.sqrt(np.sum((sample - self.class_means[c]) ** 2))

            # Predict the class with minimum distance
            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """Calculate accuracy of the model"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


# Main execution
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "/Users/serdarkara/Desktop/Python/XAI-kuzey/XAI-kuzey/iris.csv"

    try:
        # Load your custom iris data
        X, y, target_names = load_custom_iris(file_path)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train our model
        model = SimpleIrisClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        training_accuracy = model.score(X_train, y_train)
        testing_accuracy = model.score(X_test, y_test)

        print(f"Training accuracy: {training_accuracy:.4f}")
        print(f"Testing accuracy: {testing_accuracy:.4f}")


        # Visualize the results (for 2 features only)
        def plot_decision_boundary(model, X, y):
            # We'll only use the first two features for visualization
            h = 0.02  # step size in the mesh

            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            # Create samples for the whole mesh
            Z = model.predict(np.c_[xx.ravel(), yy.ravel(),
            np.ones(xx.ravel().shape) * np.mean(X[:, 2]),
            np.ones(xx.ravel().shape) * np.mean(X[:, 3])])
            Z = Z.reshape(xx.shape)

            # Plot the decision boundary
            plt.figure(figsize=(10, 8))
            plt.contourf(xx, yy, Z, alpha=0.3)

            # Plot the training points
            for i, c in enumerate(np.unique(y)):
                idx = np.where(y == c)
                plt.scatter(X[idx, 0], X[idx, 1], label=target_names[c])

            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
            plt.title('Decision Boundary using First Two Features')
            plt.legend()
            plt.show()


        # Visualize decision boundary
        plot_decision_boundary(model, X, y)


        # Function to classify new samples
        def classify_new_sample(model, sample):
            """Classify a new sample and explain the decision"""
            sample_array = np.array(sample).reshape(1, -1)
            predicted_class_id = model.predict(sample_array)[0]
            predicted_class = target_names[predicted_class_id]

            print(f"Sample: {sample}")
            print(f"Predicted class: {predicted_class}")

            # Calculate and show distances to each class mean
            for c in model.classes:
                class_name = target_names[c]
                distance = np.sqrt(np.sum((sample - model.class_means[c]) ** 2))
                print(f"Distance to {class_name} mean: {distance:.4f}")

            return predicted_class


        # Example of using the model on a new sample
        sample = [5.1, 3.5, 1.4, 0.2]  # Example sample
        classify_new_sample(model, sample)

        # Analyze feature importance
        print("\nFeature importance analysis:")
        feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

        for i, feature in enumerate(feature_names):
            class_values = [model.class_means[c][i] for c in model.classes]
            min_val, max_val = min(class_values), max(class_values)
            importance = max_val - min_val
            print(f"{feature}: Mean difference across classes = {importance:.4f}")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        print("Please make sure your iris.csv file is in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Make sure your CSV file has the expected format with columns: sepal_length, sepal_width, petal_length, petal_width, species")


