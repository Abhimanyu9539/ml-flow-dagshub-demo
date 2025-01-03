import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import numpy as np

# Initialize DagsHub
dagshub.init(repo_owner='Abhimanyu9539', repo_name='ml-flow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Abhimanyu9539/ml-flow-dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Decision Tree model
max_depth = 3  # Increased from 1 for better performance
random_state = 42
n_estimators = 10

# Set MLflow experiment
mlflow.set_experiment('iris-rf')

with mlflow.start_run():
    
    # Create and train the model
    rf = RandomForestClassifier(max_depth=max_depth, random_state=random_state,n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log parameters
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('random_state', random_state)
    mlflow.log_param('test_size', 0.2)
    mlflow.log_param('dataset_size', len(X))
    
    # Log metrics
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)
    
    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - RF (max_depth={max_depth})')
    plt.tight_layout()
    
    # Save and log the confusion matrix
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()  # Close the plot to free memory
    
    # Create feature importance plot
    feature_importance = rf.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(iris.feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    
    # Save and log feature importance plot
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # Log classification report as text
    class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
    with open("classification_report.txt", "w") as f:
        f.write(class_report)
    mlflow.log_artifact("classification_report.txt")
    
    # Log the source code
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        # Handle case when running in Jupyter notebook
        print("Note: Running in notebook environment, source code not logged")
    
    # Log the model
    mlflow.sklearn.log_model(
        rf, 
        artifact_path="model",
        registered_model_name="iris_random_forest"
    )
    
    # Set tags
    mlflow.set_tag('author', 'Abhimanyu')
    mlflow.set_tag('model', 'Random Forest')
    mlflow.set_tag('dataset', 'iris')
    mlflow.set_tag('algorithm', 'RandomForestClassifier')
    
    # Print results
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Max Depth: {max_depth}')
    
    print("\nClassification Report:")
    print(class_report)