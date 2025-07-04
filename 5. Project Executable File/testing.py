from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Define the models with better default parameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)": SVC(gamma='auto', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Map the target variable values from [-1, 0, 1] to [0, 1, 2]
y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})

# Create a dictionary to store results
results = {}

for name, model in models.items():
    print(f"\n\033[1m{name}\033[0m")
    print("="*60)
    
    # Fit and predict
    if name in ["Logistic Regression", "SVM (RBF)", "KNN"]:
        model.fit(X_train_scaled, y_train_mapped)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train_mapped)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_mapped, y_pred)
    f1 = f1_score(y_test_mapped, y_pred, average='weighted')
    cm = confusion_matrix(y_test_mapped, y_pred)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Enhanced confusion matrix visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0 (-1)', 'Class 1 (0)', 'Class 2 (1)'],
                yticklabels=['Class 0 (-1)', 'Class 1 (0)', 'Class 2 (1)'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test_mapped, y_pred, 
                               target_names=['Class 0 (-1)', 'Class 1 (0)', 'Class 2 (1)']))
    print("-"*60)

# Compare model performances
print("\n\033[1mModel Performance Comparison\033[0m")
print("="*60)
for name, metrics in results.items():
    print(f"{name:<20} | Accuracy: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1_score']:.4f}")