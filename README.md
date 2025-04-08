1. Import Libraries:
python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
We import the necessary libraries:

numpy: For numerical operations (manipulating arrays).

sklearn: For generating data, using a classifier (RandomForestClassifier), splitting the data, and calculating accuracy.

2. Generate Synthetic Data:
python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
make_classification generates a synthetic dataset with:

1000 samples (n_samples=1000).

20 features per sample (n_features=20), with 15 being informative (n_informative=15).

The random_state=42 ensures reproducibility.

X contains the features, and y contains the corresponding labels.

3. Split Data:
python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)
Here’s how we split the data:

80-20 Split: The dataset is divided into X_train/y_train (80%) and X_test/y_test (20%) for training and testing.

Labeled vs Unlabeled:

The training set is further split into 30% labeled (X_labeled, y_labeled) and 70% unlabeled (X_unlabeled).

4. Train Initial Model:
python
model = RandomForestClassifier(random_state=42)
model.fit(X_labeled, y_labeled)
We initialize a Random Forest Classifier.

We train it (fit) on the labeled data (X_labeled, y_labeled).

5. Self-Training Loop:
python
max_iterations = 10
confidence_threshold = 0.8
for iteration in range(max_iterations):
We limit the number of self-training iterations with max_iterations=10.

confidence_threshold=0.8: The model must be 80% confident in its predictions on unlabeled data before adding them.

Main Steps Inside the Loop:

Predict Unlabeled Data:

python
probabilities = model.predict_proba(X_unlabeled)
max_probs = np.max(probabilities, axis=1)
confident_indices = np.where(max_probs > confidence_threshold)[0]
predict_proba: Predicts class probabilities for each sample in X_unlabeled.

max_probs: Extracts the highest probability for each prediction.

confident_indices: Finds samples where the confidence exceeds the threshold (0.8).

Check Confident Predictions:

python
if len(confident_indices) == 0:
    print("No confident predictions in iteration", iteration)
    break
If there are no predictions above the confidence threshold, stop the loop.

Expand Labeled Dataset:

python
X_confident = X_unlabeled[confident_indices]
y_confident = model.predict(X_unlabeled)[confident_indices]

X_labeled = np.vstack((X_labeled, X_confident))
y_labeled = np.hstack((y_labeled, y_confident))
Extract confidently predicted samples (X_confident) and their predicted labels (y_confident).

Add these samples to the labeled dataset (X_labeled, y_labeled).

Remove Confident Samples from Unlabeled Data:

python
X_unlabeled = np.delete(X_unlabeled, confident_indices, axis=0)
Remove samples that were added to the labeled dataset from X_unlabeled.

Retrain the Model:

python
model.fit(X_labeled, y_labeled)
Retrain the model on the expanded labeled dataset.

6. Evaluate the Model:
python
y_pred = model.predict(X_test)
print("Accuracy on the test set:", accuracy_score(y_test, y_pred))
Evaluate the model's performance on the test set (X_test/y_test).

accuracy_score: Calculates the accuracy of predictions.
