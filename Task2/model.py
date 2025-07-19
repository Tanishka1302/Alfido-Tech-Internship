import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset (you can use your own CSV too)
data = pd.DataFrame({
    'Age': [25, 30, 45, 35, 50],
    'Salary': [50000, 60000, 80000, 72000, 90000],
    'Purchased': [0, 1, 1, 0, 1]
})

X = data[['Age', 'Salary']]
y = data['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
