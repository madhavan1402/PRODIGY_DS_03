import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load datasets
data_bank = pd.read_csv("bank.csv", sep=';')
data_bank_full = pd.read_csv("bank-full.csv", sep=';')

# Display information about datasets
print("bank.csv info:")
data_bank.info()
print("\nbank-full.csv info:")
data_bank_full.info()

# Load bank-names.txt (optional metadata or description)
with open("bank-names.txt", "r") as f:
    bank_names_content = f.read()
print("\nbank-names.txt content:")
print(bank_names_content)

# Encode categorical variables for bank.csv
data_bank_encoded = pd.get_dummies(data_bank, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'], drop_first=True)
data_bank_full_encoded = pd.get_dummies(data_bank_full, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'], drop_first=True)

# Define features (X) and target (y) for bank.csv
X_bank = data_bank_encoded.drop(columns=[col for col in data_bank_encoded.columns if col.startswith('y_')])
y_bank = data_bank_encoded[[col for col in data_bank_encoded.columns if col.startswith('y_')]].iloc[:, 0]  # Binary target variable

# Split dataset bank.csv into training and testing sets
X_bank_train, X_bank_test, y_bank_train, y_bank_test = train_test_split(X_bank, y_bank, test_size=0.2, random_state=42)

# Train Decision Tree Classifier for bank.csv
clf_bank = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_bank.fit(X_bank_train, y_bank_train)

# Evaluate the model for bank.csv
y_bank_pred = clf_bank.predict(X_bank_test)
print("\nBank.csv Dataset Results:")
print("Accuracy:", accuracy_score(y_bank_test, y_bank_pred))
print("Classification Report:\n", classification_report(y_bank_test, y_bank_pred))

# Visualize the decision tree for bank.csv and save the figure
plt.figure(figsize=(20, 10))
plot_tree(clf_bank, feature_names=X_bank.columns, class_names=['no', 'yes'], filled=True)
plt.title("Decision Tree - bank.csv")
plt.savefig("decision_tree_bank.png")
plt.close()

# Repeat similar processing for bank-full.csv
X_bank_full = data_bank_full_encoded.drop(columns=[col for col in data_bank_full_encoded.columns if col.startswith('y_')])
y_bank_full = data_bank_full_encoded[[col for col in data_bank_full_encoded.columns if col.startswith('y_')]].iloc[:, 0]

X_bank_full_train, X_bank_full_test, y_bank_full_train, y_bank_full_test = train_test_split(X_bank_full, y_bank_full, test_size=0.2, random_state=42)

clf_bank_full = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_bank_full.fit(X_bank_full_train, y_bank_full_train)

# Evaluate the model for bank-full.csv
y_bank_full_pred = clf_bank_full.predict(X_bank_full_test)
print("\nBank-Full.csv Dataset Results:")
print("Accuracy:", accuracy_score(y_bank_full_test, y_bank_full_pred))
print("Classification Report:\n", classification_report(y_bank_full_test, y_bank_full_pred))

# Visualize the decision tree for bank-full.csv and save the figure
plt.figure(figsize=(20, 10))
plot_tree(clf_bank_full, feature_names=X_bank_full.columns, class_names=['no', 'yes'], filled=True)
plt.title("Decision Tree - bank-full.csv")
plt.savefig("decision_tree_bank_full.png")
plt.close()

# Export decision tree rules for both datasets
rules_bank = export_text(clf_bank, feature_names=list(X_bank.columns))
print("\nDecision Tree Rules for bank.csv:\n", rules_bank)

rules_bank_full = export_text(clf_bank_full, feature_names=list(X_bank_full.columns))
print("\nDecision Tree Rules for bank-full.csv:\n", rules_bank_full)
