import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Charger le dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigree","Age","Outcome"
]

data = pd.read_csv(url, names=columns)

print("\nAperçu des données :")
print(data.head())

# 2. Visualisation
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 3. Séparer variables et labels
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Normalisation
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Entraînement du modèle
model = LogisticRegression()

model.fit(X_train, y_train)

# 7. Prédictions
predictions = model.predict(X_test)

# 8. Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy :", accuracy)

print("\nClassification Report :")
print(classification_report(y_test, predictions))

# 9. Matrice de confusion
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# 10. Test manuel
print("\nTest avec un patient fictif")

patient = [[2,120,70,20,79,25.0,0.5,33]]

patient = scaler.transform(patient)

prediction = model.predict(patient)

if prediction[0] == 1:
    print("Prediction : risque de diabète")
else:
    print("Prediction : pas de diabète")