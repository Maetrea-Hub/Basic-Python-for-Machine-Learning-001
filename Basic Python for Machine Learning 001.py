import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Dataset_Weather_7_Days (Label).csv", sep=";")
print("Preview data:")
print(df.head())
print("\n")

X = df[["temperature", "humidity"]]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Mencari nilai K terbaik : ")
best_k = None
best_acc = 0

for k in range(1, 16):
  knn_temp = KNeighborsClassifier(n_neighbors=k)
  knn_temp.fit(X_train_scaled, y_train)
  pred_temp = knn_temp.predict(X_test_scaled)
  acc_temp = accuracy_score(y_test, pred_temp)
  print("K = ", k, "| Akurasi : ", acc_temp)

  if acc_temp > best_acc:
    best_acc = acc_temp
    best_k = k

print("\nK terbaik dari pencarian : ", best_k)
print("Akurasi terbaik : ", best_acc)
print("\n")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("Akurasi model final : ", acc)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_test["temperature"], X_test["humidity"], c=pd.factorize(y_pred)[0], cmap="viridis")

plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Scatter Plot Prediksi KNN")
plt.colorbar(label="Predicted Class")
plt.show()

h = 0.1
x_min, x_max = X["temperature"].min() - 1, X["temperature"].max() + 1
y_min, y_max = X["humidity"].min() - 1, X["humidity"].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)

Z = knn.predict(grid_scaled)
Z = pd.factorize(Z)[0]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")

plt.scatter(X["temperature"], X["humidity"], c=pd.factorize(y)[0], edgecolors="k")

plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Decision Boundary - KNN")
plt.show()