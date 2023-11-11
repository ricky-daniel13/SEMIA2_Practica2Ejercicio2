
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as score
import pandas as pd
import numpy as np

# Funcion para calcular la especificidad
def specificity_score(y_test, preds):
    conf_matrix = score.confusion_matrix(y_test, preds)
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    return true_negatives / (true_negatives + false_positives)

print("############################### Datos: Calidad de Vinos #############################################")
dir_csv = '.\\archive.ics.uci.edu_ml_machine-learning-databases_wine-quality_winequality-white.csv'

data = pd.read_csv(dir_csv)
print(data.head())
#print(data.info())

# Dividir los datos en X y Y
X = np.array(data.iloc[:, :-1])  # Tomamos la ultima columna como nuestro target
y = np.array(data.iloc[:, -1])

#print(y)


#print(y)
# Dividir en sets de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Estandarizar los parametros para knn y svm
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar calificadores
logistic_regression = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=10)
svm = SVC(kernel='linear')
naive_bayes = GaussianNB()

logistic_regression.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Predicciones
logistic_regression_pred = logistic_regression.predict(X_test)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)


print("Comprobacion: ")
for i in range(min(len(X_test),10)):
    print(f"Valor real: {y_test[i]}")
    print(f"\tPrediccion Regrecion Logistica: {logistic_regression_pred[i]}")
    print(f"\tPrediccion K Neighbours: {knn_pred[i]}")
    print(f"\tPrediccion SVM: {svm_pred[i]}")
    print(f"\tPrediccion naive bayes: {naive_bayes_pred[i]}")







print("############################### Datos: Seguros Vehicular #############################################")

data = pd.read_csv('.\\automakers.csv', dtype=float)
print(data.head())


print("No se puede, no son datos clasificables")

print("############################### Datos: Diabetes #############################################")


data = pd.read_csv('.\\raw.githubusercontent.com_jbrownlee_Datasets_master_pima-indians-diabetes.csv', dtype=float, header=None)
print(data.head())
#print(data.info())

# Dividir los datos en X y Y
X = np.array(data.iloc[:, :-1])  # Tomamos la ultima columna como nuestro target
y = np.array(data.iloc[:, -1])

# Dividir en sets de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Estandarizar los parametros para knn y svm
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar calificadores
logistic_regression = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=10)
svm = SVC(kernel='linear')
naive_bayes = GaussianNB()

logistic_regression.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Predicciones
logistic_regression_pred = logistic_regression.predict(X_test)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)

print("Comprobacion: ")
for i in range(min(len(X_test),10)):
    print(f"Valor real: {y_test[i]}")
    print(f"\tPrediccion Regrecion Logistica: {logistic_regression_pred[i]}")
    print(f"\tPrediccion K Neighbours: {knn_pred[i]}")
    print(f"\tPrediccion SVM: {svm_pred[i]}")
    print(f"\tPrediccion naive bayes: {naive_bayes_pred[i]}")

