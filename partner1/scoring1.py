from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Paso 1: Cargar los datos
# Reemplaza cargar_datos() con la función adecuada para cargar tus datos
X, y = cargar_datos()

# Paso 2: Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Paso 3: Definir el pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalización de características
    # Modelo de clasificación, en este caso Regresión Logística
    ('classifier', LogisticRegression())
])

# Paso 4: Ajustar el pipeline al conjunto de entrenamiento
pipeline.fit(X_train, y_train)

# Paso 5: Evaluar el rendimiento del modelo en el conjunto de prueba
accuracy = pipeline.score(X_test, y_test)
print('Exactitud del modelo:', accuracy)
