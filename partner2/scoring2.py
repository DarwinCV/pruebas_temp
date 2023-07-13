from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Paso 1: Cargar los datos
# Reemplaza cargar_datos() con la función adecuada para cargar tus datos
X, y = load_datos_partners()

for i in range(1,3):
    print(i)

# Paso 2: Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Paso 3: Definir el pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalización de características
    ('regressor', LinearRegression())  # Modelo de regresión lineal
])

# Paso 4: Ajustar el pipeline al conjunto de entrenamiento
pipeline.fit(X_train, y_train)

# Paso 5: Evaluar el rendimiento del modelo en el conjunto de prueba
r2_score = pipeline.score(X_test, y_test)
print('Coeficiente de correlacion (R2):', r2_score)
print('fin de modelamiento')


