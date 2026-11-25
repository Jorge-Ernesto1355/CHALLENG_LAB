"""Entrenamiento del modelo de mantenimiento predictivo."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
OBJETIVO = "estado"

# 1. Lectura y limpieza
df = pd.read_csv("sensores_planta.csv").dropna()
print(f"📄 Registros cargados: {len(df)}")
print(df[OBJETIVO].value_counts().sort_index().to_string())

X = df[FEATURES]
y = df[OBJETIVO]

# 2. División 80/20 (estratificada para que haya los 3 estados en prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Entrenamiento
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

# 4. Evaluación
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")
assert precision >= 0.80, "La precisión es menor a 0.80"

# 5. Exportación
joblib.dump(modelo, "modelo_mantenimiento.pkl")
print("💾 Modelo exportado: modelo_mantenimiento.pkl")
