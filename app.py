import pandas as pd # Manipulación de datos
import numpy as np # Operaciones aritmeticas
import matplotlib.pyplot as plt # Visualización
import pickle # Permite guardar y cargar modelos ML


## Algoritmos---------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
##--------------------------------------------

from sklearn.model_selection import RandomizedSearchCV # Ajuste de hiperparámetros
from sklearn.model_selection import train_test_split # División de conjuntos de datos
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Métricas
from sklearn.preprocessing import MinMaxScaler # Normalización de datos
from sklearn.inspection import permutation_importance # Importancia por permutación


# Evitar mensajes de warnings
import warnings
warnings.filterwarnings('ignore')


# Cargar datos
data_inicial = pd.read_excel('datamxfull2024.xlsx')

data = data_inicial.groupby(['FCIDTIENDA','FDIDDIA'])[['FNVENTA_TOTAL','FNVENTA_REAL_AA']].sum().reset_index().copy()


# Obtener día de la semana y semana del mes, en base al campo fecha
data['DIA_SEMANA'] = data['FDIDDIA'].dt.dayofweek
data['SEMANA_MES'] = data['FDIDDIA'].apply(lambda x: (x.day - 1) // 7 + 1)


# Ordenar los registros según tienda y fecha, para generar los lags correctamente
data = data.sort_values(by=['FCIDTIENDA', 'FDIDDIA'])

# Crear lags de 7,14 y 21 días
for lag in [7,14,21]:
    data[f'FNVENTA_LAG{lag}_DIA'] = (
        data.groupby(['FCIDTIENDA'])['FNVENTA_TOTAL']
        .shift(lag)
    )

data = data.dropna()

# Se convierte el campo FCIDTIENDA a texto
data['FCIDTIENDA'] = data['FCIDTIENDA'].astype(str)
# Se asigna , como indice, la fecha
data = data.set_index('FDIDDIA')

# Separar la data en predictores, asignados en la variable X, y en targets, asignados en la variable Y.
X = data.drop(columns=['FNVENTA_TOTAL'])
y = data['FNVENTA_TOTAL']

# Generar el conjunto de entrenamiento y test. Para casos de series de tiempo, se debe asignar shuffle = False, ya que se necesita un orden temporal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

# Se señala los campos a normalizar para obtener valores entre 0 y 1 y así evitar que las magnitudes de las variables afecten al modelo.
columns_to_scale = ['FNVENTA_REAL_AA','DIA_SEMANA','SEMANA_MES','FNVENTA_LAG7_DIA', 'FNVENTA_LAG14_DIA', 'FNVENTA_LAG21_DIA']

# Se "entrena" la normalización utilizando tan solo el conjunto de entrenamiento para luego aplicar transformación en el conjunto de entrenamiento y test
scaler = MinMaxScaler()

X_train_scaled = X_train.copy()
X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])

X_test_scaled = X_test.copy()
X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])


# Se selecciona el modelo "GradientBoostingRegressor" debido que "RandomForestRegressor", realizando un análisis previo, tendría ligero sobreajuste.
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled,y_train)

# Se guarda el modelo entrenado anteriomente en un archivo .pkl
with open('model.pkl', 'wb') as archivo:
    pickle.dump(model, archivo)

# Se carga el modelo guardado en el anterior linea de código
with open('model.pkl', 'rb') as archivo:
    model_cargado = pickle.load(archivo)




# Se obtiene las predicciones utilizando el conjunto de datos de entrenamiento para obtener las métricas de error
y_pred_train = model_cargado.predict(X_train_scaled)

mae = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred_train)

# Se obtiene las predicciones utilizando el conjunto de datos de test para obtener las métricas de error
y_pred_test = model_cargado.predict(X_test_scaled)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# Ejemplo de función de ingeniería de características
def feature_engineering_func(forecast_df):
    forecast_df['DIA_SEMANA'] = forecast_df['FDIDDIA'].dt.dayofweek
    forecast_df['SEMANA_MES'] = forecast_df['FDIDDIA'].apply(lambda x: (x.day - 1) // 7 + 1)
    forecast_df['FNVENTA_LAG7_DIA'] = forecast_df.groupby('FCIDTIENDA')['FNVENTA_TOTAL'].shift(7)
    forecast_df['FNVENTA_LAG14_DIA'] = forecast_df.groupby('FCIDTIENDA')['FNVENTA_TOTAL'].shift(14)
    forecast_df['FNVENTA_LAG21_DIA'] = forecast_df.groupby('FCIDTIENDA')['FNVENTA_TOTAL'].shift(21)
    return forecast_df


data_predecir = pd.read_excel('datOSmx2024_prediccion2.xlsx')

data_predecir_feature = feature_engineering_func(data_predecir)


X_pred = data_predecir_feature[data_predecir_feature['FNVENTA_TOTAL'].isnull()].drop(columns=['FNVENTA_TOTAL']).dropna().set_index('FDIDDIA').copy()
# revisar los casos límites

columns_to_scale = ['FNVENTA_REAL_AA','DIA_SEMANA','SEMANA_MES','FNVENTA_LAG7_DIA', 'FNVENTA_LAG14_DIA', 'FNVENTA_LAG21_DIA']
X_pred_scaled = X_pred.copy()
X_pred_scaled[columns_to_scale] = scaler.transform(X_pred[columns_to_scale])

y_pred = model_cargado.predict(X_pred_scaled)

X_pred['PREDICCION'] = y_pred

X_pred.reset_index().to_excel('resultados.xlsx')

response_model = pd.read_excel('resultados.xlsx')

print(response_model.head())