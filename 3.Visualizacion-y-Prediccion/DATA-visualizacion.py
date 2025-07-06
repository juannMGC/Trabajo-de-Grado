import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import requests
from streamlit_lottie import st_lottie
import joblib

st.set_page_config(page_title='Manizales ML - Calidad del Aire', layout='wide')

# Cargar animación Lottie
def cargar_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_salud = cargar_lottie_url("https://lottie.host/3d5ee517-de54-4bf5-a42f-8c87c929dc5c/Bk2pUIGElP.json")

st.info('¡Esta aplicación usa Machine Learning para predecir el impacto en la salud según la calidad del aire!')

# Cargar dataset
df = pd.read_csv("datos_impacto_salud_calidad_aire.csv")

# Mostrar datos
with st.expander('📊 Ver datos'):
    st.dataframe(df)
    st.write("Variables predictoras (X)")
    X = df.drop(['PuntajeImpactoSalud', 'ClaseImpactoSalud', 'ID_Registro'], axis=1)
    st.write(X)
    st.write("Variable objetivo (y): ClaseImpactoSalud")
    y = df['ClaseImpactoSalud']

# Escalar datos
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

with st.expander('📈 Visualización de datos'):
    st.scatter_chart(df, x='ICA', y='PuntajeImpactoSalud', color='ClaseImpactoSalud')

# Barra lateral: Configuración
with st.sidebar:
    st.title("🧪 App Salud Ambiental")
    st_lottie(lottie_salud, speed=1, height=150, key="salud")
    st.header('⚙️ Configuración del Modelo')
    algoritmo = st.selectbox("Selecciona el algoritmo:",
                             ["Random Forest", "KNN", "Regresión Logística", "Árbol de Decisión", "Red Neuronal"])
    if algoritmo == "KNN":
        vecinos = st.slider("Número de vecinos:", 1, 20, 5)
        modelo = KNeighborsClassifier(n_neighbors=vecinos)
    elif algoritmo == "Regresión Logística":
        modelo = LogisticRegression()
    elif algoritmo == "Árbol de Decisión":
        profundidad = st.slider("Profundidad máxima:", 1, 20, 5)
        modelo = DecisionTreeClassifier(max_depth=profundidad)
    elif algoritmo == "Red Neuronal":
        capa_oculta = st.slider("Tamaño capa oculta:", 1, 100, 50)
        modelo = MLPClassifier(hidden_layer_sizes=(capa_oculta,), max_iter=500)
    else:
        estimadores = st.slider("Estimadores:", 1, 100, 50)
        profundidad = st.slider("Profundidad máxima:", 1, 20, 5)
        modelo = RandomForestClassifier(n_estimators=estimadores, max_depth=profundidad)

# Barra lateral: Entrada de datos
with st.sidebar:
    st.header('✏️ Ingresar Datos')
    with st.expander("Ingresar características"):
        ICA = st.slider("ICA", 0, 500, 100)
        PM10 = st.slider("PM10", 0.0, 500.0, 80.0)
        PM2_5 = st.slider("PM2.5", 0.0, 500.0, 60.0)
        NO2 = st.slider("NO2", 0.0, 200.0, 50.0)
        SO2 = st.slider("SO2", 0.0, 100.0, 10.0)
        O3 = st.slider("O3", 0.0, 200.0, 70.0)
        Temperatura = st.slider("Temperatura (°C)", -10.0, 50.0, 25.0)
        Humedad = st.slider("Humedad (%)", 0.0, 100.0, 60.0)
        VelocidadViento = st.slider("Velocidad del viento (m/s)", 0.0, 30.0, 3.0)
        CasosRespiratorios = st.slider("Casos Respiratorios", 0, 1000, 100)
        CasosCardiovasculares = st.slider("Casos Cardiovasculares", 0, 1000, 50)
        IngresosHospitalarios = st.slider("Admisiones Hospitalarias", 0, 1000, 80)

    datos_entrada = pd.DataFrame({
        'ICA': [ICA],
        'PM10': [PM10],
        'PM2_5': [PM2_5],
        'NO2': [NO2],
        'SO2': [SO2],
        'O3': [O3],
        'Temperatura': [Temperatura],
        'Humedad': [Humedad],
        'VelocidadViento': [VelocidadViento],
        'CasosRespiratorios': [CasosRespiratorios],
        'CasosCardiovasculares': [CasosCardiovasculares],
        'IngresosHospitalarios': [IngresosHospitalarios]
    })

# Mostrar entrada
with st.expander("🧮 Datos de Entrada"):
    st.write("**Entrada del usuario**")
    st.dataframe(datos_entrada)

# Escalar entrada
datos_entrada_escalado = escalador.transform(datos_entrada)

# Entrenamiento y predicción
modelo.fit(X_escalado, y)
prediccion = modelo.predict(datos_entrada_escalado)
proba_prediccion = modelo.predict_proba(datos_entrada_escalado)

# Mostrar predicción
st.subheader("🔮 Predicción del Impacto en la Salud")
etiquetas = ['Muy Alto', 'Alto', 'Moderado', 'Bajo', 'Muy Bajo']
proba_df = pd.DataFrame(proba_prediccion, columns=etiquetas)

st.dataframe(proba_df, column_config={
    etiqueta: st.column_config.ProgressColumn(etiqueta, format='%f', width='medium', min_value=0, max_value=1)
    for etiqueta in etiquetas
}, hide_index=True)

st.success(f"⚠️ Impacto estimado en la salud: **{etiquetas[int(prediccion[0])]}**")

# Guardar modelo + escalador
with st.sidebar:
    st.header('💾 Guardar Modelo')
    with st.expander("Guardar modelo"):
        if st.button("Guardar Modelo"):
            nombre_archivo = f"{algoritmo.replace(' ', '_').lower()}_modelo_salud.pkl"
            joblib.dump((modelo, escalador), nombre_archivo)
            with open(nombre_archivo, "rb") as archivo:
                st.download_button("📥 Descargar Modelo", data=archivo, file_name=nombre_archivo)

# Cargar modelo
with st.sidebar:
    with st.expander('📤 Subir modelo para Verificación'):
        archivo_subido = st.file_uploader("Subir modelo", type=["pkl"])
        if archivo_subido is not None:
            modelo_cargado, escalador_cargado = joblib.load(archivo_subido)
            st.success('✅ Modelo cargado correctamente')
            if st.button("Verificar Score"):
                score = modelo_cargado.score(X_escalado, y)
                st.success(f'📈 Score del modelo: {score:.4f}')


# # Función para establecer la imagen de fondo con CSS
# def set_background_image(url,opacity=0.5):
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: url('{url}') no-repeat center center fixed;
#             background-size: cover;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# #Llamar a la función para establecer el fondo de pantalla
# set_background_image("https://i.morioh.com/52c215bc5f.png")
