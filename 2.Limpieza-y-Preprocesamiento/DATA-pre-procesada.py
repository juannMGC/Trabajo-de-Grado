import pandas as pd
import numpy as np

def cargar_datos(ruta):
    # Cargar datos y convertir diferentes representaciones de faltantes a NaN
    df = pd.read_excel(ruta)  # Cambia a pd.read_csv() si es necesario
    
    # Lista de representaciones comunes de valores faltantes
    missing_values = ["", " ", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "?", "-", "--"]
    
    # Reemplazar todos estos valores por NaN
    df = df.replace(missing_values, np.nan)
    
    return df
    
# 1er momento de preprocesamiento eliminacion de columnas innecesarias
def eliminar_columnas(df):
    columnas_a_eliminar = [
        'Estacion_ICA_SO2',
        'Estacion_PM10',
        'Estacion_PM2_5',
        'Estacion_SO2',
        'Estacion_O3',
        'Estacion_Temperatura',
        'Estacion_Humedad',
        'Estacion_VelocidadViento',
        'Estacion_CasosRespiratorios',
        'Estacion_IngresosHospitalariosR',
        'Estacion_CasosCardiovasculares',
        'Estacion_IngresosHospitalariosC',
        'DiagnosticosRespiratorios',
        'DiagnosticosCardiovasculares',
        'CasosRespiratorios',
        'CasosCardiovasculares'
    ]
    
    # Eliminar solo las columnas que existen en el DataFrame
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    df = df.drop(columns=columnas_existentes)
    
    return df

# 2do momento de preprocesamiento representaciones de faltantes ("", "NA", "N/A", etc.)
def reportar_valores_faltantes(df):
    print("\nReporte de Valores Faltantes (NA):")
    print(df.isna().sum())
    print("\nPorcentaje de valores faltantes por columna:")
    print((df.isna().mean()*100).round(2).astype(str) + "%")

def guardar_datos(df, ruta_salida):
    # Asegurarse que los NA se guarden correctamente
    df.to_csv(ruta_salida, index=False, na_rep='NA')

# 3er momento de preprocesamiento


if __name__ == "__main__":
    # Cargar datos originales y estandarizar valores faltantes
    ruta_entrada = "BD_impacto_salud_objetivo.xlsx"
    datos = cargar_datos(ruta_entrada)
    
    # Reportar valores faltantes antes de procesar
    print("\nValores faltantes en datos originales:")
    reportar_valores_faltantes(datos)
    
    # Eliminar columnas no necesarias
    datos_limpios = eliminar_columnas(datos)
    
    # Reportar valores faltantes después de eliminar columnas
    print("\nValores faltantes después de eliminar columnas:")
    reportar_valores_faltantes(datos_limpios)
    
    # Guardar resultado
    ruta_salida = "BD_impacto_salud_columnas_limpias.csv"
    guardar_datos(datos_limpios, ruta_salida)
    
    print(f"- Columnas restantes: {list(datos_limpios.columns)}")