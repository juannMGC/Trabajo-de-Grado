import pandas as pd

# Cargar la base-archivo
df = pd.read_excel("BD_impacto_salud_.xlsx")

# Asegúrarme de que las fechas estén bien formateadas
columnas_fecha = [col for col in df.columns if "Fecha_" in col]
for col in columnas_fecha:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Lista de grupos (columna de fecha, columnas asociadas al grupo)
grupos = {
    "Fecha_ICA_SO2": ["ICA_SO2"],
    "Fecha_PM10": ["PM10"],
    "Fecha_PM2_5": ["PM2_5"],
    "Fecha_SO2": ["SO2"],
    "Fecha_O3": ["O3"],
    "Fecha_Temperatura": ["Temperatura"],
    "Fecha_Humedad": ["Humedad"],
    "Fecha_VelocidadViento": ["VelocidadViento"],
    "Fecha_CasosRespiratorios": ["CasosRespiratorios"],
    "Fecha_IngresosHospitalariosR": ["IngresosHospitalariosR"],
    "Fecha_CasosCardiovasculares": ["CasosCardiovasculares"],
    "Fecha_IngresosHospitalariosC": ["IngresosHospitalariosC"],
}

# Variables fijas que no tienen fecha, queremos mantenerlas segun su fecha
otras_vars = [col for col in df.columns if not "Fecha_" in col and all(col not in g for g in grupos.values())]

# Lista para guardar los nuevos dataframes por grupo
df_alineados = []

for fecha_col, var_list in grupos.items():
    sub_df = pd.DataFrame()
    sub_df["Fecha"] = df[fecha_col]
    
    # Agregamos variables del grupo
    for var in var_list:
        sub_df[var] = df[var]
        
    # Agregamos las otras columnas fijas (estación, ciudad, diagnósticos, etc.)
    for col in otras_vars:
        sub_df[col] = df[col]
    
    df_alineados.append(sub_df)

# Concatenamos todos los sub-dataframes
df_completo = pd.concat(df_alineados, axis=0)

# Agrupamos por fecha para juntar todos los datos por individuo (fecha)
df_final = df_completo.groupby("Fecha", as_index=False).first().sort_values("Fecha")

# Guardar el DataFrame
df_final.to_excel("BD_impacto_salud_objetivo.xlsx", index=False)