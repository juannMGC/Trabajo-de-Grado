import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import warnings
warnings.filterwarnings('ignore')

# Configurar logging para seguimiento del proceso
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessingPipeline:
    """
    Pipeline completo para procesamiento de datos de calidad del aire y salud
    """

    def __init__(self, archivo_entrada="BD_impacto_salud.xlsx"):
        self.archivo_entrada = archivo_entrada
        self.df_original = None
        self.df_procesado = None
        self.df_final = None
        self.scaler_X = None
        self.scaler_y = None
        self.label_encoder = None
        
    def cargar_datos(self):
        """Carga y valida el archivo de entrada"""
        try:
            logging.info(f"Cargando archivo: {self.archivo_entrada}")
            self.df_original = pd.read_excel(self.archivo_entrada)
            logging.info(f"Archivo cargado exitosamente. Dimensiones: {self.df_original.shape}")
            logging.info(f"Columnas encontradas: {list(self.df_original.columns)}")
            return True
        except Exception as e:
            logging.error(f"Error al cargar el archivo: {e}")
            return False
    
    def formatear_fechas(self):
        """Convierte columnas de fecha al formato datetime"""
        logging.info("Formateando columnas de fecha...")
        columnas_fecha = [col for col in self.df_original.columns if "Fecha_" in col]
        logging.info(f"Columnas de fecha encontradas: {columnas_fecha}")
        
        for col in columnas_fecha:
            self.df_original[col] = pd.to_datetime(self.df_original[col], errors='coerce')
            valores_nulos = self.df_original[col].isnull().sum()
            if valores_nulos > 0:
                logging.warning(f"Columna {col}: {valores_nulos} valores no pudieron convertirse a fecha")
    
    def alinear_por_fechas(self):
        """Alinea todas las variables por fecha"""
        logging.info("Alineando datos por fecha...")
        
        # Definir grupos de variables con sus fechas
        grupos = {
            "Fecha_ICA": ["ICA"],
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
        
        # Identificar variables que no tienen fecha asociada
        todas_vars_con_fecha = [var for var_list in grupos.values() for var in var_list]
        otras_vars = [col for col in self.df_original.columns 
                     if not col.startswith("Fecha_") and col not in todas_vars_con_fecha]
        
        logging.info(f"Variables sin fecha asociada: {otras_vars}")
        
        # Crear DataFrames alineados por fecha
        df_alineados = []
        for fecha_col, var_list in grupos.items():
            if fecha_col not in self.df_original.columns:
                logging.warning(f"Columna de fecha {fecha_col} no encontrada")
                continue
                
            sub_df = pd.DataFrame()
            sub_df["Fecha"] = self.df_original[fecha_col]
            
            # Agregar variables del grupo
            for var in var_list:
                if var in self.df_original.columns:
                    sub_df[var] = self.df_original[var]
                else:
                    logging.warning(f"Variable {var} no encontrada")
            
            # Agregar otras variables (se repetirán para cada fecha)
            for col in otras_vars:
                if col in self.df_original.columns:
                    sub_df[col] = self.df_original[col]
            
            # Solo agregar si tiene datos válidos
            if not sub_df["Fecha"].isnull().all():
                df_alineados.append(sub_df)
        
        if not df_alineados:
            raise ValueError("No se pudieron crear DataFrames alineados")
        
        # Combinar y agrupar por fecha
        df_completo = pd.concat(df_alineados, axis=0, ignore_index=True)
        
        # Agrupar por fecha tomando el primer valor válido de cada variable
        self.df_procesado = df_completo.groupby("Fecha", as_index=False).first().sort_values("Fecha")
        
        logging.info(f"Datos alineados. Filas resultantes: {len(self.df_procesado)}")
        logging.info(f"Rango de fechas: {self.df_procesado['Fecha'].min()} a {self.df_procesado['Fecha'].max()}")
    
    def limpiar_datos(self):
        """Elimina columnas innecesarias y limpia los datos"""
        logging.info("Limpiando datos...")
        
        # Columnas a eliminar
        columnas_a_eliminar = [
            # Estaciones (información redundante)
            "Estacion_ICA", "Estacion_PM10", "Estacion_PM2_5", "Estacion_SO2", "Estacion_O3",
            "Estacion_Temperatura", "Estacion_Humedad", "Estacion_VelocidadViento",
            "Estacion_CasosRespiratorios", "Estacion_IngresosHospitalariosR",
            "Estacion_CasosCardiovasculares", "Estacion_IngresosHospitalariosC",
            # Diagnósticos (si no son necesarios)
            "DiagnosticosRespiratorios", "DiagnosticosCardiovasculares",
            # Casos que pueden ser redundantes con ingresos
            "CasosRespiratorios", "CasosCardiovasculares"
        ]
        
        # Eliminar columnas que existan
        columnas_existentes = [col for col in columnas_a_eliminar if col in self.df_procesado.columns]
        if columnas_existentes:
            self.df_procesado.drop(columns=columnas_existentes, inplace=True)
            logging.info(f"Columnas eliminadas: {columnas_existentes}")
        
        # Reemplazar espacios vacíos con NaN
        self.df_procesado.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Convertir columnas con comas a punto decimal
        for col in self.df_procesado.columns:
            if col != 'Fecha' and self.df_procesado[col].dtype == object:
                self.df_procesado[col] = self.df_procesado[col].astype(str).str.replace(',', '.')
                self.df_procesado[col] = pd.to_numeric(self.df_procesado[col], errors='coerce')
    
    def crear_puntaje_impacto(self):
        """Crea el puntaje de impacto en salud"""
        logging.info("Creando puntaje de impacto en salud...")
        
        # Definir pesos para cada variable (ajustables según criterio médico/científico)
        pesos = {
            'PM2_5': 0.25,        # Partículas finas - mayor impacto
            'PM10': 0.20,         # Partículas gruesas
            'SO2': 0.10,          # Dióxido de azufre
            'O3': 0.05,           # Ozono
            'ICA': 0.05,          # Índice de calidad del aire
            'ConsultaHospitalariosR': 0.10,  # Consultas respiratorias
            'ConsultaHospitalariosC': 0.05,  # Consultas cardiovasculares
            'IngresosHospitalariosR': 0.10,  # Ingresos respiratorios
            'IngresosHospitalariosC': 0.10   # Ingresos cardiovasculares
        }
        
        # Calcular puntaje solo para variables que existen
        puntaje_componentes = []
        pesos_usados = {}
        
        for var, peso in pesos.items():
            if var in self.df_procesado.columns:
                componente = peso * self.df_procesado[var].fillna(0)
                puntaje_componentes.append(componente)
                pesos_usados[var] = peso
                logging.info(f"Variable {var} incluida con peso {peso}")
            else:
                logging.warning(f"Variable {var} no encontrada para el cálculo del puntaje")
        
        if puntaje_componentes:
            self.df_procesado['PuntajeImpactoSalud'] = sum(puntaje_componentes)
            logging.info(f"Puntaje creado usando variables: {list(pesos_usados.keys())}")
            logging.info(f"Suma de pesos utilizados: {sum(pesos_usados.values()):.2f}")
            
            # Mostrar estadísticas del puntaje
            stats = self.df_procesado['PuntajeImpactoSalud'].describe()
            logging.info(f"Estadísticas del puntaje:\n{stats}")
        else:
            logging.error("No se pudieron encontrar variables para crear el puntaje")
    
    def crear_clase_impacto(self, metodo='percentiles'):
        """Crea la clasificación de impacto en salud"""
        logging.info(f"Creando clasificación de impacto usando método: {metodo}")
        
        if 'PuntajeImpactoSalud' not in self.df_procesado.columns:
            logging.error("Primero debe crear el puntaje de impacto")
            return
        
        puntajes = self.df_procesado['PuntajeImpactoSalud']
        
        if metodo == 'percentiles':
            # Clasificación basada en percentiles
            percentiles = puntajes.quantile([0.2, 0.4, 0.6, 0.8]).values
            
            def clasificar(valor):
                if pd.isna(valor):
                    return "Desconocido"
                elif valor <= percentiles[0]:
                    return "Muy_Bajo"
                elif valor <= percentiles[1]:
                    return "Bajo"
                elif valor <= percentiles[2]:
                    return "Medio"
                elif valor <= percentiles[3]:
                    return "Alto"
                else:
                    return "Muy_Alto"
                    
        elif metodo == 'cuartiles':
            # Clasificación por cuartiles
            Q1, Q2, Q3 = puntajes.quantile([0.25, 0.5, 0.75])
            
            def clasificar(valor):
                if pd.isna(valor):
                    return "Desconocido"
                elif valor <= Q1:
                    return "Bajo"
                elif valor <= Q2:
                    return "Medio_Bajo"
                elif valor <= Q3:
                    return "Medio_Alto"
                else:
                    return "Alto"
                    
        elif metodo == 'manual':
            # Clasificación manual (ajustar rangos según necesidad)
            def clasificar(valor):
                if pd.isna(valor):
                    return "Desconocido"
                elif valor <= 10:
                    return "Impacto_Minimo"
                elif valor <= 25:
                    return "Impacto_Leve"
                elif valor <= 50:
                    return "Impacto_Moderado"
                elif valor <= 75:
                    return "Impacto_Severo"
                else:
                    return "Impacto_Critico"
        
        self.df_procesado['ClaseImpactoSalud'] = puntajes.apply(clasificar)
        
        # Mostrar distribución de clases
        distribucion = self.df_procesado['ClaseImpactoSalud'].value_counts()
        logging.info(f"Distribución de clases:\n{distribucion}")
    
    def preparar_para_ml(self, test_size=0.2, escalar_datos=True):
        """Prepara los datos para machine learning"""
        logging.info("Preparando datos para machine learning...")
        
        # Crear una copia para ML
        self.df_final = self.df_procesado.copy()
        
        # Eliminar fecha si existe
        if 'Fecha' in self.df_final.columns:
            self.df_final.drop(columns=['Fecha'], inplace=True)
        
        # Verificar que existe la variable objetivo
        if 'ClaseImpactoSalud' not in self.df_final.columns:
            logging.error("No existe la variable objetivo 'ClaseImpactoSalud'")
            return False
        
        # Separar características y objetivo
        X = self.df_final.drop(columns=['ClaseImpactoSalud'])
        y = self.df_final['ClaseImpactoSalud']
        
        logging.info(f"Variables predictoras: {list(X.columns)}")
        logging.info(f"Dimensiones X: {X.shape}")
        
        # Imputar valores faltantes
        # Para X (variables numéricas)
        X_imputado = X.fillna(X.mean())
        
        # Para y (variable objetivo)
        if y.dtype.kind in 'fc':  # Si es numérica
            y_imputado = y.fillna(y.mean())
        else:  # Si es categórica
            y_imputado = y.fillna(y.mode()[0] if len(y.mode()) > 0 else "Desconocido")
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputado, y_imputado, test_size=test_size, random_state=42, stratify=y_imputado
        )
        
        logging.info(f"Datos de entrenamiento: {X_train.shape[0]}")
        logging.info(f"Datos de prueba: {X_test.shape[0]}")
        
        if escalar_datos:
            # Estandarizar características
            self.scaler_X = StandardScaler()
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
            
            # Manejar variable objetivo
            if y_train.dtype.kind in 'fc':  # Si es numérica
                self.scaler_y = StandardScaler()
                y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
            else:  # Si es categórica
                self.label_encoder = LabelEncoder()
                y_train_scaled = self.label_encoder.fit_transform(y_train.astype(str))
                y_test_scaled = self.label_encoder.transform(y_test.astype(str))
            
            # Crear DataFrame final para guardar
            df_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
            df_train_final['ClaseImpactoSalud'] = y_train_scaled
        else:
            df_train_final = X_train.copy()
            df_train_final['ClaseImpactoSalud'] = y_train.values
        
        self.df_final = df_train_final
        
        # Guardar información de procesamiento
        self.info_procesamiento = {
            'test_size': test_size,
            'escalado': escalar_datos,
            'variables_predictoras': list(X.columns),
            'tamaño_entrenamiento': len(X_train),
            'tamaño_prueba': len(X_test),
            'tipo_objetivo': 'numerico' if y.dtype.kind in 'fc' else 'categorico',
            'clases_objetivo': y_imputado.unique().tolist() if y.dtype.kind not in 'fc' else None
        }
        
        return True
    
    def guardar_archivos(self, 
                        archivo_objetivo="BD_impacto_salud_objetivo.xlsx",
                        archivo_ml_excel="BD_impacto_salud_ml.xlsx",
                        archivo_ml_csv="BD_impacto_salud_ml.csv"):
        """Guarda los archivos procesados"""
        logging.info("Guardando archivos...")
        
        try:
            # Guardar archivo intermedio (con fecha)
            if self.df_procesado is not None:
                self.df_procesado.to_excel(archivo_objetivo, index=False)
                logging.info(f"Archivo intermedio guardado: {archivo_objetivo}")
            
            # Guardar archivo para ML
            if self.df_final is not None:
                self.df_final.to_excel(archivo_ml_excel, index=False)
                self.df_final.to_csv(archivo_ml_csv, index=False)
                logging.info(f"Archivos ML guardados: {archivo_ml_excel}, {archivo_ml_csv}")
                
                return True
        except Exception as e:
            logging.error(f"Error al guardar archivos: {e}")
            return False
    
    def ejecutar_pipeline_completo(self, metodo_clasificacion='percentiles'):
        """Ejecuta todo el pipeline de procesamiento"""
        logging.info("=== INICIANDO PIPELINE DE PROCESAMIENTO ===")
        
        try:
            # 1. Cargar datos
            if not self.cargar_datos():
                return False
            
            # 2. Formatear fechas
            self.formatear_fechas()
            
            # 3. Alinear por fechas
            self.alinear_por_fechas()
            
            # 4. Limpiar datos
            self.limpiar_datos()
            
            # 5. Crear puntaje de impacto
            self.crear_puntaje_impacto()
            
            # 6. Crear clasificación de impacto
            self.crear_clase_impacto(metodo=metodo_clasificacion)
            
            # 7. Preparar para ML
            if not self.preparar_para_ml():
                return False
            
            # 8. Guardar archivos
            if not self.guardar_archivos():
                return False
            
            logging.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
            self.mostrar_resumen()
            return True
            
        except Exception as e:
            logging.error(f"Error en el pipeline: {e}")
            return False
    
    def mostrar_resumen(self):
        """Muestra un resumen del procesamiento"""
        print("\n" + "="*50)
        print("RESUMEN DEL PROCESAMIENTO")
        print("="*50)
        
        if hasattr(self, 'info_procesamiento'):
            info = self.info_procesamiento
            print(f"Variables predictoras: {len(info['variables_predictoras'])}")
            print(f"Tamaño entrenamiento: {info['tamaño_entrenamiento']}")
            print(f"Tamaño prueba: {info['tamaño_prueba']}")
            print(f"Tipo de objetivo: {info['tipo_objetivo']}")
            if info['clases_objetivo']:
                print(f"Clases objetivo: {info['clases_objetivo']}")
        
        if self.df_procesado is not None:
            print(f"Registros procesados: {len(self.df_procesado)}")
            if 'ClaseImpactoSalud' in self.df_procesado.columns:
                print("Distribución de clases:")
                print(self.df_procesado['ClaseImpactoSalud'].value_counts())
        
        print("="*50)

# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Crear instancia del pipeline
    pipeline = DataProcessingPipeline("BD_impacto_salud.xlsx")
    
    # Ejecutar pipeline completo
    exito = pipeline.ejecutar_pipeline_completo(metodo_clasificacion='percentiles')
    
    if exito:
        print("\n✅ Procesamiento completado exitosamente!")
        print("Archivos generados:")
        print("- BD_impacto_salud_objetivo.xlsx (datos intermedios)")
        print("- BD_impacto_salud_ml.xlsx (datos para ML)")
        print("- BD_impacto_salud_ml.csv (datos para ML)")
    else:
        print("\n❌ Error en el procesamiento")