# Instrucciones manejo

Inicialmente, para la implementacion de esta prueba se hace uso de los datos de ([Amazon reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data)). POR FAVOR DESCARGAR LOS DATOS DE TAL FORMA QUE QUEDEN EN LA SIGUIENTE DIRECCION **"./dataset/amazon_review_polarity_csv.tgz"**.  Esta base de datos consta de aproximadamente 1'800.000 muestras para entrenamiento y mas de 200.000 muestras para testeo. 

Para mayor profundidad y detalles en los flujos de los procesos y diagramas

**Como primer paso** para hacer uso de este desarrollo, el primer paso es **instalar los paquetes necesarios**, esto ejecutando el comando:

```python
pip install -r requirements.txt
```

**Como segundo paso** (y si se desean utilizar los datos propuestos), es necesario ejecutar el script:

```python
python3 dataset_extraction.py
```

Este script se encarga de crear las carpetas necesarias, descomprimir y organizar los datos originales. Para fines del ejemplo, la base de datos es distribuida así:
- Del set de datos para entrenamiento, se toman 1'500.000 filas y se almacenan en **"/raw/train/<nombre_del_archivo>.csv"**. Este conjunto simulará un conjunto para el entrenamiento y prueba del modelo (se divide en sets de entrenamiento y prueba)
- Por otro lado, al el set de prueba se le retiran las columnas dejando únicamente la columna de texto y es almacenada en **"/raw/inference/<nombre_del_archivo>.csv"**. El objetivo de este es simular un bache de textos a los cuales aplicar la predicción del modelo.

**Como tercer paso**, una vez se tienen los datos para entrenamiento e inferencia. El usuario puede visualizar los flujos de los procesos en las notebooks respectivas:

- **./train_pipelines.ipynb:** Contiene la pipeline y paso a paso para el proceso de entrenamiento del modelo. Este sigue el siguiente proceso:
  
  1. **Validacion:**
    - Gira en torno al objeto **FileVerification**
    - Para inicializarlo, se requiere la ruta del archivo de los datos crudos de entrenamiento (source_path), nombre de columnas para entrenamiento (required_columns = ["score","text"]) y los formatos validos de datos (valid_formats). Por ahora solo hay implementacion con formatos .csv
    - Este tiene los siguientes metodos:
      - **format_verification:** Ejecuta la verificacion del formato del archivo de datos
      - **readability_verification:** Verifica que el archivo sea legible 
      - **column_verification:** Verifica que la base de datos contenga las columnas requeridas
      - **column_filtration:** Retorna un DataFrame de pandas filtrado, que contiene unicamente las columnas requeridas
      - **verify:** Ejecuta completamente el proceso de verificacion. Retorna la direccion de los datos pre-procesados para el ETL.

  2. **Procesamiento y ETL:**
    - Gira en torno al bojeto FileProcessing
    - Para inicializarlo, se requiere la ruta del archivo de los datos preprocesados (source_path) y las columnas requeridas para el entrenamiento (required_columns = ["score","embedding"])
    - Este tiene los siguientes metodos:
      - **load_data:** Carga los datos de entrenamiento en una df
      - **clean_data:** Ejecuta el proceso propuesto de limpieza de datos
      - **transform_data:** Ejecuta el proceso propuesto de transformacion de datos. En este se incluye la generacion de los embeddings
      - **process:** Ejecuta completamente el procesamiento de los datos. Retorna la direccion de los datos post-procesados para el entrenamiento

  3. **Entrenamiento del modelo:**
    - Gira en torno al objeto **TrainingProcess**
    - Para inicializarlo, se requiere la ruta del archivo de los datos postprocesados (source_path) y la version del modelo que se quiere entrenar (version)
    - Este tiene los siguientes metodos:
      - **load_data:** Carga los datos para el entrenamiento
      - **process_features:** Organiza los vectores generados por el embedding, tambien genera el LabelEncoder para codificar las salidas del modelo (en este caso los scores). Entrega los conjuntos de features y labels en un diccionario {"X":features,"y":labels}
      - **train_xgboost:** Realiza todo el proceso de busqueda y entrenamiento de mejor modelo
      - **store_model:** Despues de lo realizado y entrenado el modelo, el objeto queda con la informacion de los parametros para el LabelEncoder y para el modelo. Este metodo permite almacenar estos parametros bajo la version especificada.
      - **full_training_process:** Realiza completamente el proceso de entrenamiento, una vez finalizado el entrenamiento, muestra al usuario las puntuaciones en el conjunto de entrenamiento y de prueba y da la opcion al usuario de continuar con el proceso o buscar otro modelo

- **./inference_pipelines.ipynb:** Contiene la pipeline y paso a paso para el proceso de prediccion. Sigue el siguiente proceso:
  
  1. **Validacion**
  2. **Procesamiento y ETL**
  3. **Inferencia:**
    - Gira en torno al objeto **InferenceProcess**
    - Para inicializarlo, se requiere la ruta de los datos posprocesados para realizar la prediccion. Como parametro opcional se tiene la version del modelo, si no se especifica se usa la ultima disponible
    - Este tiene los siguientes metodos:
      - **load_data:** Carga los datos para la prediccion
      - **load_label_encoder:** Carga el LabelEncoder para decodificar las predicciones
      - **load_inference_model:** Carga el modelo respectivo
      - **process_features:** Organiza los embeddings para la prediccion, entrega un diccionario con los features a usar para la prediccion {"X":features}
      - **evaluation:** Realiza la prediccion y la almacena en la carpeta de "**./predictions/**"
      - **full_prediction_process:** Realiza completamente el proceso de prediccion, entrega el path en el que se almacenan los resultados de las predicciones.

**NOTA:** En caso que no se desee realizar el proceso de entrenamiento. **En el repositorio se dejan los parametros de un modelo ya entrenado,** para utilizarlo solo siga los pasos en el pipeline para prediccion