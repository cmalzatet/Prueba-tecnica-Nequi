# Propuesta de arquitectura (AWS)

- [Propuesta de arquitectura (AWS)](#propuesta-de-arquitectura-aws)
  - [1. Introducción](#1-introducción)
  - [2. Arquitectura](#2-arquitectura)
  - [3. Almacenamiento de datos](#3-almacenamiento-de-datos)
    - [Estructura de directorios](#estructura-de-directorios)
  - [4. Ingesta](#4-ingesta)
  - [5. Procesamiento y ETL](#5-procesamiento-y-etl)
  - [6. Entrenamiento del modelo](#6-entrenamiento-del-modelo)
  - [7. Inferencia del modelo en baches](#7-inferencia-del-modelo-en-baches)



## 1. Introducción

En este archivo documenta la estructura, detalles y decisiones tomadas en cuanto a la arquitectura de la solución, la cual es propuesta sobre la nube de AWS, buscando sacar provecho de las herramientas especializadas con las que se cuenta en esta.

## 2. Arquitectura

![arquitectura](/Documentacion/imgs/Diagrama%20arquitectura.jpg)

## 3. Almacenamiento de datos

El almacenamiento de información, tanto datos crudos, datos procesados para el entrenamiento e inferencia, resultados de inferencias, versiones de modelos y logs se hace en **AWS S3**.

La carga de datos, tanto para entrenamiento como inferencia puede realizarse manualmente al bucket de s3, por una plataforma o interfaz usando integraciones como el SDK de python o consultas API REST, o de forma automática con eventos disparados cuando se realice alguna acción en otro recurso (como recepción de mensajes o reviews).

Se decidió esta alternativa para el almacenamiento por lo fácil y rápida que resulta la integración de S3 para la carga de información, también permite almacenamiento de datasets masivos con costos relativamente bajos y al ser nativo de AWS, la conexión con las herramientas de procesamiento es directa.

### Estructura de directorios

La estructura de directorios propuesta es:

```plaintext
s3://amazon-reviews-nlp/
    ├── raw/   
    │   ├── train/
    │   └── inference/
    │   
    ├── errors/
    │      
    ├── chunks/
    │   ├── train/
    │   └── inference/
    │
    ├── processed/
    │   ├── train/
    │   └── inference/    
    │ 
    ├── etl/
    │   ├── train/
    │   └── inference/
    │
    ├── models/
    │   ├── current/
    │   ├── v1.0/
    │   ├── v1.1/
    │   └── ...
    │
    ├── predictions/
    │
    ├── monitoring/
    │   ├── logs/
    │   ├── metrics/
    │   └── alerts/
```

Esta está distribuida de la siguiente forma:

- **raw/:** En esta carpeta se almacenan los nuevos conjuntos de datos antes de verificacion y procesamiento o datos crudos
  - **train/:** Datos crudos para entrenamiento
  - **inference/:** Datos crudos para inferencia

- **errors/:** En esta se almacenan los archivos que no cumplen con las condiciones necesarias para pasar al ETL

- **chunks/:** Carpeta en la que se almacenan archivos particionados por exceder el tamaño máximo para pasar al ETL
  - **train/:** Carpeta archivos particionados para entrenamiento
  - **inference/:** Carpeta archivos particionados para inferencia

- **processed/:** Almacena los documentos verificados y filtrados (en caso de ser necesario). Estos ya están listos para pasar al ETL
  - **train/:** Archivos para entrenamiento listos para ETL
  - **inference/:** Archivos para inferencia listos para ETL

- **etl/:** Carpeta que almacena los datos después de pasar por el ETL.
  - **train/:** Archivos para entrenamiento post ETL
  - **inference/:** Archivos para inferencia post ETL

- **models/:** Carpeta para almacenamiento de diferentes versiones del modelo
  - **current/:** Almacena el último modelo en producción
  - **vX.X/:** Cada subcarpeta indica la versión del modelo y contiene los documentos que lo componen

- **predictions/:** Almacena las predicciones/inferencias realizadas por el modelo en producción

- **monitoring/:** Carpeta para almacenamiento del monitoreo del proceso en general.
  - **logs/:** Almacena los logs de todas las partes del proceso
  - **metrics/:** Almacena métricas obtenidas en entrenamientos
  - **alerts/:** Almacena alertas disparadas por problemas o errores en las diferentes partes del proceso

## 4. Ingesta 

Al momento de recibir datos nuevos, se realiza la detección y verificaciones utilizando una **AWS Lambda**. Al detectarse un nuevo archivo en la carpeta de entrada "/raw/train/" (para entrenamiento) o "/raw/inference/" (para inferencia), la Lambda realiza el siguiente proceso: 

1. **Verificación de formato:**
    - Si el archivo no tiene el formato adecuado, se mueve a la carpeta "/errors/" y se genera el reporte
    - Si el archivo posee formato compatible, pasa al siguiente paso
  
2. **Verificación de lectura:**
   - Si el archivo no puede leerse (dañado o corrupto), se mueve a la carpeta "/errors/" y se genera el reporte
   - Si el archivo puede leerse correctamente, pasa al siguiente paso
  
3. **Verificación de tamaño:**
   -  Si el tamaño del archivo es menor al límite inferior, se mueve a la carpeta "/errors/" y se genera el reporte
   -  Si el tamaño del archivo es mayor límite superior, este se divide en diferentes partes y estas se almacenan en la carpeta "chunks/train" o "chunks/inference" según el caso
   - Si el tamaño del archivo está entre el rango aceptable, pasa al siguiente paso

4. **Verificación de columnas:**
   - Si el archivo no contiene las columnas necesarias, se mueve a la carpeta "/errors/" y se genera el reporte
   - Si el archivo contiene justamente las columnas necesarias, pasa al siguiente paso
   - Si el archivo contiene las columnas necesarias y además de estas, tiene información extra, se filtran las columnas con información innecesaria para el proceso y el archivo filtrado pasa al siguiente paso

5. **Almacenamiento de documentos para ETL:** Los archivos aprobados y preprocesados son almacenados en "/processed/train/" o "/processed/inference/" según el caso

**Nota:** Los archivos particionados y almacenados en "chunks/" disparan el proceso de verificación, pero realizado sobre los mismos archivos particionados.


Como se puede observar en el proceso de verificación, la Lambda también registra eventos, procesos y logs. Después de finalizar el proceso de verificación, la Lambda procede a iniciar automáticamente el pipeline de ETL.

Se escoge AWS Lambda para este proceso de ingesta porque permite conectarse y recibir los eventos generados por el S3 de almacenamiento además de las conexiones con las herramientas de procesamiento, dando la oportunidad de automatizar varios procesos y pipelines. Por otro lado, al poder desplegarse en entornos serverless, lo que permite ejecutar varias tareas de verificación en paralelo, dando paso a un proceso escalable.

## 5. Procesamiento y ETL

Para la ejecución de las tareas relacionadas con el procesamiento y ETL, se escoge **AWS Glue**, pues es una de las herramientas especializadas que AWS ofrece para este tipo de procesos. Además de ser serverless, permite la ejecución de tareas de procesamiento usando Apache Spark.

Los procesos encapsulados en la tarea de procesamiento son los siguientes:

1. **Carga de datos:**
   - Se cargan los datos seleccionados de las carpetas "/processed/training/" o "/processed/inference/" según el caso a Glue

2. **Limpieza de datos:**
   - Se eliminan valores nulos o corruptos
   - Se normalizan los textos:
     - Textos a minúsculas
     - Eliminación de carácteres especiales
     - Si el caso lo require, se eliminan stop words

3. **Transformación de datos:**
   - Se realiza tokenización de textos
   - Se pueden agregar columnas resultado de operaciones entre las columnas ya existentes
 
4. **Almacenamiento de resultados:**
   - Las estructuras de datos resultantes son entonces almacenadas en las carpetas de S3 "/etl/training/" o "/etl/inference/" según el caso.
  
Si en algún paso del proceso se presentan problemas o errores, el archivo es desplazado a "/errors/".

## 6. Entrenamiento del modelo

Para el entrenamiento del modelo, similar al caso anterior, escogemos la herramienta especializada que AWS pone a nuestra disposición, **AWS SageMaker**. Esta decisión tomada por la capacidad y flexibilidad que este provee. Compatible con frameworks de machine learning, posee la capacidad de auto escalamiento de acuerdo al volúmen de los datos. Por otro lado, ofrece beneficios como la búsqueda automática de hiperparámetros para los modelos.

Para la selección del modelo, se poseen varias opciones. Si la implementación se realizara directamente en SageMaker, se podría utilizar el algoritmo **BlazingText**, optimizado para el procesamiento de textos y tareas de NLP, cuenta con un de tokenización propio (por lo cual no sería necesario realizar la tokenización en el paso anterior). Otra opción es el fine-tuning de modelos basados en BERT, obteniendo los modelos base de Hugging Face y ejecutando la tarea de entrenamiento y hosting en SageMaker. Finalmente, otra opción es usar modelos de clasificación menos exigente en cuanto a recursos de procesamiento. <u>Por cuestiones de rapidez y recursos, en este caso se implementan modelos basados en algoritmos XGBoost.</u>

Considerando esto, el proceso a realizar con SageMaker sería el siguiente:

1. **Carga de datos:** Se leen los datos almacenados en "/etl/training/".

2. **Partición de datos:** Los datos se dividen en conjuntos de entrenamiento (train) y prueba (test).

3. **Selección y entrenamiento del modelo:** Se escoge el modelo a entrenar. Como se explicó anteriormente, en el presente se utiliza un algoritmo de XGBoost. Luego de esto se seleccionan los hiperparámetros del modelo y se procede al proceso de entrenamiento. Una alternativa a esto es ejecutar un **GridSearch** que permite buscar el mejor conjunto de hiperpámetros en un subespacio definido.

4. **Validación del modelo:** Después de el entrenamiento del modelo, se valida el modelo con el conjunto de prueba. Según los resultados obtenidos, se puede realizar una modificación en los hiperparámetros buscando un modelo con mejores resultados o se puede proceder al siguiente paso.

5. **Almacenamiento y versionado:** El modelo entrenado se almacena tanto en la carpeta "/models/current/" como en la carpeta "/models/vX.X/" donde vX.X son el indicativo de la versión correspondiente del modelo.

## 7. Inferencia del modelo en baches

Inicialmente, cuando llegan datos para ser clasificados por el modelo, estos son almacenados en la carpeta "/raw/inference/" y se les aplican las verificaciones y procesamiento expresado en los pasos anteriores. 

Una vez se tiene los datos listos para el modelo en la carpeta "/etl/inference/", se inicia un job a **AWS Batch**, herramienta que ejecuta los trabajos de inferencia con escalamiento horizontal, permitiendo correr varios de estos en paralelo.

Finalmente, después de hacer las inferencias correspondientes a un conjunto de datos, los resultados son almacenados en la carpeta "/predictions/" para finalmente ser entregados al usuario.

