import pandas as pd
import numpy as np
import ast
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split

from libs.utils.file_management import (
    move_file,
    save_dataframe_csv
)


def string_list_processing(text):
    text = text.replace("\n", " ")  
    text = ",".join(text.split())
    text = text.replace("[,","[")
    text = text.replace(",,",",")
    return text

class InferenceProcess:
    def __init__(self,data_inference_path:str, model_version:str="current"):
        self.source_path = data_inference_path
        self.model_version = model_version
        
    def load_data(self):
        try:
            file_format = self.source_path.split(".")[-1]
            
            if file_format == "csv":
                self.df = pd.read_csv(self.source_path)
                return True
                                
            else:
                print("Formato no valido")
                raise Exception(f"El archivo {self.source_path} no tiene un formato valido")
            
        except Exception as e:
            print(f"Error cargando datos {self.source_path}")
            raise Exception(f"Error cargando los datos {self.source_path} - {e}")
        
    def load_label_encoder(self):
        
        try:
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.load(f"./model/{self.model_version}/label_encoder.npy")
            return True
            
        except Exception as e:
            print("Error cargando label encoder, ", e)
            raise e
        
    def load_inference_model(self):
        try:
            self.model = XGBClassifier()
            self.model.load_model(f"./model/{self.model_version}/model_params.json")
            
            
            return True
            
        except Exception as e:
            print("Error cargando label encoder, ", e)
            raise e
        
        
    def process_features(self):
        try:
            self.df["embedding"] = self.df["embedding"].apply(lambda x: string_list_processing(x))
            self.df["embedding"] = self.df["embedding"].apply(ast.literal_eval)
            embeddigs_df = pd.DataFrame(self.df["embedding"].to_list(), index=self.df.index)
            embeddigs_df.head()
            embeddigs_df.columns = [f"{i}" for i in range(embeddigs_df.shape[1])]
            
            X = embeddigs_df
            
            print("Label Encoder almacenado en .label_encoder")
            
            return {"X": X}
        
        except Exception as e:
            print(f"Error procesando datos {self.source_path}")
            raise Exception(f"Error procesando los datos {self.source_path} - {e}")
        

    def evaluation(self, X):
        try:
            prediction = self.model.predict(X)
            
            decoded_predictions = self.label_encoder.inverse_transform(prediction)
            
            predictionDf = pd.DataFrame(decoded_predictions, columns=["inference"])
            
            file_name = self.source_path.split('/')[-1]
            result_storage_path = f'./predictions/inference_{self.source_path}'
            
            storage_result = save_dataframe_csv(predictionDf, result_storage_path)
            
            if not storage_result:
                raise Exception("error almacenando los resultados")
            
            
            return result_storage_path
        
        except Exception as e:
            print("Error al realizar prediccion, ", e)
            raise e
        
    def full_prediction_process(self):
        try:
            print("lectura de datos")
            data_loaded = self.load_data()
            
            if not data_loaded:
                raise Exception("Error cargando datos")
            
            print("organizando catacteristicas para la prediccion")
            features = self.process_features()
            if not data_loaded:
                raise Exception("Error procesando caracteristicas")
            
            print("carga de label encoder")
            label_encoder_loaded = self.load_label_encoder()
            
            if not label_encoder_loaded:
                raise Exception("Error cargando label encoder")
            
            print("carga de modelo")
            model_loaded = self.load_inference_model()
            
            if not model_loaded:
                raise Exception("Error cargando modelo")
            
            X = features["X"]
            
            print("ejecutando prediccion")
            result_storage_path = self.evaluation(X=X)
            
            if not result_storage_path:
                raise Exception("Error realizando prediccion")
            
            print(f"resultados almacenados en: ", result_storage_path)
            
            return result_storage_path
            
            
        except Exception as e:
            print("Error realizando prediccion, ", e)
            raise e