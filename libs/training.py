import pandas as pd
import numpy as np
import ast
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split


def string_list_processing(text):
    text = text.replace("\n", " ")  
    text = ",".join(text.split())
    text = text.replace("[,","[")
    text = text.replace(",,",",")
    return text

class TrainingProcess:
    def __init__(self, source_path:str, version:str):
        self.source_path = source_path
        self.version = version
        self.label_encoder = None
        self.model = None
        
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
        
    
    def process_features(self):
        try:
            self.df["embedding"] = self.df["embedding"].apply(lambda x: string_list_processing(x))
            self.df["embedding"] = self.df["embedding"].apply(ast.literal_eval)
            embeddigs_df = pd.DataFrame(self.df["embedding"].to_list(), index=self.df.index)
            embeddigs_df.head()
            embeddigs_df.columns = [f"{i}" for i in range(embeddigs_df.shape[1])]
            
            X = embeddigs_df
            y = self.df["score"]
            
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            
            print("Label Encoder almacenado en .label_encoder")
            
            return {"X": X, "y": y}
        
        except Exception as e:
            print(f"Error procesando datos {self.source_path}")
            raise Exception(f"Error procesando los datos {self.source_path} - {e}")
        
        
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        try:
            print("Definiendo modelo")
            classifier = XGBClassifier(n_estimators=100,random_state=42)

            search_space = {
                "max_depth": Integer(2,8),
                "learning_rate": Real(0.001, 1.0, prior="log-uniform"),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
                "colsample_bylevel": Real(0.5, 1.0),
                "colsample_bynode": Real(0.5, 1.0),
                "reg_alpha": Real(0.0, 10.0),
                "reg_lambda": Real(0.0, 10.0),
                "gamma": Real(0.0, 10.0)
            }
            
            opt = BayesSearchCV(classifier, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=42)
            
            print("Buscando mejor modelo...")
            opt.fit(X_train, y_train)
            
            print("Probando...")
            test_score = opt.score(X_test, y_test)
            
            self.model = opt.best_estimator_
            
            print(f"Puntuacion entrenamiento: {opt.best_score_} \n Puntuacion prueba: {test_score}")
            print("Modelo almacenado en .modelo")
            
            return True
            
        except Exception as e:
            print(f"Error procesando datos")
            raise Exception(f"Error procesando los datos - {e}")
        
    def store_model(self):
        try:
            if not self.version:
                raise Exception("Ingrese una version valida")
            
            model_path = f"./model/v{self.version}/"
            current_path = "./model/current/"
            
            if not os.path.isdir("./model/"):
                print("creando carpeta de modelos")
                os.makedirs(os.path.dirname("./model/"))
                
            if not os.path.isdir(model_path):
                print(f"creando carpeta de version {self.version}")
                os.makedirs(os.path.dirname(model_path))
                
            if not os.path.isdir(current_path):
                print("creando carpeta de modelo actual")
                os.makedirs(os.path.dirname(current_path))
            
            if os.path.isdir(model_path):
                raise Exception("Ya hay un registro de esta version")
            
            
            np.save(model_path+"label_encoder.npy", self.label_encoder.classes_ )
            np.save(current_path+"label_encoder.npy", self.label_encoder.classes_ )
            
            self.model.save_model(model_path+"model_params.json")
            self.model.save_model(current_path+"model_params.json")
            
            return [model_path, current_path]
        
        except Exception as e:
            print("Error almacenando modelo ", e)
            raise e
        
    
    def full_training_process(self):
        try:
            
            if not self.version:
                raise Exception("Ingrese version valida")
            
            data_loaded = self.load_data()
            if not data_loaded:
                raise Exception("Error cargando datos")
            
            features = self.process_features()
            if not data_loaded:
                raise Exception("Error procesando caracteristicas")
            
            X = features["X"]
            y = features["y"]
            
            finish = False
            
            while not finish:
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model_trained = self.train_xgboost(X_train, y_train, X_test, y_test)
                
                if not model_trained:
                    raise Exception("Error en entrenamiento de modelo")
                    finish = True
                    
                else:
                    finalizar_str = input("desea volver a intentar [S] - si, [N] - no")
                    
                    if finalizar_str.lower() == "s" or finalizar_str.lower() == "si":
                        finish = True
                        print("finalizando proceso")
                        
                    elif finalizar_str.lower() == "n" or finalizar_str.lower() == "no":
                        finish = False
                        print("repitiendo proceso")
                    
                    else:
                        finish = True
                        print("entrada invalida, continuando con el proceso")
                        
            
            model_stored = self.store_model()
            if not model_stored:
                raise Exception("Error almacenando modelo")
            
            print("modelo almacenado en: ", model_stored)
            
            return model_stored
        
        except Exception as e:
            print("Error en proceso de entrenamiento ,", e)
            raise e