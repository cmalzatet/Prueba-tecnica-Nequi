import pandas as pd
import os
import re
import string
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from libs.utils.file_management import (
    move_file,
    save_dataframe_csv
)


class FileProcessing:
    def __init__(self, source_path:str, required_columns:list):
        self.source_path = source_path
        self.required_columns = required_columns
    
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
            print(f"Error cargando los datos {self.source_path}")
            raise Exception(f"Error cargando los datos {self.source_path} - {e}")
        
    def clean_data(self):
        
        try:            
            self.df = self.df.dropna().drop_duplicates()
            self.df['text'] = self.df['text'].apply(lambda x: re.sub(f"[{string.punctuation}]", "", x.lower()))
            return True
            
        except Exception as e:
            print("Error limpiando los datos - ", e)
            raise e
    
    
    def transform_data(self):
        try:
            sentences = self.df['text'].apply(simple_preprocess).tolist()
            model = Word2Vec(sentences, vector_size=200, window=5, min_count=1, workers=4)

            # Convertir cada texto en su embedding promedio
            def get_embedding(text):
                tokens = simple_preprocess(text)
                vectors = [model.wv[word] for word in tokens if word in model.wv]
                return sum(vectors) / len(vectors) if vectors else [0] * 100

            self.df['embedding'] = self.df['text'].apply(get_embedding)
            return True
            
        except Exception as e:
            print("Error realizando transformacion de la data, ", e)
            raise e
        
    def process(self):
        try:
            analysis_type = self.source_path.split("/")[-2]
            file_name = self.source_path.split("/")[-1]
            
            data_loaded = self.load_data()
            if not data_loaded:
                raise Exception("Error cargando los datos")
            
            if len(self.required_columns) > 1:
            
                data_cleaned = self.clean_data()
                if not data_cleaned:
                    raise Exception("Error limpiando los datos")
            
            data_transformed = self.transform_data()
            if not data_transformed:
                raise Exception("Error transformando los datos")
            
            store_df = self.df[self.required_columns]
            
            new_path = f"./etl/{analysis_type}/{file_name}"
                
            result = save_dataframe_csv(dataframe=store_df, destination=new_path)
            
            if result:
                print("verificacion completa, resultado almacenado en ", new_path)
                return new_path
            
        except Exception as e:
            print("Error procesando los datos: ", e)
            
            file_name = self.source_path.split("/")[-1]
            error_path = f"./errors/{file_name}"
            
            result = move_file(source=self.source_path, destination=error_path)