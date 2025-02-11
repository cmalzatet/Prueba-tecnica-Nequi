import os
import pandas as pd

from libs.utils.file_verification import (
    test_read_csv,
    filter_dataframe
)

from libs.utils.file_management import (
    move_file,
    save_dataframe_csv
)

class FileVerification:
    def __init__(self, source_path:str, required_columns:list, valid_formats:list = ["csv"]):
        
        self.source_path = source_path
        self.required_columns = required_columns
        self.valid_formats = valid_formats
        
    def format_verification(self):
    
        try:
            file_format = self.source_path.split(".")[-1]
            
            if file_format in self.valid_formats:
                print("Formato valido")
                return True
            
            else:
                print("Formato no valido")
                return False
        
        except Exception as e:
            print(f"Error verificando formato del archivo {self.source_path}", e)
            raise e

    def readability_verification(self):
        try:
            file_format = self.source_path.split(".")[-1]
            
            if file_format == "csv":
                result = test_read_csv(file_path=self.source_path)
                
                if result:
                    return True            
                else:
                    return False
                
            else:
                print("Formato no valido")
                raise Exception(f"El archivo {self.source_path} no tiene un formato valido")
        except:
            print("No pudo leerse el archivo")
            return False

    def size_verification(self):
        # No implementado para este caso
        # Suponemos que todos los datasets cumplen restriccion de tamaño
        return True
    
    def column_verification(self):
        try:
            file_format = self.source_path.split(".")[-1]
            
            if file_format == "csv":
                self.df = pd.read_csv(self.source_path)
                columns = set(self.df.columns)

                result = set(self.required_columns).issubset(columns)
                
                return result
                
            else:
                print("Formato no valido")
                raise Exception(f"El archivo {self.source_path} no tiene un formato valido")
            
        except Exception as e:
            print(f"Error verificando las columnas del archivo {self.source_path}")
            raise (f"Error verificando las columnas del archivo {self.source_path}")
        

    def column_filtration(self):
        try:
            
            file_format = self.source_path.split(".")[-1]
            
            if file_format == "csv":
                filteredDf = filter_dataframe(dataframe=self.df, cols=self.required_columns)
                
                return filteredDf
                
            else:
                print("Formato no valido")
                raise Exception(f"El archivo {self.source_path} no tiene un formato valido")
            
        except Exception as e:
            print(f"Error filtrando las columnas del archivo {self.source_path}")
            raise e
        
    def verify(self):
        try:
            
            analysis_type = self.source_path.split("/")[-2]
            file_name = self.source_path.split("/")[-1]
            
            print("analysis_type",analysis_type)
            print("file_name",file_name)
            
            valid_format = self.format_verification()
            
            if not valid_format:
                raise Exception("Formato invalido")
            
            readable_file = self.readability_verification()
            
            if not readable_file:
                raise Exception("Formato invalido")
            
            valid_size = self.size_verification()
            
            if not valid_size:
                raise Exception("Tamaño invalido")
            
            valid_columns = self.column_verification()
            
            if not valid_columns:
                raise Exception("Columnas inválidas")
            
            filtered_df = self.column_filtration()
            
            new_path = f"./processed/{analysis_type}/{file_name}"
                
            result = save_dataframe_csv(dataframe=filtered_df, destination=new_path)
            
            if result:
                print("verificacion completa, resultado almacenado en ", new_path)
                return new_path
            
            else:
                print("Error almacenando resultado final")
                raise Exception("Error almacenando resultado final")
            
        except Exception as e:
            print("Error en proceso de verificacion, archivo no valido para procesamiento: ", e)
            print("Moviendo archivo a errors")
            
            file_name = self.source_path.split("/")[-1]
            error_path = f"./errors/{file_name}"
            
            result = move_file(source=self.source_path, destination=error_path)
            