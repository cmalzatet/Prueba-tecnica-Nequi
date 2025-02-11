import os
import pandas as pd
import shutil

def move_file(source:str, destination:str):
    
    try:
        if not os.path.isfile(source):
            print(f"Archivo {source} no existe")
            raise Exception(f"Archivo {source} no existe")
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        shutil.move(source, destination)
        
        print("Movimiento completado exitosamente")
        
        return True
    
    except Exception as e:
        print("Error al movel el archivo: ", e)
        raise e
    
def save_dataframe_csv(dataframe:pd.DataFrame, destination:str):
    
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        dataframe.to_csv(destination, index=False)
        
        print(f"Dataframe guardado correctamente en {destination}")
        return True
    
    except Exception as e:
        print("Error almacenando dataframe")
        raise e