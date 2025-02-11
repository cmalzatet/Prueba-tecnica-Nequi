import os
import pandas as pd

def test_read_csv(file_path:str):
    
    try:
        df = pd.read_csv(file_path)
        print(".csv leido con exito")
        return True
        
    except Exception as e:
        print("Error al leer CSV")
        raise e

def filter_dataframe(dataframe:pd.DataFrame, cols:list):
    try:
        new_df = dataframe[cols]
        print("Dataframe filtrado correctamente")
        return new_df
    
    except Exception as e:
        print(f"Error filtrando dataframe")
        raise e
