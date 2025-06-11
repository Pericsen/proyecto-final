import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta

with open('./credentials/credentials_munidigital.json') as f:
    config = json.load(f)

ACCESS_TOKEN = config["ACCESS_TOKEN"]
CONTENT_TYPE = config["CONTENT_TYPE"]

script_dir = os.path.dirname(os.path.abspath(__file__))
model_data_folder_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'munidigital')
os.makedirs(model_data_folder_path, exist_ok=True)

fecha_hasta = datetime.today().strftime('%d/%m/%Y')
fecha_desde = (datetime.today() - timedelta(days=180)).strftime('%d/%m/%Y')

def get_data():
    url = 'https://munidigital.com/MuniDigitalCore/api/incidentes' # TESTING INCIDENTES

    TOKEN = ACCESS_TOKEN

    headers = {
        "token": TOKEN,
        "Content-Type": CONTENT_TYPE
    }

    params = {
        "fecha_desde": fecha_desde,
        "fecha_hasta": fecha_hasta
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()

        if data['status'] == 'ok' and data['result']:

            df_incidentes = pd.DataFrame(data['result'])
            
            print("datos obtenidos correctamente, guardando en csv.")
            return df_incidentes

        elif data['status'] == 'ok' and not data['result']:
            print("No se encontraron datos")

        else:
            print("Error al obtener los datos")
            print(response.status_code)
            print(response.json())

    else:
        print("Error en la solicitud:", response.status_code)

def preparar_df(df):

    # Elegimos las columnas que necesitamos para entrenar el modelo
    cols = ['areaServicioDescripcion', 'observaciones', 'prioridadDescripcion']

    df = df[cols]   

    # Definimos las areas que queremos usar para la clasificacion
    areas = ['Arbolado Urbano', 
             'Alumbrado', 
             'CLIBA (Higiene Urbana)', 
             'Higiene Urbana', 
             'Obras Públicas']

    # Filtramos el dataframe por las areas que nos interesan
    df = df[(df['areaServicioDescripcion'] == 'Arbolado Urbano') | 
            (df['areaServicioDescripcion'] == 'Alumbrado') | 
            (df['areaServicioDescripcion'] == 'CLIBA (Higiene Urbana)') | 
            (df['areaServicioDescripcion'] == 'Higiene Urbana') | 
            (df['areaServicioDescripcion'] == 'Obras Públicas')]
    
    df = df[df['observaciones'].duplicated()==False] # Eliminamos duplicados

    return df

def guardar_csv(df, fecha_desde, fecha_hasta):
    fecha_desde = fecha_desde.replace('/', '-')
    fecha_hasta = fecha_hasta.replace('/', '-')
    # Guardamos los datos en un csv
    filename = f"incidentes_{fecha_desde}_{fecha_hasta}.csv"
    file_path = os.path.join(model_data_folder_path, filename)
    df.to_csv(file_path, index=False)
    print(f"Datos guardados en {file_path}")

def main():
    df = get_data()
    df = preparar_df(df)
    guardar_csv(df, fecha_desde=fecha_desde, fecha_hasta=fecha_hasta)

if __name__ == '__main__':
    main()