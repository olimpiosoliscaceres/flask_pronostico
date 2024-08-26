from flask import Flask, request, render_template,redirect, url_for,send_file

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

lv_extension='.csv'
lv_nom_file='dato_ca'
lv_nom_file_origen = lv_nom_file + lv_extension

app = Flask(__name__)

def calculate_pronostico(lv_nom_file):
    # Load the data
    lv_extension='.csv'
    lv_nom_file_origen = lv_nom_file + lv_extension

    df = pd.read_csv(lv_nom_file_origen, sep=';', parse_dates=['PERIODO'])
    df['PERIODO'] = pd.to_datetime(df['PERIODO'], format='%Y-%m-%d %H')
    df.set_index('PERIODO', inplace=True)
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1H')
    df = df.reindex(date_range)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'PERIODO'}, inplace=True)

    # ********* 2.- Imputacion con Interpolacion Lineal con restriccion de 5 datos consecutivos ******************
    import numpy as np
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    # 2.1.- Cargar Datos
    lv_nom_file_imputacion = lv_nom_file + '_InterpolaLineal'
    lv_nom_file_imputacion_destino = lv_nom_file_imputacion + lv_extension

    #df = pd.read_csv(lv_nom_file_origen, sep=';', dayfirst=True, parse_dates=['PERIODO'])

    # Crear una copia del DataFrame original
    df_copy = df.copy(deep=True)
    # Establecer la columna PERIODO como el índice
    df_copy.set_index('PERIODO', inplace=True) 

    # 2.2.- Imputar hasta 5 valores faltantes por el metodo de Interpolacion Lineal
    df_copy = df_copy.interpolate(method='linear', limit=5, limit_direction='forward', axis=0)

    # Guarda el DataFrame reindexado en un nuevo archivo CSV con cabecera
    #df_copy.to_csv(lv_nom_file_imputacion_destino, sep=';', header=True)

    # *********** 3.-  Pronostico de datos horarios mediante Random Forest ****************
    # Pronostico de datos horarios de PM10 mediante Random Forest 
    # pip install numpy pandas scikit-learn matplotlib
    # 3.1.- Preparar los datos

    #import pandas as pd
    #import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    # Configuraciones de columnas y archivo
    tx_campo_periodo = 'PERIODO'
    tx_campo = 'PM10'  # "SO2"

    # Cargar datos de la serie temporal
    #data = pd.read_csv('DATOF_PM10_Imputado_Interpolacion.csv', sep=';', parse_dates=[tx_campo_periodo], index_col=tx_campo_periodo)

    # Asegurarse de que los datos estén en el formato correcto
    df_copy = df_copy.asfreq('H')
    #df_copy = df_copy.fillna(method='ffill')  # Rellenar los valores faltantes si es necesario
    df_copy = df_copy.ffill()  # Rellenar los valores faltantes con forward fill

    # Función para crear un conjunto de datos para la predicción multi-step recursiva
    def create_lagged_features(series, n_lags):
        df_tmp = pd.DataFrame(series)
        for lag in range(1, n_lags + 1):
            df_tmp[f'lag_{lag}'] = df_tmp[tx_campo].shift(lag)
        df_tmp.dropna(inplace=True)
        return df_tmp

    # Crear características con retardos
    n_lags = 24  # Número de retardos (horas anteriores a considerar)
    df_lagged = create_lagged_features(df_copy[tx_campo], n_lags)

    # Dividir en características (X) y el objetivo (y)
    x, y = df_lagged.drop(tx_campo, axis=1), df_lagged[tx_campo]

    # 3.2.- Entrenar el modelo Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)

    # 3.3.- Realizar predicciones multi-step recursivas desde el inicio de la serie
    def predict_recursive_from_start(model, df_copy, n_lags, steps):
        predictions = []
        current_input = df_copy.iloc[:n_lags][tx_campo].values  # Inicial input con los primeros n_lags valores reales
        current_input = current_input[-n_lags:]  # Asegurarse de que el tamaño de la entrada es correcto
        #print(df_copy.head(13))
        for i in range(n_lags, len(df_copy)):
            pred = model.predict(current_input.reshape(1, -1))[0]
            predictions.append(pred)
            current_input = np.append(current_input[1:], df_copy.iloc[i][tx_campo])

        # Predicción futura
        for _ in range(steps):
            pred = model.predict(current_input.reshape(1, -1))[0]
            predictions.append(pred)
            current_input = np.append(current_input[1:], pred)
    
        return predictions
    # **************************************************************
    # Predecir desde el inicio hasta las próximas 24 horas
    predictions = predict_recursive_from_start(model, df_copy, n_lags, 24)

    # Crear un DataFrame con las predicciones
    #training_predictions = predictions[:len(data) - n_lags]
    training_predictions = predictions[:len(df_copy)]
    future_predictions = predictions[len(df_copy) - n_lags:]

    all_dates = df_copy.index.tolist() + pd.date_range(start=df_copy.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H').tolist()

    # Verificar las longitudes de las predicciones y las fechas
    #len_predictions = len(training_predictions + future_predictions)
    #len_dates = len(all_dates)

    # Combinar las fechas de entrenamiento y las futuras
    forecast = pd.DataFrame(training_predictions + future_predictions, index=all_dates, columns=['Prediccion'])
    # Dar un nombre al campo índice
    forecast.index.name = 'PERIODO'

    # Unir los DataFrames prevaleciendo el índice de la derecha
    #data_forecast = df_copy.merge(forecast, left_index=True, right_on='PERIODO', how='right')
    #right_index=True en lugar de right_on='PERIODO'
    data_forecast = df_copy.merge(forecast, left_index=True, right_index=True, how='right')

    # Guardar el DataFrame con las predicciones en un nuevo archivo CSV
    lv_nom_file_predicciones= lv_nom_file + '_Predicciones_RF.csv'
    data_forecast.to_csv(lv_nom_file_predicciones, sep=';', header=True, index=True)

    # Graficar las predicciones
    plt.figure(figsize=(12, 6))
    plt.plot(df_copy.index, df_copy[tx_campo], label='Actual')
    plt.plot(forecast.index, forecast['Prediccion'], label='Pronostico', color='red')
    plt.legend()
    plt.xlabel(tx_campo_periodo)
    plt.ylabel(tx_campo)
    plt.title('Pronóstico de 24 Horas de datos Horarios de PM10 de Santa Anita - 2023 Aplicando Inteligencia Artificial')
    plt.show()
   
    mse = mean_squared_error(df_copy, training_predictions)
    rmse = mse ** 0.5
    # Return results
    results_summary = {
        'Root Mean Squared Error (RMSE)': rmse,
        'Sample Predictions': forecast[:10],
    }
    #return results_summary
    return data_forecast

# **** fin de calculate_pronostico() *****************************

@app.route('/')
def home():
    # Cargar los datos desde el archivo CSV
    lv_extension='.csv'
    lv_nom_file='dato_ca'
    lv_nom_file_origen = lv_nom_file + lv_extension    

    #df2 = pd.read_csv('dato_ca_2022_2024_2.csv')
    df2 = pd.read_csv(lv_nom_file_origen, sep=';', parse_dates=['PERIODO'])
    # Convertir el DataFrame a HTML con Bootstrap
    table_html = df2.to_html(classes='table table-striped', index=False)


    export_url = url_for('export', lv_nom_file=lv_nom_file_origen)
    # Renderizar la plantilla y pasar la tabla HTML
    #return render_template('index.html', table_html=table_html)
    return render_template('index.html', table_html=table_html, export_url=export_url)
    #return render_template('index.html')

@app.route('/pronostico', methods=['POST'])
def pronostico():
    #data_path = 'dato_ca_2022_2024_2.csv'
    lv_nom_file='dato_ca'
    lv_nom_file_origen = lv_nom_file + lv_extension
    lv_nom_file_predicciones= lv_nom_file + '_Predicciones_RF.csv'

    #result = calculate_pronostico(lv_nom_file)
    # No se retorna nada aquí
    #pass
    #return render_template('index.html', table_html=table_html)
    # Redirigir al usuario a la página principal
    #return redirect(url_for('index'))
    data_forecast = calculate_pronostico(lv_nom_file)
    
    # Convertir el índice en una columna
    data_forecast.reset_index(inplace=True)
    
    table_html_data_forecast = data_forecast.to_html(classes='table table-striped', index=False)

    # Generar el enlace de exportación con el nombre del archivo
    #export_url = url_for('export', lv_nom_file='dato_ca_2022_2024_2.csv')
    #export_url = url_for('export', lv_nom_file='dato_ca_2022_2024_2_Predicciones_RF.csv')
    export_url = url_for('export', lv_nom_file=lv_nom_file_predicciones)
    
    
    # Renderizar la plantilla y pasar la tabla HTML y el enlace de exportación
    return render_template('index.html', table_html=table_html_data_forecast, export_url=export_url)
    
    # Renderizar la plantilla y pasar la tabla HTML
    #return render_template('index.html', table_html=table_html_data_forecast)
    ##return redirect(url_for('home'))

#@app.route('/export/<lv_nom_file>')
#def export(lv_nom_file):
@app.route('/export')
def export():
    lv_nom_file = request.args.get('lv_nom_file', default=None, type=str)
    print('lv_nom_file:',lv_nom_file)
    #lv_extension='.csv'
    #lv_nom_file='dato_ca_2022_2024_2'
    #lv_nom_file_extension = lv_nom_file + lv_extension

    if lv_nom_file is None:
        return "Error: falta el nombre del archivo para exportar.", 400
    
    # Aquí deberías tener la lógica para generar o ubicar el archivo para exportar.
    path_to_file = f"{lv_nom_file}"
    
    try:
        return send_file(path_to_file, as_attachment=True)
    except Exception as e:
        return str(e), 500     
        
if __name__ == '__main__':
    app.run(debug=True)
