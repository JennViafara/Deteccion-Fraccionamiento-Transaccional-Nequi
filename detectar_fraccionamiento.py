import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta


def pareto_search_fraccionamiento_transaccional(df_account_number_with_4):
    pareto = {}
    for account_number in tqdm(df_account_number_with_4['account_number'].unique()):
            median = umbral[account_number]
            df_temp = df_account_number_with_4[df_account_number_with_4['account_number']==account_number]
            for misma_ventana in df_temp['misma_ventana'].unique():
                    c = 0
                    sum_transaction_amount = 0
                    df_misma_ventana = df_temp[df_temp['misma_ventana']==misma_ventana]
                    t = len(df_misma_ventana)
                    for transaction_amount in df_misma_ventana['transaction_amount']:
                            if transaction_amount <= median:
                                    c+=1
                                    sum_transaction_amount = sum_transaction_amount + transaction_amount
                    if c/t > 0.6 and (sum_transaction_amount/df_temp['monto_total_user'].iloc[0]) > 0.2 :
                            pareto[misma_ventana]=1
    return pareto

def agrupar_ventanas_24_horas(filtered2_df1):
# Definir la ventana de tiempo como 24 horas
    window_time = timedelta(hours=24)

    # Inicializar variables para seguimiento
    current_account_number = None
    transaction_count = 0
    start_time = None

    # Lista para almacenar los 'account_number' que cumplen con la condición
    account_numbers_with_4_or_more_transactions = []
    _id_24 = []
    indicador_24 = []
    list_id = []
    list_indicador = []
    total_rows = len(filtered2_df1)
    with tqdm(total=total_rows, desc='Procesando filas') as pbar:
        for index, row in filtered2_df1.iterrows():
            if row['account_number'] == current_account_number:
                # Verificar si la diferencia de tiempo es menor o igual a 24 horas
                diff = row['transaction_date'] - start_time
                if diff <= window_time:
                    transaction_count += 1
                    _id_24.append(row['_id'])
                    indicador_24.append(indicador)
                else:
                    # Reiniciar el contador si la diferencia de tiempo supera 24 horas
                    transaction_count = 1
                    start_time = row['transaction_date']
                    _id_24 = [row['_id']]
                    indicador = row['_id']
                    indicador_24 = [indicador]
                filtered2_df1.at[index,'diff_time_hours'] = diff.total_seconds() / 3600.0
            else:
                # Cambiar de cuenta, reiniciar el contador
                current_account_number = row['account_number']
                transaction_count = 1
                start_time = row['transaction_date']
                _id_24 = [row['_id']]
                indicador = row['_id']
                indicador_24 = [indicador]

            # Si se encuentra al menos cuatro transacciones, agregar 'account_number' a la lista
            if transaction_count >= 4 and current_account_number not in account_numbers_with_4_or_more_transactions:
                account_numbers_with_4_or_more_transactions.append(current_account_number)
                list_id.extend(_id_24)
                list_indicador.extend(indicador_24)
            if _id_24[0] not in list_id and transaction_count >=4:
                list_id.extend(_id_24)
                list_indicador.extend(indicador_24)
            if current_account_number in account_numbers_with_4_or_more_transactions and transaction_count > 4:
                list_id.append(_id_24[-1])
                list_indicador.append(indicador_24[-1])
            pbar.update(1)

    return filtered2_df1[filtered2_df1['account_number'].isin(account_numbers_with_4_or_more_transactions)], list_id, list_indicador

### ruta del archivo .csv
     
file_name = 'user_data_group_1.csv'

# file_name = 'user_data_group_2.csv'

# file_name = 'user_data_group_3.csv'

# file_name = 'user_data_group_4.csv'

# file_name = 'user_data_group_5.csv'

# file_name = 'user_data_group_6.csv'

# Cargar uno de los archivos CSV en un DataFrame
df1 = pd.read_csv(file_name)

threshold = len(df1)*0.05
col_to_drop = df1.columns[df1.isna().sum() <= threshold]
df1.dropna(subset=col_to_drop,inplace=True)
print('FILAS CON DATOS FALTANTES ELIMINADAS SI CORRESPONDEN SOLO AL 5% O MENOS')

df1.dropna(axis=1)
print('VALORES NULOS ELIMINADOS')

df1 = df1.drop_duplicates(subset='_id', keep='first')
# Filas duplicadas


# Convertir la columna 'transaction_amount' a tipo float
df1['transaction_amount'] = df1['transaction_amount'].astype(float)
# Ordenar el DataFrame por 'account_number' y 'transaction_date'
df1.sort_values(by=['account_number', 'transaction_date'], inplace=True)


# Contar el número de transacciones por 'user_id' y crear una nueva columna 'num_transactions'
user_transaction_counts = df1.groupby('user_id')['_id'].count().reset_index()
user_transaction_counts.rename(columns={'_id': 'num_transactions'}, inplace=True)
# Filtrar los registros de 'user_id' que tienen al menos 4 transacciones
user_ids_with_4_or_more_transactions = user_transaction_counts[user_transaction_counts['num_transactions'] >= 4]['user_id']
# Filtrar el DataFrame original para mantener solo los registros de los 'user_id' seleccionados
filtered_df1 = df1[df1['user_id'].isin(user_ids_with_4_or_more_transactions)]

# Convertir en el formato adecuado la columna transaction_date
filtered_df1['transaction_date'] = pd.to_datetime(filtered_df1['transaction_date'])
filtered_df1['year_month'] = filtered_df1['transaction_date'].dt.to_period('M')


# Agrupar por 'account_number' y mes, contar transacciones y crear una nueva columna 'transactions_in_month'
filtered_df1['year_month'] = filtered_df1['transaction_date'].dt.to_period('M')  # Crear columna con el mes y año de la transacción
account_monthly_transaction_counts = filtered_df1.groupby(['account_number', 'year_month'])['_id'].count().reset_index()
account_monthly_transaction_counts.rename(columns={'_id': 'transactions_in_month'}, inplace=True)

# Filtrar 'account_number' que hicieron al menos tres transacciones en un mes
accounts_with_3_or_more_transactions_in_month = account_monthly_transaction_counts[account_monthly_transaction_counts['transactions_in_month'] >= 3]['account_number']

# Filtrar el DataFrame original para mantener solo los registros de las cuentas seleccionadas
filtered2_df1 = filtered_df1[filtered_df1['account_number'].isin(accounts_with_3_or_more_transactions_in_month)]

# Eliminar la columna temporal 'year_month' si ya no es necesaria
#filtered_df.drop(columns=['year_month'], inplace=True)



print('*****AGRUPANDO DATOS EN VENTANAS DE 24 HORAS*****')

# Filtrar el DataFrame original para mantener solo los registros de 'account_number' seleccionados
final_filtered_df, list_id, list_indicador = agrupar_ventanas_24_horas(filtered2_df1)

# Copia 'final_filtered_df' en un nuevo DataFrame 'df1'
df = final_filtered_df.copy()
print('****CREANDO COLUMNAS PARA EL PROCESO*****')
tqdm.pandas()
# Añade la columna 'varias_transacciones_xdia' a 'df1' con valores 1 si '_id' está en 'list_id', 0 en otro caso
df['varias_transacciones_xdia'] = df['_id'].progress_apply(lambda x: 1 if x in list_id else 0)

df_mod = df.copy()
df_mod['misma_ventana'] = 0

# Crea un diccionario para mapear _id a indicador
id_to_indicador = dict(zip(list_id, list_indicador))

# Actualiza los valores de 'misma_ventana' basados en '_id'
df_mod['misma_ventana'] = df_mod['_id'].map(id_to_indicador)

# Reemplaza los valores NaN por ceros en la columna 'misma_ventana'
df_mod['misma_ventana'].fillna(0, inplace=True)

# Crear columna con medianas por account number para usar como umbral en el proceso
umbral = df_mod.groupby('account_number')['transaction_amount'].median()
# Agrupa el DataFrame por 'account_number' y calcula el monto total por usuario
df_mod['monto_total_user'] = df_mod.groupby('account_number')['transaction_amount'].transform('sum')

# Filtra data frame para el analisis final
df_account_number_with_4 = df_mod[df_mod['varias_transacciones_xdia']==1]

###En las ventanas de 24 horas buscamos las ventanas que cumplen el criterio que definimos para señalar los posibles fraccionamientos. El criterio consiste en: La ventana debe tener al menos 60% de registros con transaction_amount menor que la mediana de transaction_amount para ese usuario, además la suma de los montos que abarca ese 60% debe ser al menos el 20% del movimiento historico de ese usuario. Estos porcentajes se basan en la ley de Pareto pero con alguna modificación para que se adapte al problema.
print('****BUSQUEDA DE POSIBLE FRACCIONAMIENTO TRANSACCIONAL*****')

pareto = pareto_search_fraccionamiento_transaccional(df_account_number_with_4)

#### Creación de columna donde se marcan los posibles casos de fraccionamiento transaccional
df_mod['posible_fraccionamiento'] = df_mod['misma_ventana'].map(pareto)
df_mod['posible_fraccionamiento'].fillna(0, inplace=True)
df_mod['umbral'] = df_mod['account_number'].map(umbral)

#### Guardado de data frame con detección de posible fraccionamiento 

df_mod.to_csv(file_name[:-4]+'_ft.csv', index=False)