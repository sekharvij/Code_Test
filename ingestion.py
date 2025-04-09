import pandas as pd

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format.")

def merge_data(df1, df2, on_columns):
    
    if isinstance(on_columns, str):
        on_columns = [col.strip() for col in on_columns.split(',')]
    return df1.merge(df2, on=on_columns, how='inner')
