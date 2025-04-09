import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer

def impute_missing(df, strategy='mean'):
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=3)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                # Explicit for categoricals
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def transform_numeric(df, method='normalize'):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'normalize':
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif method == 'standardize':
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif method == 'log':
        for col in num_cols:
            df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    return df

def encode_categoricals(df, method='onehot'):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    if method == 'onehot':
        return pd.get_dummies(df, columns=cat_cols)

    elif method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    else:
        raise ValueError("Encoding method must be 'onehot' or 'label'")
