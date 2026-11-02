import pandas as pd


def check_data_quality(df):

    missing_values = df.isnull().sum()

    duplicate_rows = df.duplicated().sum()

    data_types = df.dtypes.astype(str)

    return missing_values, duplicate_rows, data_types


def clean_data(df):

    df_clean = df.copy()

    # Remove duplicate records
    df_clean = df_clean.drop_duplicates()

    # Fill missing numerical values
    numerical_columns = df_clean.select_dtypes(
        include=["number"]
    ).columns

    for col in numerical_columns:
        df_clean[col] = df_clean[col].fillna(
            df_clean[col].median()
        )

    # Fill missing categorical values
    categorical_columns = df_clean.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for col in categorical_columns:
        mode = df_clean[col].mode(dropna=True)

        if not mode.empty:
            df_clean[col] = df_clean[col].fillna(mode[0])

    return df_clean