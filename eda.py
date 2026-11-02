import pandas as pd


def statistical_summary(df):

    numerical_data = df.select_dtypes(include=["number"])

    if numerical_data.empty:
        return pd.DataFrame()

    summary = numerical_data.describe().T

    summary = summary.rename(columns={
        "count": "Count",
        "mean": "Mean",
        "std": "Standard Deviation",
        "min": "Minimum",
        "25%": "25th Percentile",
        "50%": "Median",
        "75%": "75th Percentile",
        "max": "Maximum"
    })

    return summary
