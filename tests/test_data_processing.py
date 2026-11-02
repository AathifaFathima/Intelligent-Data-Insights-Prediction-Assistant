import pandas as pd

from data_processing import check_data_quality, clean_data


def test_remove_duplicates():

    df = pd.DataFrame({
        "Name": ["A", "A", "B"],
        "Marks": [80, 80, 90]
    })

    cleaned_df = clean_data(df)

    assert len(cleaned_df) == 2


def test_missing_values():

    df = pd.DataFrame({
        "Marks": [80, None, 90]
    })

    cleaned_df = clean_data(df)

    assert cleaned_df["Marks"].isnull().sum() == 0


def test_data_quality():

    df = pd.DataFrame({
        "Marks": [80, None, 90]
    })

    missing, duplicates, data_types = check_data_quality(df)

    assert missing["Marks"] == 1
    assert duplicates == 0

def test_no_duplicates():

    df = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Marks": [70, 80, 90]
    })

    cleaned_df = clean_data(df)

    assert len(cleaned_df) == 3


def test_multiple_missing_values():

    df = pd.DataFrame({
        "Marks": [70, None, 90, None, 80]
    })

    cleaned_df = clean_data(df)

    assert cleaned_df["Marks"].isnull().sum() == 0


def test_multiple_duplicates():

    df = pd.DataFrame({
        "Name": ["A", "A", "B", "B", "C"],
        "Marks": [70, 70, 80, 80, 90]
    })

    cleaned_df = clean_data(df)

    assert len(cleaned_df) == 3

def test_numeric_missing_value_filled_with_median():

    df = pd.DataFrame({
        "Marks": [10, 20, None, 40, 50]
    })

    cleaned_df = clean_data(df)

    assert cleaned_df["Marks"].iloc[2] == 30

def test_categorical_missing_value_filled():

    df = pd.DataFrame({
        "Grade": ["A", "A", None, "B"]
    })

    cleaned_df = clean_data(df)

    assert cleaned_df["Grade"].isnull().sum() == 0