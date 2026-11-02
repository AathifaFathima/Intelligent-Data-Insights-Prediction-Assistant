"""Reusable preprocessing + model training for tabular CSV datasets."""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _prepare_features(df, target_column):
    """Remove obvious IDs and transform recognizable date/time columns."""
    X = df.drop(columns=[target_column]).copy()
    removed = []
    date_columns = []
    for col in list(X.columns):
        name = str(col).lower().strip()
        if re.search(r'(^id$|(^|_)id$|^id_|_uuid$|^uuid$)', name) or name in {'student_id', 'task_id', 'user_id'}:
            removed.append(col)
            X = X.drop(columns=col)
            continue
        if pd.api.types.is_datetime64_any_dtype(X[col]) or any(token in name for token in ('date', 'time', 'timestamp', 'deadline')):
            parsed = pd.to_datetime(X[col], errors='coerce')
            if parsed.notna().mean() >= 0.75:
                X[f'{col}__year'] = parsed.dt.year.astype(float)
                X[f'{col}__month'] = parsed.dt.month.astype(float)
                X[f'{col}__dayofweek'] = parsed.dt.dayofweek.astype(float)
                X[f'{col}__hour'] = parsed.dt.hour.astype(float)
                date_columns.append(col)
                X = X.drop(columns=col)
    # Prevent huge one-hot expansions from free-text / nearly-unique values.
    for col in list(X.select_dtypes(include=['object', 'string', 'category']).columns):
        n_unique = X[col].nunique(dropna=True)
        if n_unique > 100 and n_unique / max(len(X), 1) > 0.5:
            removed.append(col)
            X = X.drop(columns=col)
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(float)
    return X, removed, date_columns


def _build_pipeline(X, model_type):
    numerical = X.select_dtypes(include=['number']).columns.tolist()
    categorical = [c for c in X.columns if c not in numerical]
    if not numerical and not categorical:
        raise ValueError('No usable feature columns remain after preprocessing.')
    transformers = []
    if numerical:
        transformers.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=False))
        ]), numerical))
    if categorical:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=5,
                                      max_categories=30, sparse_output=True))
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    if model_type == 'Classification':
        estimator = RandomForestClassifier(n_estimators=80, random_state=42,
                                           n_jobs=-1, min_samples_leaf=2)
    else:
        estimator = RandomForestRegressor(n_estimators=80, random_state=42,
                                          n_jobs=-1, min_samples_leaf=2)
    return Pipeline([('preprocessor', preprocessor), ('model', estimator)])


def train_dynamic_model(df, target_column, model_type):
    """Return trained pipeline, evaluation dictionary and UI metadata."""
    if target_column not in df.columns:
        raise ValueError('Target column not found in dataset.')
    if model_type not in ('Classification', 'Regression'):
        raise ValueError('Choose Classification or Regression.')
    data = df.dropna(subset=[target_column]).copy()
    if len(data) < 20:
        raise ValueError('At least 20 non-missing target rows are needed for a useful train/test split.')
    y = data[target_column]
    if model_type == 'Regression':
        y = pd.to_numeric(y, errors='coerce')
        if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError('Regression target must contain only finite numeric values.')
        if y.nunique() < 2:
            raise ValueError('Regression target must have different values.')
    else:
        if y.nunique() < 2:
            raise ValueError('Classification target needs at least 2 classes.')
        if y.nunique() > min(50, len(y) // 5):
            raise ValueError('Too many unique target values for classification. Choose Regression for continuous numbers.')
        if y.value_counts().min() < 2:
            raise ValueError('Each class needs at least 2 rows for stratified evaluation.')
        y = y.astype(str)
    X, removed, date_columns = _prepare_features(data, target_column)
    if X.shape[1] == 0:
        raise ValueError('No usable input columns remain (only IDs or high-cardinality text).')
    # Convert unhandled types into string categories and replace infinities.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype('string').astype(object).where(X[col].notna(), np.nan)
    X = X.replace([np.inf, -np.inf], np.nan)
    stratify = y if model_type == 'Classification' else None
    if stratify is not None and (len(y) * 0.2 < y.nunique()):
        raise ValueError('Not enough rows for every class to appear in the test split.')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipeline = _build_pipeline(X, model_type)
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    if model_type == 'Classification':
        labels = pipeline.named_steps['model'].classes_.tolist()
        metrics = {
            'Accuracy': float(accuracy_score(y_test, predicted)),
            'Precision': float(precision_score(y_test, predicted, average='weighted', zero_division=0)),
            'Recall': float(recall_score(y_test, predicted, average='weighted', zero_division=0)),
            'F1 Score': float(f1_score(y_test, predicted, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, predicted, labels=labels).tolist(),
            'labels': labels,
        }
    else:
        metrics = {
            'MAE': float(mean_absolute_error(y_test, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, predicted))),
            'R2': float(r2_score(y_test, predicted)),
        }
    info = {
        'model_type': model_type,
        'target_column': str(target_column),
        'feature_columns': [str(c) for c in X.columns],
        'excluded_columns': [str(c) for c in removed],
        'converted_date_columns': [str(c) for c in date_columns],
        'training_rows': int(len(X_train)),
        'testing_rows': int(len(X_test)),
        'model': type(pipeline.named_steps['model']).__name__,
        'note': 'This is an exploratory model. Inspect target leakage and model quality before real-world use.',
    }
    return pipeline, metrics, info, X


def prepare_prediction_row(values, original_df, target_column, expected_columns):
    """Apply the same feature engineering to a new input row."""
    row = pd.DataFrame([values])
    row[target_column] = pd.NA
    # _prepare_features uses cardinality to remove text columns; use training
    # feature names to align outputs instead of re-detecting high-cardinality.
    row, _, _ = _prepare_features(row, target_column)
    for col in expected_columns:
        if col not in row.columns:
            row[col] = np.nan
    return row.reindex(columns=expected_columns)
