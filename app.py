import hashlib
import io
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_processing import check_data_quality, clean_data
from model_training import train_dynamic_model, prepare_prediction_row

st.set_page_config(page_title='Intelligent Data Insights', page_icon='📊', layout='wide')
st.title('📊 Intelligent Data Insights & Prediction Assistant')
st.caption('Upload a CSV to explore, clean, train a model, predict and export results.')

uploaded = st.file_uploader('Upload CSV dataset', type=['csv'])
if uploaded is None:
    st.info('Upload a CSV file to begin.')
    st.stop()

raw = uploaded.getvalue()
file_id = hashlib.sha256(raw).hexdigest()
if st.session_state.get('dataset_id') != file_id:
    for key in ('model_bundle', 'train_metrics'):
        st.session_state.pop(key, None)
    st.session_state.dataset_id = file_id

try:
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
except Exception as exc:
    st.error(f'Could not read CSV: {exc}')
    st.stop()
if df.empty or not len(df.columns):
    st.error('The CSV has no data rows or columns.')
    st.stop()
if df.columns.duplicated().any():
    st.error('CSV has duplicate column names. Please rename them first.')
    st.stop()

st.success(f'Loaded {len(df):,} rows and {len(df.columns)} columns.')
st.subheader('1. Dataset Preview')
st.dataframe(df.head(20), use_container_width=True)
a, b, c = st.columns(3)
a.metric('Rows', f'{len(df):,}')
b.metric('Columns', len(df.columns))
c.metric('Missing Cells', f'{df.isna().sum().sum():,}')

st.subheader('2. Data Quality')
missing, duplicates, types = check_data_quality(df)
st.write(f'Duplicate rows: **{duplicates:,}**')
st.dataframe(pd.DataFrame({'Column': df.columns, 'Missing': df.isna().sum().values,
                           'Type': df.dtypes.astype(str).values}), hide_index=True)

st.subheader('3. Data Cleaning')
try:
    cleaned = clean_data(df.copy())
    st.write(f'After cleaning: **{len(cleaned):,} rows**, **{cleaned.isna().sum().sum():,} missing cells**.')
    st.dataframe(cleaned.head(10), use_container_width=True)
    st.download_button('📥 Download Cleaned CSV', cleaned.to_csv(index=False).encode('utf-8'),
                       'cleaned_dataset.csv', mime='text/csv')
except Exception as exc:
    st.warning(f'Automatic cleaning could not complete: {exc}')
    cleaned = df.copy()

st.subheader('4. Exploratory Data Analysis')
numeric_cols = cleaned.select_dtypes(include='number').columns.tolist()
if numeric_cols:
    st.dataframe(cleaned[numeric_cols].describe().T, use_container_width=True)
else:
    st.info('No numerical columns for numeric summary.')

st.subheader('5. Data Visualization')
if numeric_cols:
    hist_col = st.selectbox('Histogram column', numeric_cols)
    fig, ax = plt.subplots()
    ax.hist(pd.to_numeric(cleaned[hist_col], errors='coerce').dropna(), bins=25)
    ax.set_title(f'Distribution of {hist_col}')
    ax.set_xlabel(hist_col)
    st.pyplot(fig)
    plt.close(fig)
if len(numeric_cols) >= 2:
    xcol = st.selectbox('Scatter X', numeric_cols)
    ycol = st.selectbox('Scatter Y', numeric_cols, index=1)
    fig, ax = plt.subplots()
    sample = cleaned[[xcol, ycol]].dropna().sample(n=min(2000, len(cleaned[[xcol, ycol]].dropna())), random_state=42)
    ax.scatter(sample[xcol], sample[ycol], alpha=0.4, s=10)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    st.pyplot(fig)
    plt.close(fig)
bar_col = st.selectbox('Bar chart column', cleaned.columns.tolist())
counts = cleaned[bar_col].astype(str).value_counts().head(15)
fig, ax = plt.subplots()
counts.plot.bar(ax=ax)
ax.set_title(f'Top values: {bar_col}')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
if len(numeric_cols) >= 2:
    corr = cleaned[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(min(10, max(5, len(corr) * 0.5)), min(9, max(4, len(corr) * 0.5))))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader('6. Machine Learning')
model_type = st.radio('Select model type', ['Classification', 'Regression'], horizontal=True)
target = st.selectbox('Select target column', cleaned.columns.tolist(), index=None,
                      placeholder='Choose the value to predict')
st.caption('Choose a category target for classification or a continuous numeric target for regression.')
st.warning('Important: remove columns that reveal the answer (target leakage) before relying on model results.')

if st.button('Train Model', type='primary'):
    if target is None:
        st.warning('Select a target column first.')
    else:
        try:
            with st.spinner('Preprocessing and training model...'):
                model, metrics, info, feature_data = train_dynamic_model(cleaned, target, model_type)
            st.session_state.model_bundle = {
                'model': model, 'metrics': metrics, 'info': info,
                'features': feature_data, 'target': target, 'model_type': model_type,
                'dataset_id': file_id,
            }
            st.success('Model trained successfully!')
        except Exception as exc:
            st.session_state.pop('model_bundle', None)
            st.error(f'Training failed: {exc}')

bundle = st.session_state.get('model_bundle')
if bundle and bundle['dataset_id'] == file_id and bundle['target'] == target and bundle['model_type'] == model_type:
    st.subheader('7. Model Evaluation')
    metrics = bundle['metrics']
    if model_type == 'Classification':
        cols = st.columns(4)
        for col, name in zip(cols, ('Accuracy', 'Precision', 'Recall', 'F1 Score')):
            col.metric(name, f'{metrics[name]:.2%}')
        st.write('Confusion Matrix')
        st.dataframe(pd.DataFrame(metrics['confusion_matrix'], index=metrics['labels'],
                                  columns=metrics['labels']))
    else:
        cols = st.columns(3)
        for col, name in zip(cols, ('MAE', 'RMSE', 'R2')):
            col.metric(name, f'{metrics[name]:.3f}')
    st.caption(f"Train rows: {bundle['info']['training_rows']:,}; test rows: {bundle['info']['testing_rows']:,}")
    if bundle['info']['excluded_columns']:
        st.info('Automatically excluded columns: ' + ', '.join(bundle['info']['excluded_columns']))

    st.subheader('8. Prediction')
    st.write('Enter values for the model input columns.')
    inputs = {}
    original = cleaned
    with st.form('predict_form'):
        # Collect original features; excluded columns are not needed.
        excluded = set(bundle['info']['excluded_columns']) | {target}
        for column in original.columns:
            if column in excluded:
                continue
            series = original[column]
            if pd.api.types.is_numeric_dtype(series):
                median = pd.to_numeric(series, errors='coerce').median()
                default = float(median) if pd.notna(median) else 0.0
                inputs[column] = st.number_input(str(column), value=default)
            elif column in bundle['info']['converted_date_columns']:
                first = series.dropna().astype(str)
                inputs[column] = st.text_input(str(column), value=first.iloc[0] if len(first) else '')
            else:
                choices = series.dropna().astype(str).unique().tolist()
                if len(choices) > 100:
                    choices = choices[:100]
                if choices:
                    inputs[column] = st.selectbox(str(column), choices)
                else:
                    inputs[column] = st.text_input(str(column))
        predict_clicked = st.form_submit_button('Predict')
    if predict_clicked:
        try:
            row = prepare_prediction_row(inputs, original, target, bundle['info']['feature_columns'])
            answer = bundle['model'].predict(row)[0]
            st.success(f'Predicted {target}: {answer}')
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')

    st.subheader('9. Download Model & Information')
    model_bytes = io.BytesIO()
    joblib.dump(bundle['model'], model_bytes)
    st.download_button('📥 Download Complete Model Pipeline (.pkl)', model_bytes.getvalue(),
                       f'{model_type.lower()}_pipeline.pkl', mime='application/octet-stream')
    st.download_button('📥 Download Model Information (.json)',
                       json.dumps(bundle['info'], indent=2),
                       f'{model_type.lower()}_model_info.json', mime='application/json')
    st.caption('Only load .pkl files you created or trust. Pickle files can execute code when loaded.')
else:
    st.info('Select a target and train the selected model to view metrics, predictions and downloads.')
