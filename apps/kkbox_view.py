"""Real-data churn workspace within the single Edition Streamlit app."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from kkbox import ROOT, FEATURES, SOURCE_URL, holdout_rows, job_status, predict, save_prediction, start_job, status


def render(db_path):
    db_path = Path(db_path)
    work_dir = db_path.parent / 'kkbox'
    st.header('Subscription churn · KKBOX')
    st.write('Predict non-renewal using the original KKBOX music-subscription records. Billing features are calculated before the cohort cutoff; churn labels come from the competition.')
    st.caption('KKBOX is a music service. Brand filters apply to the simulated commercial workspace, not these subscribers.')
    state = status(db_path)
    current = state['model']
    if current:
        artifact = current['artifact']
        if artifact['source_kind'] == 'synthetic_test_fixture':
            st.warning('Integration-test fixture only. These are synthetic test records, not real KKBOX results.')
        else:
            st.success('Local churn model trained on imported KKBOX competition files.')
        metrics = artifact['metrics']
        columns = st.columns(4)
        for column, (title, value) in zip(columns, [('Test subscribers', f"{metrics['rows']:,}"), ('ROC AUC', f"{metrics['auc']:.3f}"), ('Log loss', f"{metrics['logloss']:.4f}"), ('Observed churn', f"{metrics['prevalence']:.1%}")]):
            column.metric(title, value)
        st.caption('Model: ' + artifact['selected_family'] + ' · Feature cutoff: ' + artifact['snapshot'] + ' · Billing history only')
        st.info('Evaluation uses untouched subscribers from one historical cohort. It does not measure future-month performance, newspaper conversions or the effects of retention offers.')
        comparison = [{'Model': 'Constant training prevalence', **{k: artifact['constant_baseline'][k] for k in ['auc', 'average_precision', 'logloss', 'brier']}}]
        comparison += [{'Model': name + ' · uncalibrated', **{k: result[k] for k in ['auc', 'average_precision', 'logloss', 'brier']}} for name, result in artifact['uncalibrated_models'].items()]
        comparison += [{'Model': 'Selected model · calibrated', **{k: metrics[k] for k in ['auc', 'average_precision', 'logloss', 'brier']}}]
        st.subheader('Held-out model comparison')
        st.dataframe(pd.DataFrame(comparison), hide_index=True)
        st.caption('Lower log loss and Brier score are better; higher ROC AUC and average precision are better. Selection used a separate partition, before this test evaluation.')
        st.subheader('Calibration')
        st.line_chart(pd.DataFrame(metrics['calibration']).set_index('predicted')[['observed']])
        st.caption('The horizontal axis is mean predicted risk; the vertical axis is observed churn in each probability bin. Ideally they agree.')
        st.subheader('Run the local model')
        records = holdout_rows(db_path, artifact['dataset_id'])
        by_id = {r['msno']: r for r in records}
        selected = st.selectbox('Held-out subscriber', list(by_id), format_func=lambda user: 'KKBOX · ' + user[:12])
        record = by_id[selected]
        st.dataframe(pd.DataFrame([record['values']]), hide_index=True)
        probability = predict(record['values'], artifact)
        st.metric('Predicted non-renewal risk', f'{probability:.1%}')
        st.caption('Observed competition outcome: ' + ('churned' if record['is_churn'] else 'did not churn') + '. This historical outcome is displayed for inspection and is not a model input.')
        if st.button('Save this churn prediction', key='kkbox-save-prediction'):
            result = save_prediction(db_path, current['id'], selected)
            st.success(f"Prediction #{result['id']} saved with its model and dataset IDs.")
        with st.expander('Training evidence and limitations'):
            st.write(artifact['selection_rule'])
            st.dataframe(pd.DataFrame(artifact['partitions']).T)
            st.write(artifact['limitations'])
            st.json(artifact['manifest'])
            if artifact.get('mlflow_run_id'):
                st.link_button('Inspect this MLflow run', 'http://127.0.0.1:5001/#/experiments/' + artifact['mlflow_experiment_id'] + '/runs/' + artifact['mlflow_run_id'])
                st.code('MLflow run ID: ' + artifact['mlflow_run_id'])
    else:
        st.info('Waiting for the original KKBOX files. No real-data churn model has been trained yet.')
        if state['datasets']:
            st.caption('An import is present; training must finish successfully before predictions are available.')
        st.write('This workflow is separate from the older KKBOX-derived benchmark with synthetic subscription labels. That benchmark cannot substitute for the original churn records.')

    st.subheader('Import data and train locally')
    st.link_button('Get the official KKBOX files on Kaggle', SOURCE_URL)
    st.write('Sign in to Kaggle and accept the competition rules yourself. Download and extract **train_v2.csv** and **transactions.csv** into one local folder. **transactions_v2.csv** is optional. If only train.csv is supplied, the pipeline uses its earlier cohort cutoff.')
    st.caption('CSV, CSV.GZ and single-CSV ZIP inputs are supported. Extract .7z archives first. Listening logs and member demographics are not required for this billing-only baseline.')
    with st.form('kkbox-import'):
        folder = st.text_input('Folder containing KKBOX CSV files', placeholder='/Users/your-name/Downloads/kkbox')
        limit = st.number_input('Maximum subscribers', min_value=500, max_value=500000, value=50000, step=500)
        official = st.checkbox('These are original KKBOX competition files obtained under the Kaggle rules')
        submitted = st.form_submit_button('Import and train locally')
    if submitted:
        if not official:
            st.error('Confirm the source of the files before starting a real-data run.')
        else:
            try:
                pid = start_job(folder, db_path, work_dir, limit)
                st.session_state['kkbox-started-pid'] = pid
                st.success('Import and training started locally. Refresh the run status below to see progress.')
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    job = job_status(work_dir)
    if job:
        st.write('**Run status:** ' + job['status'])
        st.caption(job.get('message', 'Starting…'))
        if job['status'] == 'failed':
            st.error(job['message'])
        if job['status'] == 'complete' and (not current or current['id'] != job.get('model_id')):
            st.success('Training finished. Refresh to load its held-out results.')
    if st.button('Refresh run status', key='kkbox-refresh'):
        st.rerun()
    with st.expander('Technology stack and command-line workflow'):
        st.table(pd.DataFrame([
            {'Tool': 'Python', 'Use': 'Streaming import, baseline training, calibration and portable local prediction'},
            {'Tool': 'SQL / SQLite', 'Use': 'Point-in-time billing features, dataset lineage, model and prediction records'},
            {'Tool': 'MLflow', 'Use': 'Training run, model comparison metrics, dataset fingerprints and model artifacts'},
            {'Tool': 'Streamlit', 'Use': 'This one application: import, evaluation and inference'},
            {'Tool': 'Snowflake', 'Use': 'Prepared sandbox tables/view, upload and SQL parity check; no account connected'}]))
        st.code('python -m pip install -r requirements-kkbox.txt\npython backend/train_kkbox.py --input-dir /absolute/path/to/kkbox --limit 50000\nmlflow server --backend-store-uri sqlite:///data/mlflow.sqlite3 --host 127.0.0.1 --port 5001', language='bash')
        st.caption('Training tracks to local MLflow by default. The model is exported as numeric JSON; serving does not load pickle files or need a GPU.')
    with st.expander('Snowflake integration · prepared, not connected'):
        st.write('The same imported records can be uploaded to an explicitly configured sandbox. The feature view mirrors local SQL, and verify compares every feature, label and split before claiming parity. No Snowflake query runs automatically.')
        st.code('python backend/kkbox_warehouse.py plan\npython backend/kkbox_warehouse.py init --connection edition\npython backend/kkbox_warehouse.py load --connection edition --dataset <dataset-id>\npython backend/kkbox_warehouse.py verify --connection edition --dataset <dataset-id>', language='bash')
        for filename in ['kkbox_snowflake_schema.sql', 'kkbox_snowflake_features.sql']:
            with st.expander(filename):
                st.code((ROOT / 'sql' / filename).read_text(), language='sql')
