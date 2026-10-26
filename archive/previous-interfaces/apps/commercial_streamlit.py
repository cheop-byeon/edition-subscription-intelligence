"""JD-aligned analyst workbench. Uses the same serving code and SQLite registry."""
import json, os, sys
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from commercial import Commercial
from database import Store
from paywall import POLICY

st.set_page_config(page_title='Edition · Paywall analytics',page_icon='◈',layout='wide')
st.title('Edition · Subscription & paywall analytics')
st.caption('Self-learning project · Simulated data · Independent of Schibsted')
app=Commercial(Store(os.environ.get('EDITION_DB',str(ROOT/'data/edition.sqlite3'))))
brand=st.sidebar.selectbox('Brand workspace',['all']+list(app.artifact()['brands']))
page=st.sidebar.radio('Analysis',['Overview','Paywall lab','Experiments','Model operations','Warehouse'])
st.sidebar.link_button('Reader journey','http://127.0.0.1:4173/article')
st.sidebar.link_button('Existing workspace','http://127.0.0.1:4173/commercial')
summary=app.summary(brand)

if page=='Overview':
    st.header('Who should see an offer, and when?')
    st.write('The paywall and offer engine selects eligible subscription offers and their timing. Sales propensity helps with acquisition; churn helps with retention. Article access is an independent entitlement/editorial decision.')
    columns=st.columns(4)
    for column,(label,key) in zip(columns,[('Readers','readers'),('Contactable','eligible'),('Sales audience ≥20%','sales_audience'),('Churn audience ≥30%','churn_audience')]):column.metric(label,summary[key])
    rows=app.readers(brand)
    frame=pd.DataFrame([{'Reader':r['id'],'Brand':r['brand'],'Lifecycle':'Subscriber' if r['subscribed'] else 'Prospect','Subscribe within 30d':None if r['sales'] is None else r['sales']*100,'Cancel within 30d':None if r['churn'] is None else r['churn']*100,'Contactable':r['eligible']} for r in rows])
    st.dataframe(frame,hide_index=True,column_config={name:st.column_config.NumberColumn(name,format='%.1f%%') for name in ['Subscribe within 30d','Cancel within 30d']})
    st.caption('Empty probabilities are not applicable. Scores are simulated-data model predictions, not campaign uplift.')
    st.subheader('The stack in this learning project')
    st.table(pd.DataFrame([{'Technology':'Python + SQL','Status':'Local training, serving and SQLite operations'},{'Technology':'MLflow','Status':'Local experiment tracking; serving activation remains in SQLite registry'},{'Technology':'Streamlit','Status':'This live analyst app'},{'Technology':'Snowflake','Status':'SQL, loader and export integration prepared; account not configured'}]))
elif page=='Paywall lab':
    st.header('Access, offer and timing')
    rows=app.readers(brand);by_id={r['id']:r for r in rows}
    with st.form('paywall'):
        reader=st.selectbox('Reader',list(by_id),format_func=lambda id:id+' · '+('Subscriber' if by_id[id]['subscribed'] else 'Prospect'))
        articles=st.number_input('Articles read this session',min_value=0,max_value=100,value=2)
        dismissals=st.number_input('Prior offer dismissals',min_value=0,max_value=100,value=0)
        premium=st.checkbox('Paid article',value=True);public=st.checkbox('Public-interest editorial exemption')
        submitted=st.form_submit_button('Evaluate decision')
    if submitted:
        result=app.paywall({'reader_id':reader,'premium':premium,'public_interest':public,'articles_in_session':articles,'dismissals':dismissals})
        st.session_state['paywall_result']=result
    result=st.session_state.get('paywall_result')
    if result and (brand=='all' or result['brand']==brand):
        st.metric('Article access',result['content_access']);st.metric('Commercial action',result['action'])
        st.write(result['reason']);st.json(result)
    st.info('Deferring an offer never grants paid-content access. No personalisation consent uses a contextual standard offer. Editorial exemptions and subscriptions can grant access.')
    with st.expander('Inspect illustrative brand policies'):st.json(POLICY)
elif page=='Experiments':
    st.header('What do the recorded outcomes show?')
    st.caption('Current experiment: '+POLICY['version']+'. Earlier policy versions are not pooled with this experiment.')
    results=pd.DataFrame(summary['experiment'])
    results['unobserved_outcomes']=results.decisions-results.recorded_outcomes
    results['conversion_per_recorded_outcome']=results.conversions/results.recorded_outcomes.replace(0,float('nan'))
    st.dataframe(results,hide_index=True)
    st.warning('These are manually simulated outcomes. Missing follow-up, repeated decisions per reader and non-random participation prevent treating these counts as a causal revenue estimate. Assignment is stable by reader, so a future experiment needs reader-level analysis and a fixed outcome window.')
    st.bar_chart(results.set_index('arm')[['decisions','recorded_outcomes','conversions']])
elif page=='Model operations':
    st.header('Model evidence and lifecycle')
    st.dataframe(pd.DataFrame([{'id':r['id'],'version':r['version'],'active':r['active'],'sales_auc':r['metrics']['sales']['auc'],'churn_auc':r['metrics']['churn']['auc']} for r in summary['registry']]),hide_index=True)
    candidates={r['id']:r for r in summary['registry'] if not r['active']}
    if candidates:
        chosen=st.selectbox('Available model version',list(candidates),format_func=lambda id:candidates[id]['version'])
        if st.button('Activate selected version / roll back'):
            app.activate(chosen);st.success('Model activation saved to the shared serving registry.');st.rerun()
    st.link_button('Open local MLflow runs','http://127.0.0.1:5001')
    st.caption('MLflow must be running locally. Activation uses the same registry as the API.')
    st.json(summary['lineage'])
    st.dataframe(pd.DataFrame(summary['monitoring']['features']),hide_index=True)
    st.caption(summary['monitoring']['type'])
elif page=='Warehouse':
    st.header('Snowflake integration')
    st.info('No Snowflake account configured. Local SQLite remains operational. Nothing on this page connects to Snowflake automatically.')
    st.code('python backend/warehouse.py plan\npython backend/warehouse.py init --connection edition\npython backend/warehouse.py load --connection edition\npython backend/warehouse.py export --connection edition --dataset <printed-sha256>\npython backend/train_lifecycle.py --features-file data/training/snowflake_features.jsonl --mlflow',language='bash')
    st.caption('Once configured, the explicit commands use the official connector, sandbox SQL tables and feature view. Warehouse queries can consume credits. No account creation or queries have been executed.')
    for filename in ['snowflake_schema.sql','snowflake_features.sql']:
        with st.expander(filename):st.code((ROOT/'sql'/filename).read_text(),language='sql')
