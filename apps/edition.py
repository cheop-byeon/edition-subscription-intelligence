"""Edition: the single subscription intelligence application."""
import json, os, sys
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
sys.path.insert(0,str(ROOT/'apps'))
from commercial import Commercial
from database import Store
from paywall import POLICY

st.set_page_config(page_title='Edition · Subscription Intelligence',page_icon='◈',layout='wide')
st.markdown('<style>[data-testid="stAppDeployButton"] {display:none;}</style>', unsafe_allow_html=True)
st.title('Edition · Subscription Intelligence')
st.caption('Local ML portfolio · Data preparation, churn modelling and MLOps · Independent of Schibsted')
db_path=os.environ.get('EDITION_DB',str(ROOT/'data/edition.sqlite3'))
app=Commercial(Store(db_path))
brand=st.sidebar.selectbox('Brand workspace',['all']+list(app.artifact()['brands']))
PAGES=['Overview','Subscribers & retention','Readers & campaigns','Paywall','Experiments','Assistant','Models & data','Research']
initial=st.query_params.get('page','Overview')
page=st.sidebar.radio('Workspace',PAGES,index=PAGES.index(initial) if initial in PAGES else 0)
st.sidebar.caption('One workspace · one shared database')
if st.session_state.get('last_brand')!=brand:
    st.session_state['conversation']=[]
    st.session_state.pop('paywall_result',None)
    st.session_state['last_brand']=brand
summary=app.summary(brand)

if page=='Overview':
    st.header('Subscription churn · local ML portfolio')
    st.write('A self-learning project for a data science job application: prepare real subscription data, train and evaluate a churn model, and apply its scores to retention audience analysis.')
    from kkbox import status as kkbox_status
    current=kkbox_status(db_path)['model']
    if current:
        metrics=current['artifact']['metrics']
        for column,(label,value) in zip(st.columns(3),[('Held-out subscribers',f"{metrics['rows']:,}"),('ROC AUC',f"{metrics['auc']:.3f}"),('Log loss',f"{metrics['logloss']:.4f}")]):
            column.metric(label,value)
        st.caption('KKBOX music subscription records · one historical cohort. These results do not measure newspaper paywalls or campaign effects.')
    else:
        st.info('Start under Models & data to import the original KKBOX files and train locally.')
    st.subheader('Learning workflow')
    st.write('Models & data: import → SQL features → train → evaluate → track with MLflow. Subscribers & retention: score held-out subscribers → inspect risk → save research audience drafts.')
    st.caption('Readers & campaigns, Paywall, Experiments and Assistant retain the earlier simulated commercial exercises. They do not use KKBOX as evidence for sales, paywall or campaign effectiveness.')
    st.subheader('Technology status')
    st.table(pd.DataFrame([{'Technology':'Python + SQL','Status':'Local training, serving and SQLite operations'},{'Technology':'MLflow','Status':'Local experiment tracking; serving activation remains in SQLite registry'},{'Technology':'Streamlit','Status':'This live analyst app'},{'Technology':'Snowflake','Status':'SQL, loader and export integration prepared; account not configured'}]))
elif page=='Subscribers & retention':
    from retention_view import render
    render(db_path)
elif page=='Paywall':
    st.header('Paywall simulation · access, offer and timing')
    st.caption('Illustrative exercise. KKBOX has no article exposures or paywall responses to train this policy.')
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
        st.write(result['reason'])
        with st.expander('Decision evidence'):st.json(result)
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
elif page=='Models & data':
    from kkbox_view import render as render_kkbox
    render_kkbox(db_path)
    st.divider()
    st.header('Simulated commercial model exercises')
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
if page=='Models & data':
    st.subheader('Data connections')
    st.info('No Snowflake account configured. Local SQLite remains operational. Nothing on this page connects to Snowflake automatically.')
    st.code('python backend/warehouse.py plan\npython backend/warehouse.py init --connection edition\npython backend/warehouse.py load --connection edition\npython backend/warehouse.py export --connection edition --dataset <printed-sha256>\npython backend/train_lifecycle.py --features-file data/training/snowflake_features.jsonl --mlflow',language='bash')
    st.caption('Once configured, the explicit commands use the official connector, sandbox SQL tables and feature view. Warehouse queries can consume credits. No account creation or queries have been executed.')
    for filename in ['snowflake_schema.sql','snowflake_features.sql']:
        with st.expander(filename):st.code((ROOT/'sql'/filename).read_text(),language='sql')

if page=='Readers & campaigns':
    st.header('Simulated readers and campaign exercises')
    cohort=st.selectbox('Lifecycle',['All readers','Prospects · sales','Subscribers · churn'])
    rows=app.readers(brand)
    if cohort!='All readers':rows=[r for r in rows if r['subscribed']==(cohort=='Subscribers · churn')]
    st.dataframe(pd.DataFrame([{'Reader':r['id'],'Brand':r['brand'],'Lifecycle':'Subscriber' if r['subscribed'] else 'Prospect','Subscribe in 30d':None if r['sales'] is None else r['sales']*100,'Cancel in 30d':None if r['churn'] is None else r['churn']*100,'Contactable':r['eligible']} for r in rows]),hide_index=True,column_config={k:st.column_config.NumberColumn(k,format='%.1f%%') for k in ['Subscribe in 30d','Cancel in 30d']})
    st.caption('Sales applies to prospects; churn applies to subscribers. Empty scores are not applicable. All commercial profiles are simulated.')
    with st.form('campaign'):
        name=st.text_input('Campaign name',max_chars=120)
        kind=st.selectbox('Objective',['sales','churn'])
        threshold=st.slider('Minimum propensity',0.0,1.0,.3,.01)
        saved=st.form_submit_button('Save campaign draft')
    if saved:
        try:
            result=app.campaign({'name':name,'kind':kind,'threshold':threshold,'brand':brand})
            st.success(f"Draft #{result['id']} saved with {result['members']} eligible readers. No messages sent.")
            summary=app.summary(brand)
        except ValueError as exc:st.error(str(exc))
    st.subheader('Saved drafts')
    for campaign in summary['campaigns']:
        st.write(f"**{campaign['name']}** · {campaign['kind']} · {campaign['members']} readers · {campaign['brand']}")
        # Previewing a page must not create an export audit event. Generate only on click.
        if st.button('Prepare audience CSV',key=f"prepare-{campaign['id']}"):
            st.session_state[f"csv-{campaign['id']}"]=app.export(campaign['id'])
        csv=st.session_state.get(f"csv-{campaign['id']}")
        if csv:st.download_button('Download audience',csv,file_name=f"edition-audience-{campaign['id']}.csv",mime='text/csv',key=f"download-{campaign['id']}")

if page=='Paywall':
    result=st.session_state.get('paywall_result')
    if result and (brand=='all' or result['brand']==brand):
        st.divider();st.subheader('Reader preview')
        st.caption(result['brand']+' · Fictional article · '+result['reader_id'])
        st.markdown('### A city finds new ways to keep its neighbourhoods connected')
        st.write('Independent reporting helps readers understand the decisions shaping their community.')
        if result['offer']:
            offer=result['offer'];st.info(f"{'Try digital access' if offer['id']=='trial_offer' else 'Digital subscription'} · {offer['price']} {offer['currency']}. {offer['terms']}")
        elif result['content_access']=='full':st.success('Full article access granted in this simulated scenario.')
        else:st.info('Subscriber access required. Only the teaser is available; deferring an offer does not unlock the article.')
        if st.session_state.get('recorded_decision')==result['decision_id']:
            st.success('Response saved. See Experiments in the sidebar for the recorded outcome.')
        else:
            accepted=False
            if result['offer']:accepted=st.button('Simulate accepting offer',type='primary')
            declined=st.button('Record simulated no-purchase response')
            if accepted or declined:
                try:
                    app.outcome({'decision_id':result['decision_id'],'converted':accepted,'revenue':result['offer']['price'] if accepted else 0})
                    st.session_state['recorded_decision']=result['decision_id'];st.rerun()
                except ValueError as exc:st.error(str(exc))
        st.caption('No actual subscription, payment or account change occurs.')

if page=='Assistant':
    st.header('Ask Edition')
    st.caption('Local Ollama assistant. Uses aggregate data for this brand; cannot send campaigns or change models.')
    for message in st.session_state.get('conversation',[]):
        with st.chat_message(message['role']):st.write(message['content'])
    question=st.chat_input('Ask about audiences, model results, or draft a campaign brief',max_chars=2000)
    if question:
        history=st.session_state.get('conversation',[])[-6:]
        with st.chat_message('user'):st.write(question)
        with st.chat_message('assistant'):
            with st.spinner('Running local model…'):
                result=app.chat({'question':question,'brand':brand,'history':history})
            st.write(result['answer']);st.caption('Model: '+result['model']+' · Source: current workspace summary')
        st.session_state['conversation']=(history+[{'role':'user','content':question},{'role':'assistant','content':result['answer'][:2000]}])[-6:]

if page=='Models & data':
    st.subheader('Reference benchmark · KKBOX-derived subscription labels')
    st.caption('This external benchmark is preserved for learning. It does not drive the commercial paywall or retention scores above.')
    with st.expander('Inspect benchmark and run its trained model'):
        from external_engine import score as benchmark_score
        artifact=json.loads((ROOT/'artifacts/external_model.json').read_text())
        metrics=artifact['metrics']
        c1,c2,c3=st.columns(3);c1.metric('Test ROC AUC',f"{metrics['auc']:.3f}");c2.metric('Average precision',f"{metrics['average_precision']:.3f}");c3.metric('Test rows',f"{artifact['splits']['test']['n']:,}")
        st.write('Published KKBOX-derived data with synthetic subscription labels. These metrics measure that benchmark, not real newspaper subscriptions or offer effects.')
        profiles=app.store.profiles()
        if not profiles:profiles=artifact['profiles']
        selected=st.selectbox('Benchmark input',[p['id'] for p in profiles])
        record=next(p for p in profiles if p['id']==selected)
        st.json(record['values'],expanded=False)
        if st.button('Run reference model'):
            result=benchmark_score(record['values'],artifact)
            id=app.store.record(record['values'],result)
            st.metric('Synthetic-label probability',f"{result['probability']:.2%}")
            st.caption(f"Saved prediction #{id}. Imported examples may include training periods; frozen holdout metrics above are the evaluation evidence.")
        st.link_button('Dataset source',artifact['source_url'])
    with st.expander('Register a trained commercial candidate'):
        upload=st.file_uploader('Model artifact JSON',type=['json'])
        if upload and st.button('Register candidate'):
            try:
                if upload.size>1000000:raise ValueError('Artifact exceeds 1 MB.')
                result=app.register({'artifact':json.loads(upload.getvalue())});st.success(f"Candidate #{result['id']} registered; not activated. Refresh to compare it.")
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))

if page=='Research':
    st.header('Research and implementation evidence')
    st.write('One application for subscription intelligence. Commercial scores, the external reference benchmark and the local assistant have distinct evidence and data sources.')
    for source in json.loads((ROOT/'research/sources.json').read_text()):
        with st.expander(f"{source['year']} · {source['title']}"):
            st.write(source['summary']);st.caption(source.get('status',''))
            st.link_button('Paper',source['paper'])
            if source.get('code'):st.link_button('Code / method documentation',source['code'])
    with st.expander('How Edition maps to the JD'):
        st.markdown((ROOT/'research/JD_ALIGNMENT.md').read_text())
