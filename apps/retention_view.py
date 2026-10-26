"""Portfolio demonstration of churn scoring feeding retention analysis."""
import pandas as pd
import streamlit as st
from retention import audience, drafts, save_draft
from kkbox import status


@st.cache_data(show_spinner='Scoring held-out subscribers locally…')
def scored(db_path, model_id):
    return audience(db_path)


def render(db_path):
    st.header('Subscribers & retention')
    st.write('Explore historical subscribers → inspect churn risk → save a retention audience draft.')
    current = status(db_path)['model']
    if not current:
        st.info('Import data and train the churn model under Models & data first.')
        return
    model, rows = scored(str(db_path), current['id'])
    if model['artifact']['source_kind'] == 'synthetic_test_fixture':
        st.warning('Integration-test fixture only: synthetic test records.')
    st.caption('Source: KKBOX music subscriptions · held-out historical cohort · brand filter does not apply. No contact permissions or campaign responses are available.')
    threshold = st.slider('Minimum churn risk', 0.0, 1.0, .3, .01)
    selected = [r for r in rows if r['risk'] >= threshold]
    left, right = st.columns(2)
    left.metric('Scored subscribers', f'{len(rows):,}')
    right.metric('Research audience', f'{len(selected):,}')
    st.dataframe(pd.DataFrame([{'Subscriber': 'KKBOX · '+r['subscriber'][:12], 'Predicted non-renewal risk': r['risk']*100} for r in selected]), hide_index=True,
                 column_config={'Predicted non-renewal risk': st.column_config.NumberColumn(format='%.1f%%')})
    st.info('High churn risk does not establish who benefits from an offer. These are research audiences, not contactable marketing lists. Campaign effectiveness needs treatment and outcome data.')
    with st.form('retention-draft'):
        name = st.text_input('Retention draft name', max_chars=120)
        save = st.form_submit_button('Save retention draft')
    if save:
        try:
            draft_id = save_draft(db_path, name, threshold)
            st.success(f'Draft #{draft_id} saved locally with model, dataset and scored audience. No messages sent.')
        except ValueError as exc:
            st.error(str(exc))
    st.subheader('Saved research drafts')
    for draft in drafts(db_path):
        st.write(f"**{draft['name']}** · {len(draft['members']):,} subscribers · risk ≥ {draft['threshold']:.0%} · model #{draft['model_id']}")
    st.caption('Inspect model comparison, calibration, training lineage and import tools under Models & data. No campaign uplift or live retention outcomes are claimed.')
