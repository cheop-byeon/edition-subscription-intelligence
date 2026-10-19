"""Safe numeric tree inference; no pickle or ML dependency is needed at serving time."""
import math

def validate_profile(values,model):
    if not isinstance(values,dict):raise ValueError('Expected a feature object.')
    names=model['feature_names']
    if set(values)!=set(names):raise ValueError('Feature names must match the benchmark model schema.')
    for k,v in values.items():
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)):
            raise ValueError(f'{k} must be a finite number or null.')
    return values

def predict_export(values,model):
    validate_profile(values,model);x=[];pre=model['preprocess']
    for i,k in enumerate(model['feature_names']):
        v=values[k]
        if v is None or v<0:v=pre['median'][i]
        x.append(math.log1p(min(v,pre['cap'][i])))
    raw=model['base']
    for nodes in model['trees']:
        j=0
        while not nodes[j]['leaf']:
            node=nodes[j];j=node['left'] if x[node['feature']]<=node['threshold'] else node['right']
        raw+=nodes[j]['value']
    z=model['calibration']['slope']*raw+model['calibration']['intercept']
    return 1/(1+math.exp(-max(-40,min(40,z))))

def score(values,artifact):
    p=predict_export(values,artifact['model'])
    return {'probability':p,'model_version':artifact['version'],'target':artifact['target'],'data_kind':artifact['data_kind'],'note':'Probability of the provided synthetic subscription label. No offer effect or real newspaper conversion is inferred.'}
