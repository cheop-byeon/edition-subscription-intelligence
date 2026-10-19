export function predictExternal(values,m){
 if(!values||typeof values!=='object'||Array.isArray(values)||Object.keys(values).length!==m.feature_names.length||m.feature_names.some(k=>!Object.hasOwn(values,k)))throw Error('Feature names must match the benchmark model schema.');
 const x=m.feature_names.map((k,i)=>{let v=values[k];if(v!==null&&(typeof v!=='number'||!Number.isFinite(v)))throw Error(`${k} must be a finite number or null.`);if(v===null||v<0)v=m.preprocess.median[i];return Math.log1p(Math.min(v,m.preprocess.cap[i]))});
 let raw=m.base;for(const nodes of m.trees){let j=0;while(!nodes[j].leaf){const n=nodes[j];j=x[n.feature]<=n.threshold?n.left:n.right}raw+=nodes[j].value}
 const z=m.calibration.slope*raw+m.calibration.intercept;return 1/(1+Math.exp(-Math.max(-40,Math.min(40,z))))
}
export function scoreExternal(values,a){return {probability:predictExternal(values,a.model),model_version:a.version,target:a.target,data_kind:a.data_kind,note:'Probability of the provided synthetic subscription label. No offer effect or real newspaper conversion is inferred.'}}
