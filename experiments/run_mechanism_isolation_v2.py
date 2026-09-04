#!/usr/bin/env python
"""Identifiable 2x2 negative-control experiment for episode memorization.

Factor A controls a recognizable static episode fingerprint.  Factor B controls
whether labels share one event time and event-dependent truncation.  When B is
off, episodes have fixed length and independent row-level Bernoulli outcomes;
that cell is a matched-prevalence negative control, not an event forecaster.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_generation import generate_panel_data, get_feature_columns, prepare_modeling_data
from src.evaluation import evaluate_grouped_cv, evaluate_random_cv, compute_pooled_oof_metrics

def make_control_panel(fingerprint: bool, event_time: bool, seed: int, n_episodes: int=30, length: int=20) -> pd.DataFrame:
    """Return a panel with explicit factor semantics and auditable metadata."""
    rng=np.random.default_rng(seed)
    if event_time:
        df=generate_panel_data(n_episodes=n_episodes,T_max=60,alpha_std=.5 if fingerprint else 0.,ar_coef=.7,noise_std=.3,hazard_coef=.15,base_hazard=-3.,horizon=14,seed=seed)
        if not fingerprint: df=df.drop(columns=['X_5'])
        df['outcome_mode']='shared_event_time'; df['fixed_length']=False
        return df
    # Negative control: no common T_e, no event truncation, and independent rows.
    episode=np.repeat(np.arange(n_episodes),length); n=len(episode)
    data={'episode_id':episode,'t':np.tile(np.arange(length),n_episodes)}
    for j in range(1,5): data[f'X_{j}']=rng.normal(size=n)
    if fingerprint: data['X_5']=np.repeat(rng.normal(0,.5,n_episodes),length)
    # Row outcomes are conditionally independent and use only row-level X_1.
    p=expit(-.08+.15*data['X_1']); data['Y']=rng.binomial(1,p,n)
    df=pd.DataFrame(data); df['T_e']=np.nan; df['C_e']=float(length); df['event_observed']=0
    df['at_risk']=1; df['horizon_observed']=1; df['row_id']=np.arange(n); df['outcome_mode']='independent_row'; df['fixed_length']=True
    return df

def interval(x):
    x=np.asarray(x); m=x.mean(); s=x.std(ddof=1)/np.sqrt(len(x)); q=stats.t.ppf(.975,len(x)-1); return m-q*s,m+q*s
def main():
 p=argparse.ArgumentParser(); p.add_argument('--n-replicates',type=int,default=30); p.add_argument('--start-seed',type=int,default=42); a=p.parse_args()
 out=Path('results/camera_ready/mechanism_isolation_v2.csv'); out.parent.mkdir(parents=True,exist_ok=True); rows=[]
 for fp in (False,True):
  for et in (False,True):
   for rep in range(a.n_replicates):
    seed=a.start_seed+rep; started=time.perf_counter()
    df=make_control_panel(fp,et,seed); X,y,g=prepare_modeling_data(df,get_feature_columns(df))
    for split,res in [('grouped',evaluate_grouped_cv(X,y,g,n_splits=5)),('row_wise',evaluate_random_cv(X,y,n_splits=5,seed=seed,groups=g))]:
     rows.append(dict(replicate=rep,seed=seed,episode_fingerprint=fp,shared_event_time=et,split_method=split,model='boosted_trees',event_prevalence=y.mean(),n_eligible=len(y),auroc=compute_pooled_oof_metrics(res)['auc'],status='ok',runtime_seconds=time.perf_counter()-started))
 pd.DataFrame(rows).to_csv(out,index=False)
 x=pd.DataFrame(rows).pivot(index=['replicate','seed','episode_fingerprint','shared_event_time'],columns='split_method',values='auroc').reset_index(); x['delta_cv']=x.row_wise-x.grouped
 prev=pd.DataFrame(rows).groupby(['episode_fingerprint','shared_event_time']).event_prevalence.mean()
 summary=[]
 for (fp,et),z in x.groupby(['episode_fingerprint','shared_event_time']):
  lo,hi=interval(z.delta_cv); summary.append(dict(episode_fingerprint=fp,shared_event_time=et,successful_replicates=len(z),grouped_auroc_mean=z.grouped.mean(),row_wise_auroc_mean=z.row_wise.mean(),delta_cv_mean=z.delta_cv.mean(),delta_cv_ci_lower=lo,delta_cv_ci_upper=hi,event_prevalence_mean=prev.loc[(fp,et)]))
 s=pd.DataFrame(summary); s.to_csv('results/camera_ready/mechanism_isolation_v2_summary.csv',index=False)
 lines=['% Generated; do not edit manually.','\\begin{tabular}{llccc}','\\toprule','Fingerprint & Shared event time & Grouped & Row-wise & $\\Delta_{\\rm CV}$ (95\\% CI) \\\\','\\midrule']
 for r in s.itertuples(): lines.append(f"{'on' if r.episode_fingerprint else 'off'} & {'on' if r.shared_event_time else 'off'} & {r.grouped_auroc_mean:.3f} & {r.row_wise_auroc_mean:.3f} & {r.delta_cv_mean:.3f} [{r.delta_cv_ci_lower:.3f}, {r.delta_cv_ci_upper:.3f}] \\\\")
 lines += ['\\bottomrule','\\end{tabular}']; Path('generated/mechanism_isolation_v2_table.tex').write_text('\n'.join(lines)+'\n')
 print(s.to_string(index=False))
if __name__=='__main__': main()
