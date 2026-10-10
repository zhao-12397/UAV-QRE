#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7 verification, classified by ACTUAL KIND rather than a fixed 40/105 split (Q13).

Four categories, reported with real counts measured at run time:
  analytic        : closed-form references, invariants, geometry identities
  convergence     : numerical convergence / refinement / MC stability
  self_consistency: the code reproduces its own declared arithmetic
  simulated_input : controlled synthetic inputs (mock state jumps, patched solvers)

IMPORTANT: passing all checks is NOT a claim that the model has been externally
validated. It only means the implementation matches its own definitions and the
stated analytic references.
"""
import copy,json,math,csv
from pathlib import Path
import numpy as np
from unittest.mock import patch
import simulate_fc30 as m

def run():
    root=Path(__file__).parent
    c=json.loads((root/'params_fc30_v3.json').read_text(encoding='utf-8'))
    c['implementation']=json.loads((root/'model_supplement.json').read_text(encoding='utf-8'))
    Z=np.random.default_rng(20261007).standard_normal((1024,3))
    checks=[]
    def check(name,cond,kind):
        if not cond:raise AssertionError(f'[{kind}] {name}')
        checks.append({'check':name,'kind':kind,'passed':True})

    # ---------- analytic ----------
    s=m.scenario(c);tc,dc=m.cpa(s['pB']-s['pA'],s['vB']-s['vA'])
    check('TCPA=30, DCPA=0 analytic head-on case',abs(tc-30)<1e-10 and dc<1e-10,'analytic')
    check('gate false at t=0; true at t=10 for H=20',not m.gate(c,s)[0] and m.gate(c,m.at_time(s,10))[0],'analytic')
    p,h=m.advance_pose(np.zeros(3),0.,15.,-math.pi/18,3.);R=15/(math.pi/18)
    check('exact 30 degree right arc',abs(h+math.pi/6)<1e-12 and np.allclose(p,[R/2,-R*(1-math.sqrt(3)/2),0.]),'analytic')
    rng=np.random.default_rng(17);maxerr=0.
    for i in range(100):
        pp=np.r_[rng.normal(size=2)*20,0.];qq=np.r_[rng.normal(size=2)*20,0.]
        hh=rng.uniform(-np.pi,np.pi);hq=rng.uniform(-np.pi,np.pi);R=rng.uniform(1,20)
        word,ll=m.dubins(pp,hh,qq,hq,R);pw,hw=pp.copy(),hh
        for ch,l in zip(word,ll):pw,hw=m.advance_pose(pw,hw,1.,1/R if ch=='L' else -1/R if ch=='R' else 0.,l*R)
        maxerr=max(maxerr,float(np.linalg.norm(pw-qq)))
        check(f'Dubins endpoint analytic case {i+1}',np.linalg.norm(pw-qq)<1e-7 and abs(m.wrap(hw-hq))<1e-7,'analytic')
    check('segment collision not missed by coarse samples',m.segment_min_q(np.array([[-10,0,0],[10,0,0]]),np.eye(3),np.ones(3))==0.,'analytic')
    check('two metre vertical separation outside collision layer',not m.gate(c,m.at_time(m.scenario(c,z=2.),10))[0],'analytic')

    ev=m.outcomes(c,m.at_time(s,20),Z)
    cc=copy.deepcopy(c);cc['qre']['lambda_A']['value']=0.;cc['qre']['lambda_B']['value']=0.
    qr,roots,FA,FB=m.qre(cc,ev)
    check('lambda=0 gives independent uniform choices',qr['pA_Y']==.5 and qr['pB_Y']==.5 and qr['pNN']==.25,'analytic')
    qr0,roots0,_,_=m.qre(c,ev)
    check('symmetric inputs produce symmetric QRE',abs(qr0['pA_Y']-qr0['pB_Y'])<1e-9,'analytic')
    check('true fixed-point residual',qr0['residual']<1e-9,'analytic')
    st=m.at_time(s,28.);r=m.trajectory(c,st,'N,N')
    check('zero-error nominal overlap has Pcol=1',m.probability(c,r,0.,0.,Z)[0]==1.,'analytic')
    st=m.at_time(m.scenario(c,z=2),28.);r=m.trajectory(c,st,'N,N')
    check('zero-error height separated trajectory has Pcol=0',m.probability(c,r,0.,0.,Z)[0]==0.,'analytic')
    e=m.outcomes(c,m.at_time(s,27.),Z,True)
    check('asymmetric position input reaches distinct safety payoffs',e['A']['N,N']['Pcol_max']>e['B']['N,N']['Pcol_max'],'analytic')
    samples=[{'pA_N':.1,'pB_N':.1},{'pA_N':.9,'pB_N':.9}]
    ma,mb,n=m.individual_mean_factor(samples)
    check('individual mean product is .25, not joint mean .41',abs(ma*mb-.25)<1e-12 and abs(np.mean([q['pA_N']*q['pB_N'] for q in samples])-.41)<1e-12,'analytic')
    observed=[{'pA_N':.1,'pB_N':.9},{'pA_N':.3,'pB_N':.7}]
    forecast=[{'pA_N':.6,'pB_N':.4},{'pA_N':.8,'pB_N':.2},{'pA_N':1.,'pB_N':0.}]
    f=m.stage_individual_means(observed,forecast)
    check('unequal segment counts give full-region .56 and .44',abs(f['A_mean_N']-.56)<1e-12 and abs(f['B_mean_N']-.44)<1e-12 and f['sample_count']==5,'analytic')
    check('history-only and forecast-only stages are defined',m.stage_individual_means(observed,[])['forecast_sample_count']==0 and m.stage_individual_means([],forecast)['observed_sample_count']==0,'analytic')
    growing=[{'pA_N':x,'pB_N':x} for x in [.1,.3,.8]]
    check('shrinking remaining window changes mean',np.allclose([m.individual_mean_factor(growing[j:])[0] for j in range(3)],[.4,.55,.8]),'analytic')
    decreasing=[{'pA_N':x,'pB_N':x} for x in [.8,.3,.1]]
    check('remaining window does not imply universal risk increase',np.allclose([m.individual_mean_factor(decreasing[j:])[0] for j in range(3)],[.4,.2,.1]),'analytic')
    evs=[{'time_s':12.,'action':'Y','executed':False,'effective':False},
         {'time_s':15.,'action':'Y','executed':True,'effective':False},
         {'time_s':20.,'action':'Y','executed':True,'effective':True}]
    check('intent alone does not terminate branch',m.history_indicator(evs,12.)==1,'analytic')
    check('ineffective executed action does not terminate branch',m.history_indicator(evs,15.)==1,'analytic')
    check('confirmed effective Y terminates and remains absorbing',m.history_indicator(evs,20.)==0 and m.history_indicator(evs,22.)==0,'analytic')

    # ---------- Q1: safety loss forms ----------
    pref=float(m.value(c,'collision_probability','probability_reference'))
    check('saturating form clips at Pref',abs(m.safety_loss(0.5*pref,'saturating',pref)-.5)<1e-12 and m.safety_loss(5*pref,'saturating',pref)==1.,'analytic')
    check('continuous form never clips',m.safety_loss(50*pref,'continuous',pref)<1.,'analytic')
    check('continuous and saturating agree at Pcol=0',abs(m.safety_loss(0.,'continuous',pref))<1e-15,'analytic')

    # ---------- Q5: per-aircraft delay ----------
    cB=copy.deepcopy(c);cB['delay']['execution_delay_A_s']={'value':0.0};cB['delay']['execution_delay_B_s']={'value':0.2}
    check('per-aircraft delay falls back to global when absent',abs(m.delay_of(c,'A')-float(m.value(c,'delay','execution_delay_baseline_s')))<1e-15,'analytic')
    check('per-aircraft delay fields are read when present',m.delay_of(cB,'A')==0.0 and abs(m.delay_of(cB,'B')-.2)<1e-15,'analytic')

    # ---------- Q15: perceived() consistency ----------
    cB2=copy.deepcopy(c);cB2['information_structure']['asymmetry_method_check']['B_rel_pos_bias_m']['value']=[0.,4.,0.]
    check('observer position bias shifts both pB and reconstructed qB',np.allclose(m.perceived(cB2,s,'B',True)['qB']-s['qB'],np.array([0.,4.,0.])),'analytic')
    cA2=copy.deepcopy(c);cA2['information_structure']['asymmetry_method_check']['A_rel_pos_bias_m']['value']=[0.,4.,0.]
    check('A-side bias also shifts qB (v6 omitted it)',np.allclose(m.perceived(cA2,s,'A',True)['qB']-s['qB'],np.array([0.,4.,0.])),'analytic')

    # ---------- simulated_input ----------
    scene=m.scenario(c);real_at_time=m.at_time
    small={'t_c_s':10.,'t1_end_s':11.5,'t2_end_s':12.5,'t_f_s':13.5}
    def jump_at_time(state,t):
        shifted=real_at_time(state,t)
        if state is scene and t>=10.5-1e-9:shifted['pA']=shifted['pA']+np.array([100.,0.,0.])
        return shifted
    def controlled_outcomes(config,state,z,asym=False):
        pr={'A':.1+.0005*state['pA'][0],'B':.3+.0002*state['pB'][0]}
        return {i:{pair:{'U':pr[i],'Pcol_max':.2,'C_safety':.1,'C_efficiency':.3} for pair in m.PAIRS} for i in 'AB'}
    def controlled_qre(config,out,previous=None,forward_previous=None):
        a=out['A']['N,N']['U'];b=out['B']['N,N']['U']
        row={'pA_N':a,'pB_N':b,'pA_Y':1-a,'pB_Y':1-b,'pNN':a*b,'residual':0.,'root_count_found':1}
        return row,[(0.3,0.3)],lambda y:1.,lambda x:1.
    with patch.object(m,'at_time',side_effect=jump_at_time),patch.object(m,'outcomes',side_effect=controlled_outcomes),patch.object(m,'qre',side_effect=controlled_qre):
        timing_rows,_,_=m.dynamic(c,scene,small,Z)
    first=timing_rows[0];after=timing_rows[1]
    # Controlled probabilities are closed form: pA_N = .1 + .0005 * pA_x, with
    # pA_x(tau) = V*tau for the straight-ahead baseline plus the injected +100 m
    # jump for tau >= 10.5 s. Forecast samples sit at tau = t+dt, t+2dt, ...
    dt=float(m.value(c,'dynamic_decision','decision_update_interval_s'))
    V=float(m.value(c,'aircraft','cruise_speed_mps'))
    t0=10.
    cur=[.1+.0005*V*t0]
    fut=[.1+.0005*V*(t0+dt*(k+1)) for k in range(int(round((11.5-t0)/dt))-1)]
    check('future observation jump cannot leak into an earlier forecast',
          abs(first['stage1_A_mean_N']-np.mean(cur+fut))<1e-12,'simulated_input')
    t1=t0+dt
    exp_fc=[.1+.0005*(V*(t1+dt*(k+1))+100.) for k in range(int(round((11.5-t1)/dt))-1)]
    check('current-stage remainder responds to changed current state',
          abs(after['stage1_forecast_A_mean_N']-np.mean(exp_fc))<1e-12
          and after['stage1_forecast_sample_count']==len(exp_fc),'simulated_input')
    check('current remaining mean excludes the earlier history point',
          abs(after['stage1_A_mean_N']-np.mean([.1+.0005*(V*t1+100.)]+exp_fc))<1e-12
          and after['stage1_observed_sample_count']==1
          and after['stage1_sample_count']==1+len(exp_fc)
          and after['stage1_excluded_history_count']==1,'simulated_input')

    # ---------- results-dependent ----------
    out=root/'results'
    if (out/'summary.json').exists():
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        rows=list(csv.DictReader((out/'dynamic_qre.csv').open(encoding='utf-8-sig')))
        stages=list(csv.DictReader((out/'stage_summary.csv').open(encoding='utf-8-sig')))
        count_by_stage={int(r['stage']):int(r['sample_count']) for r in stages}
        max_prod=0.;max_rem=0.;max_fus=0.;max_cur=0.;max_win=0.;max_tail=0.
        prov=True;cov=True;spl=True
        for j,row in enumerate(rows):
            st=int(row['stage']);f=[float(row[f'stage{k}_q_mean_product']) for k in (1,2,3)]
            I=int(row['I_effective_yield_history'])
            max_prod=max(max_prod,abs(float(row['P_three_stage_dynamic'])-I*float(np.prod(f))))
            max_rem=max(max_rem,abs(float(row['R_remaining_stage_index'])-I*float(np.prod(f[st-1:]))))
            for k in (1,2,3):
                exp='completed_observed_mean' if k<st else 'current_remaining_window_from_now_NN_forecast_mean' if k==st else 'future_NN_model_forecast_mean'
                prov=prov and row[f'stage{k}_provenance']==exp
                no=int(row[f'stage{k}_observed_sample_count']);nf=int(row[f'stage{k}_forecast_sample_count'])
                ex=int(row[f'stage{k}_excluded_history_count'])
                cov=cov and no+nf==int(row[f'stage{k}_sample_count'])==count_by_stage[k]-ex
                for pl in 'AB':
                    so=no*float(row[f'stage{k}_observed_{pl}_mean_N']) if no else 0.
                    sf=nf*float(row[f'stage{k}_forecast_{pl}_mean_N']) if nf else 0.
                    max_fus=max(max_fus,abs(float(row[f'stage{k}_{pl}_mean_N'])-(so+sf)/(no+nf)))
                spl=spl and (nf==0 if k<st else no==0 if k>st else True)
            same=[q for q in rows[:j+1] if q['stage']==row['stage']]
            spl=spl and int(row[f'stage{st}_observed_sample_count'])==1 and int(row[f'stage{st}_excluded_history_count'])==len(same)-1
            max_win=max(max_win,abs(float(row[f'stage{st}_window_start_s'])-float(row['t_s'])))
            max_cur=max(max_cur,abs(float(row[f'stage{st}_observed_A_mean_N'])-float(row['pA_N'])))
            tail=[q for q in rows[j:] if q['stage']==row['stage']]
            for pl in 'AB':max_tail=max(max_tail,abs(float(row[f'stage{st}_{pl}_mean_N'])-np.mean([float(q[f'p{pl}_N']) for q in tail])))
        check('online rows retain all three factors',max_prod<1e-12,'self_consistency')
        check('remaining index drops only completed stage factors',max_rem<1e-12,'self_consistency')
        check('factor provenance records observation vs forecast',prov,'self_consistency')
        check('measured part is the current point only',max_cur<1e-12,'self_consistency')
        check('denominator equals remaining count',cov,'self_consistency')
        check('selected window individual mean matches sample sums',max_fus<1e-12,'self_consistency')
        check('past-current-future sample partitions are correct',spl,'self_consistency')
        check('current window lower bound equals current time',max_win<1e-10,'self_consistency')
        check('baseline current factor equals predicted suffix mean',max_tail<1e-12,'self_consistency')
        first=rows[0];last=rows[-1]
        check('first current stage sample has a nonempty predicted remainder',int(first['stage1_observed_sample_count'])==1 and int(first['stage1_forecast_sample_count'])==count_by_stage[1]-1,'self_consistency')
        check('last remaining grid window contains only the current point',int(last['stage3_forecast_sample_count'])==0 and int(last['stage3_sample_count'])==1,'self_consistency')
        check('all QRE residuals acceptable',summary['max_QRE_residual']<1e-8,'convergence')
        check('retrospective product equals stage-summary product',abs(summary['P_three_stage_full_history_retrospective']-float(np.prod([float(x['q_mean_product']) for x in stages])))<1e-12,'self_consistency')
        refs=list(csv.DictReader((out/'integration_resolution.csv').open(encoding='utf-8-sig')))
        check('boundary remains feasible at refined integration steps',all(x['feasible_at_baseline_grid_boundary']=='True' for x in refs),'convergence')
        mc=list(csv.DictReader((out/'mc_repeats.csv').open(encoding='utf-8-sig')))
        check('pNN identical across 1024/4096 samples: the 0/1 pattern is physical, not sampling noise',
              max(float(x['pNN']) for x in mc)-min(float(x['pNN']) for x in mc)<1e-12,'convergence')
        if (out/'prediction_horizon_sensitivity.csv').exists():
            hs=list(csv.DictReader((out/'prediction_horizon_sensitivity.csv').open(encoding='utf-8-sig')))
            check('t_c is independent of the prediction horizon H (Q7)',len({x['t_c_s'] for x in hs})==1,'self_consistency')
        check('geometry boundary is declared primary',summary['boundary_criterion_primary'].startswith('geometry'),'self_consistency')
        if summary.get('boundary_probability_acceptance_t_f_s') is not None:
            check('probability-acceptance boundary reported separately, never larger than geometry',
                  summary['boundary_probability_acceptance_t_f_s']<=summary['boundary_geometry_t_f_s']+1e-9,'self_consistency')
        term=summary.get('cutoff_boundary_numeric_diagnostic')
        check('cutoff diagnostic is labelled as a numerical diagnostic',isinstance(term,dict) and term.get('probe_kind')=='cutoff_boundary_numeric_diagnostic','self_consistency')
        check('no "left_limit" claim remains in summary keys',not any('left_limit' in k for k in summary),'self_consistency')
        if (out/'lambda_sweep.csv').exists():
            ls=list(csv.DictReader((out/'lambda_sweep.csv').open(encoding='utf-8-sig')))
            check('every lambda row reports candidate roots and search method',all('all_candidate_roots_x_y' in x and 'root_search_method' in x for x in ls),'self_consistency')
            check('forward and reverse scans agree on sign changes',all(x['forward_reverse_consistent']=='True' for x in ls),'convergence')
        audit=list(csv.DictReader((out/'effective_yield_audit.csv').open(encoding='utf-8-sig')))
        check('effective-yield audit keeps physical probability independent',audit[-1]['I_after_confirmation']=='0' and float(audit[-1]['Pcol_max_over_commanded_trajectory'])>0,'self_consistency')
        b=summary['boundaries']
        ev_log=[{'time_s':12.,'player':'A','action':'Y','executed':False,'effective':False},
                {'time_s':26.,'player':'A','action':'Y','executed':True,'effective':True}]
        branch,_,_=m.dynamic(c,m.scenario(c),b,Z,events=ev_log)
        check('effective-Y branch final risk is zero',branch[-1]['P_three_stage_dynamic']==0.,'self_consistency')
        check('intent-only row preserves behavior risk',next(r for r in branch if abs(r['t_s']-12)<1e-8)['P_three_stage_dynamic']>0,'self_consistency')
        term_rows=[r for r in branch if r['t_s']>=26.-1e-9]
        check('terminated behavior has independently nonzero NN physical audit',all(r['P_three_stage_dynamic']==0 and r['A_Pcol_max_NN']==1. for r in term_rows),'self_consistency')

    by_kind={}
    for q in checks:by_kind[q['kind']]=by_kind.get(q['kind'],0)+1
    payload={'total_passed':len(checks),'by_kind_actual':by_kind,
             'classification_note':'counts measured from the checks actually executed, not a fixed estimate',
             'scope_note':'PASSING IS NOT EXTERNAL MODEL VALIDATION. It shows the implementation matches its own definitions and the stated analytic references.',
             'max_dubins_endpoint_error_m':maxerr,'checks':checks}
    m.dump(out/'verification.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='checks'},ensure_ascii=False,indent=2))

if __name__=='__main__':run()