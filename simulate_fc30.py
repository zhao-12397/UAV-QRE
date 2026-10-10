#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FC30 two-operator game simulation: equations of 无人机博弈_彻底重修版.
v7 (working revision of v6): safety-loss form, prediction horizon, delay
asymmetry and sigma now switchable; v6 three-stage remaining-window aggregation
retained unchanged. Original-paper linear utility added as a labelled
comparison track only, never as the primary result.
Units: metres, seconds, radians internally.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

PAIRS = ('Y,Y','Y,N','N,Y','N,N')
SAFETY_FORMS = ('saturating','continuous','paper_linear')
P_REF_ROLES=('normalisation_scale','acceptance_threshold')

def value(c, *keys):
    for k in keys: c = c[k]
    return c['value'] if isinstance(c,dict) and 'value' in c else c

def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def mod(x): return x%(2*np.pi)
def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi
def unit(h): return np.array([np.cos(h),np.sin(h),0.])

def cpa(p,v):
    vv=float(v@v)
    if vv<1e-15: return None,float(np.linalg.norm(p))
    t=-float(p@v)/vv
    return t,float(np.linalg.norm(p+max(t,0.)*v))

def basis(v):
    x=np.array([v[0],v[1],0.]); n=np.linalg.norm(x)
    x=x/n if n>1e-12 else np.array([1.,0.,0.])
    return np.array([x,np.cross([0.,0.,1.],x),[0.,0.,1.]])

def segment_min_q(rel, B, axes):
    """Continuous minimum on linear inter-sample segments; prevents missed crossings."""
    u=rel@B.T/axes
    if len(u)==1: return float(np.sum(u*u))
    d=np.diff(u,axis=0); den=np.sum(d*d,axis=1)
    a=np.divide(-np.sum(u[:-1]*d,axis=1),den,out=np.zeros_like(den),where=den>1e-18)
    z=u[:-1]+np.clip(a,0,1)[:,None]*d
    return float(min(np.min(np.sum(z*z,axis=1)),np.sum(u[-1]*u[-1])))

def scenario(c,T=None,D=None,z=None,angle=None):
    V=float(value(c,'aircraft','cruise_speed_mps'))
    T=float(value(c,'scenario','baseline_target_tcpa_s') if T is None else T)
    D=float(value(c,'scenario','baseline_target_dcpa_m') if D is None else D)
    z=float(value(c,'scenario','vertical_offset_m') if z is None else z)
    ha=np.deg2rad(value(c,'scenario','heading_A_deg'))
    hb=np.deg2rad(value(c,'scenario','heading_B_deg') if angle is None else angle)
    va,vb=V*unit(ha),V*unit(hb); B=basis(vb-va)
    pa=np.zeros(3);pb=-(vb-va)*T+D*B[1]+z*B[2]
    # Missing in v3: independent merge points chosen far enough downstream.
    span=T+float(value(c,'yield_action','recovery_tail_s'))+2*float(value(c,'yield_action','baseline_heading_change_deg'))/float(value(c,'yield_action','baseline_turn_rate_degps'))
    return dict(pA=pa,pB=pb,vA=va,vB=vb,hA=ha,hB=hb,basis=B,
                qA=pa+V*span*unit(ha),qB=pb+V*span*unit(hb),initial_tcpa=T,merge_span_s=span)

def at_time(s,t):
    out=copy.deepcopy(s);out['pA']=s['pA']+s['vA']*t;out['pB']=s['pB']+s['vB']*t
    return out

def advance_pose(p,h,V,w,t):
    if abs(w)<1e-14: return p+V*t*unit(h),h
    hh=h+w*t
    pp=p+np.array([V/w*(np.sin(hh)-np.sin(h)),V/w*(np.cos(h)-np.cos(hh)),0.])
    return pp,hh

# Analytic Dubins paths: bounded horizontal turn rate, fixed speed, end at q
# with original heading. Six paths are enumerated; selected recovery is fixed.
def dubins(p,h,q,hq,R):
    dx,dy=(q-p)[:2]/R;d=np.hypot(dx,dy);theta=np.arctan2(dy,dx)
    a=mod(h-theta);b=mod(hq-theta);sa,sb,ca,cb=np.sin(a),np.sin(b),np.cos(a),np.cos(b)
    cab=np.cos(a-b); options=[]
    def add(word,ts):
        if ts is not None and np.all(np.isfinite(ts)): options.append((word,np.array(ts)))
    t0=np.arctan2(cb-ca,d+sa-sb);p2=2+d*d-2*cab+2*d*(sa-sb)
    if p2>=0:add('LSL',(mod(-a+t0),np.sqrt(p2),mod(b-t0)))
    t0=np.arctan2(ca-cb,d-sa+sb);p2=2+d*d-2*cab+2*d*(sb-sa)
    if p2>=0:add('RSR',(mod(a-t0),np.sqrt(p2),mod(-b+t0)))
    p2=-2+d*d+2*cab+2*d*(sa+sb)
    if p2>=0:
        pp=np.sqrt(p2);tt=np.arctan2(-ca-cb,d+sa+sb)-np.arctan2(-2.,pp)
        add('LSR',(mod(-a+tt),pp,mod(-b+tt)))
    p2=d*d-2+2*cab-2*d*(sa+sb)
    if p2>=0:
        pp=np.sqrt(p2);tt=np.arctan2(ca+cb,d-sa-sb)-np.arctan2(2.,pp)
        add('RSL',(mod(a-tt),pp,mod(b-tt)))
    tmp=(6-d*d+2*cab+2*d*(sa-sb))/8
    if abs(tmp)<=1:
        pp=mod(2*np.pi-np.arccos(tmp));tt=mod(a-np.arctan2(ca-cb,d-sa+sb)+pp/2)
        add('RLR',(tt,pp,mod(a-b-tt+pp)))
    tmp=(6-d*d+2*cab+2*d*(-sa+sb))/8
    if abs(tmp)<=1:
        pp=mod(2*np.pi-np.arccos(tmp));tt=mod(-a-np.arctan2(ca-cb,d+sa-sb)+pp/2)
        add('LRL',(tt,pp,mod(b-a-tt+pp)))
    if not options: raise RuntimeError('No recovery Dubins path')
    return min(options,key=lambda x:sum(x[1]))

def path_positions(p,h,V,segments,times):
    times=np.asarray(times);out=np.empty((len(times),3));heads=np.empty(len(times));omega=np.zeros(len(times))
    elapsed=0.;pos=p.copy();hd=h
    for duration,w in segments+[(float('inf'),0.)]:
        mask=(times>=elapsed-1e-10)&(times<elapsed+duration-1e-10)
        tt=times[mask]-elapsed
        if abs(w)<1e-14:
            out[mask]=pos+tt[:,None]*V*unit(hd);heads[mask]=hd
        else:
            hh=hd+w*tt
            out[mask]=pos+np.column_stack((V/w*(np.sin(hh)-np.sin(hd)),V/w*(np.cos(hd)-np.cos(hh)),np.zeros(len(tt))))
            heads[mask]=hh
        omega[mask]=w
        if not np.isfinite(duration):break
        pos,hd=advance_pose(pos,hd,V,w,duration);elapsed+=duration
    return out,heads,omega

def trajectory(c,s,pair,dt=None):
    dt=float(value(c,'numerics','integration_dt_s') if dt is None else dt)
    H=float(value(c,'collision_probability','prediction_horizon_s'))
    delays={i:delay_of(c,i) for i in 'AB'}
    angle=np.deg2rad(value(c,'yield_action','baseline_heading_change_deg'))
    wabs=np.deg2rad(value(c,'yield_action','baseline_turn_rate_degps'));w=float(value(c,'yield_action','right_turn_sign'))*wabs
    turn=angle/wabs;confirm=float(value(c,'yield_action','separation_confirmation_s'))
    axes=np.array(value(c,'geometry','combined_collision_half_axes_m'),float)
    safe=axes+np.array(c['implementation']['safety_margin_m'],float)
    tc,_=cpa(s['pB']-s['pA'],s['vB']-s['vA']);tc=max(tc or 0.,0.)
    actions=pair.split(',');prefix={};speeds={i:float(np.linalg.norm(s['v'+i])) for i in 'AB'}
    for i,act in zip('AB',actions):
        prefix[i]=[(delays[i],0.),(turn,w)] if act=='Y' else []
    # Same separation confirmation for both vehicles; straight prediction gate.
    hold_tail=max(delays['A'] if 'Y' in actions else 0.,delays['B'] if 'Y' in actions else 0.)
    tpre=np.arange(0.,max(H,tc+float(value(c,'yield_action','recovery_tail_s'))+hold_tail)+dt/2,dt)
    p0={i:path_positions(s['p'+i],s['h'+i],speeds[i],prefix[i],tpre) for i in 'AB'}
    rel=p0['B'][0]-p0['A'][0];qr=np.sum((rel@s['basis'].T/safe)**2,axis=1)
    increasing=np.r_[False,np.diff(qr)>1e-10];run=0.;recovery_time=None
    for k,t in enumerate(tpre):
        run=run+dt if increasing[k] else 0.
        if t+1e-10<max(delays['A'] if 'Y' in actions else 0.,delays['B'] if 'Y' in actions else 0.)+turn and 'Y' in actions:continue
        if run+1e-10<confirm or qr[k]<=1.:continue
        vr=speeds['B']*unit(p0['B'][1][k])-speeds['A']*unit(p0['A'][1][k])
        rr=rel[k]@s['basis'].T/safe;vv=vr@s['basis'].T/safe
        tau=np.clip(-rr@vv/max(vv@vv,1e-15),0,confirm)
        if np.sum((rr+tau*vv)**2)>1.:recovery_time=float(t);break
    segments={};metrics={};end_errors={}
    for i,act in zip('AB',actions):
        V=speeds[i];L0=float(np.linalg.norm(s['q'+i]-s['p'+i]));delay=delays[i]
        if act=='N':
            segments[i]=[];duration=L0/V;extra=0.;smooth=0.;arrive=True;end_errors[i]=0.
        elif recovery_time is None:
            segments[i]=prefix[i];duration=float(tpre[-1]);extra=None;smooth=None;arrive=False;end_errors[i]=None
        else:
            segments[i]=prefix[i]+[(max(0.,recovery_time-delay-turn),0.)]
            pp,hh=s['p'+i].copy(),s['h'+i]
            for dur,ww in segments[i]:pp,hh=advance_pose(pp,hh,V,ww,dur)
            word,lengths=dubins(pp,hh,s['q'+i],s['h'+i],V/wabs)
            rec=[(float(ll/wabs),wabs if mode=='L' else -wabs if mode=='R' else 0.) for mode,ll in zip(word,lengths)]
            segments[i]+=rec;duration=sum(d for d,_ in segments[i]);extra=max(0.,V*duration-L0)
            smooth=min(sum(ww*ww*d for d,ww in segments[i])/(np.deg2rad(value(c,'utility','smooth_turn_rate_reference_degps'))**2*max(duration,1e-12)),1.)
            ppend,hhend=pp,hh
            for dur,ww in rec:ppend,hhend=advance_pose(ppend,hhend,V,ww,dur)
            end_errors[i]=float(np.linalg.norm(ppend-s['q'+i]));arrive=end_errors[i]<1e-6 and abs(wrap(hhend-s['h'+i]))<1e-7
        metrics[i]={'baseline_length_m':L0,'arrival_time_s':duration,'extra_length_m':extra,
                    'C_length':min(extra/max(L0,1e-12),1.) if extra is not None else 1.,
                    'C_smooth':smooth if smooth is not None else 1.,
                    'rejoined':arrive,'endpoint_error_m':end_errors[i],'execution_delay_s':delay}
    end=max(H,max(x['arrival_time_s'] for x in metrics.values()))
    times=np.unique(np.r_[np.arange(0,end,dt),end,H])
    pp={i:path_positions(s['p'+i],s['h'+i],speeds[i],segments[i],times) for i in 'AB'}
    rel=pp['B'][0]-pp['A'][0]
    return dict(times=times,rel=rel,pA=pp['A'][0],pB=pp['B'][0],omegaA=pp['A'][2],omegaB=pp['B'][2],metrics=metrics,
                min_q_collision=segment_min_q(rel,s['basis'],axes),min_q_safe=segment_min_q(rel,s['basis'],safe),
                min_distance_m=segment_min_q(rel,np.eye(3),np.ones(3))**.5,recovery_time_s=recovery_time,
                recovery_completed=all(m['rejoined'] for m in metrics.values()),basis=s['basis'])

def gate(c,s):
    r=s['pB']-s['pA'];v=s['vB']-s['vA'];tc,dc=cpa(r,v);H=float(value(c,'collision_probability','prediction_horizon_s'))
    safe=np.array(value(c,'geometry','combined_collision_half_axes_m'))+c['implementation']['safety_margin_m']
    mq=segment_min_q(np.array([r,r+v*H]),s['basis'],safe)
    return bool(tc is not None and 0<tc<=H+1e-9 and mq<=1.),tc,dc,mq

def search_boundary(c,s,pair,mode='geometry',tol=1e-9,Z=None):
    """Scan EVERY configured command time: no monotonic feasibility assumption.

    mode='geometry'   : deterministic criterion, min_q_safe > 1 (+ recovery).
                        This is the v6 criterion and the primary boundary.
    mode='probability' : additional constraint Pcol_max < P_max on the commanded
                        trajectory, where P_max is an ACCEPTANCE THRESHOLD. This
                        is a different concept from Pref (a normalisation scale)
                        even when the numbers coincide, so it is named separately
                        and reported in parallel rather than merged.
    """
    tc,_=cpa(s['pB']-s['pA'],s['vB']-s['vA']);step=float(value(c,'safety_boundary','latest_command_search_step_s'))
    records=[];last=None
    # Include pre-command N,N history in validity: delayed commands cannot erase a past overlap.
    axes=np.array(value(c,'geometry','combined_collision_half_axes_m'))+c['implementation']['safety_margin_m']
    r=s['pB']-s['pA'];v=s['vB']-s['vA']
    sp,sv=sigma_of(c);pmax=float(c['implementation']['probability_acceptance_threshold'])
    for t in np.arange(0,max(tc or 0.,0.)+step/2,step):
        res=trajectory(c,at_time(s,float(t)),pair)
        prior=segment_min_q(np.array([r,r+v*t]),s['basis'],axes)
        geom=bool(prior>1 and res['min_q_safe']>1 and res['recovery_completed'])
        pcol=float('nan')
        if mode=='probability':
            # Z=None falls back to the deterministic criterion inside probability(),
            # which would make this track identical to geometry. Require samples.
            if Z is None:raise ValueError('probability criterion needs MC samples Z')
            pcol=probability(c,res,sp,sv,Z)[0]
        ok=geom and (True if mode=='geometry' else bool(pcol<pmax))
        row={'pair':pair,'criterion':mode,'command_time_s':float(t),'feasible':bool(ok),
             'geometry_feasible':geom,'min_q_safe':res['min_q_safe'],'min_q_collision':res['min_q_collision'],
             'min_distance_m':res['min_distance_m'],'recovery_completed':res['recovery_completed']}
        if mode=='probability':
            row['Pcol_max_commanded']=pcol;row['P_max_threshold']=pmax
        records.append(row)
        if ok:last=row
    bits=[x['feasible'] for x in records];monotone=not any((not bits[i]) and bits[i+1] for i in range(len(bits)-1))
    return {'pair':pair,'criterion':mode,'last':last,'monotone_on_grid':monotone,'records':records}

def delay_of(c,player):
    """Per-aircraft execution delay. Falls back to the global v3 field so that
    configs without the per-aircraft fields reproduce v6 behaviour exactly."""
    key='execution_delay_'+player+'_s'
    node=c['delay'].get(key)
    if node is None:return float(value(c,'delay','execution_delay_baseline_s'))
    return float(node['value'] if isinstance(node,dict) and 'value' in node else node)

def perceived(c,s,player,asym=False):
    """Observer-side reconstruction of the encounter state.

    pb/vb are the observer's position/velocity perception bias applied to the
    OPPONENT (relative position of B as reconstructed by `player`). Because the
    opponent's merge point qB is derived from pB(0) and hB, a consistent
    subjective reconstruction must shift qB by the same pb. v6 shifted qB only
    when player=='B', which was inconsistent: the same pb affected pB for both
    players but qB for only one. Fixed here (Q15): qB is always shifted by pb.
    """
    out=copy.deepcopy(s)
    if asym:
        pb=value(c,'information_structure','asymmetry_method_check',player+'_rel_pos_bias_m');vb=value(c,'information_structure','asymmetry_method_check',player+'_rel_vel_bias_mps')
    else:
        bias=value(c,'information_structure','perception_bias_baseline');pb=bias[player+'_rel_pos_bias_m'];vb=bias[player+'_rel_vel_bias_mps']
    obs=float(value(c,'information_structure','observation_delay_s'))
    out['pA']-=obs*s['vA'];out['pB']-=obs*s['vB']
    out['pB']+=np.array(pb);out['vB']+=np.array(vb)
    out['hB']=np.arctan2(out['vB'][1],out['vB'][0])
    # Physical encounter ellipsoid frame is held fixed; bias is in world coordinates.
    # qB is reconstructed from the biased pB/hB, so it takes the same shift.
    out['qB']=out['qB']+np.array(pb)
    return out

def probability(c,res,sp,sv,Z):
    """max_tau Pcol(tau); NOT probability of any collision over the path.
    Search ALL prediction samples (v3 four closest-time approximation is not used).
    Zero-hit MC estimates are reported with an upper bound, not a proof of safety.
    """
    H=float(value(c,'collision_probability','prediction_horizon_s'));mask=res['times']<=H+1e-9
    times=res['times'][mask];rel=res['rel'][mask];axes=np.array(value(c,'geometry','combined_collision_half_axes_m'))
    if sp==0 and sv==0:
        p=float(segment_min_q(rel,res['basis'],axes)<=1.)
        return p,0.,0.,0
    probs=[]
    # Common random numbers for reproducibility and lower sweep comparison noise.
    for start in range(0,len(times),32):
        tt=times[start:start+32];sig=np.sqrt(2*sp*sp+2*(sv*tt)**2)
        samples=rel[start:start+32,None,:]+sig[:,None,None]*Z[None,:,:]
        q=np.sum((samples@res['basis'].T/axes)**2,axis=2)
        probs.extend(np.mean(q<=1.,axis=1).tolist())
    k=int(np.argmax(probs));p=probs[k];n=len(Z);zz=1.95996398454
    den=1+zz*zz/n;center=(p+zz*zz/(2*n))/den
    radius=zz*np.sqrt(p*(1-p)/n+zz*zz/(4*n*n))/den
    return float(p),float(max(0,center-radius)),float(min(1,center+radius)),float(times[k])

def sigma_of(c):
    return (float(value(c,'uncertainty','baseline_position_sigma_m')),
            float(value(c,'uncertainty','baseline_velocity_sigma_mps')))

def safety_loss_form(c):
    return str(c.get('implementation',{}).get('safety_loss_form','saturating'))

def safety_loss(p,mode,pref):
    """Safety loss in[0,1]. Three explicitly labelled forms:

    saturating : Cs = min(Pcol/Pref,1)  -- v6 baseline (重修版第7节), retained
                 as the pre-revision reference. Note this is a HARD CLIP: any
                 Pcol >= Pref collapses to 1.0, so the safety term has no
                 gradient over that whole region.
    continuous : Cs = Pcol/(Pcol+Pref)  -- smooth, monotone, never clips.
    paper_linear : see paper_linear_safety(); replaces Pcol by CPA-based terms.

    Pref plays DIFFERENT roles and must not be conflated (Q3):
      - in saturating/continuous it is a NORMALISATION SCALE for the loss shape
      - it is NOT an acceptance threshold; acceptance is P_max, used only by the
        separately reported probability-feasibility criterion.
    """
    if mode=='saturating':return min(p/pref,1.)
    if mode=='continuous':return p/(p+pref)
    if mode=='paper_linear':raise ValueError('paper_linear must be computed by paper_linear_safety()')
    raise ValueError('Unknown safety_loss_form: '+str(mode))

def paper_linear_safety(c,state,res=None):
    """Comparison track only: original-paper Eq.(10)(11) shape, migrated.

    Original:  EU_NotKeep = lambda_safety - (a*DCPA + b*TCPA)
    with DCPA in cables(1852 m) and TCPA in minutes, coefficients tabulated by
    gross tonnage class (Debnath & Chin 2010). Neither cable/minute nor tonnage
    classes exist for FC30, so both axes are explicitly rescaled:

      danger_i = clip(1 - min_q_collision, 0, 1)      dimensionless danger
                 = 1 when the commanded trajectory touches the collision
                   ellipsoid, 0 when nominal clearance >= 1. Plays the role of
                   the paper's DCPA term (larger = more dangerous).
      tq = clip(TCPA_remaining / H, 0, 1)              dimensionless time pressure
                 = 1 at the prediction-horizon boundary, 0 at CPA.

    The safety LOSS is the normalised shortfall of the not-yielding utility:
      Cs = (a*danger + b*tq) / (a + b)
    so Cs = 1 at maximum danger with zero time margin, and -> 0 as either the
    nominal trajectory clears the ellipsoid or TCPA reaches CPA. Unlike the
    saturating clip this has NO flat region, so it can carry a time gradient.

    LAMBDA_ORIGIN: the original paper writes lambda for the safety margin in
    Eq.(10)(11). To avoid confusion with the QRE rationality parameter lambda_A/
    lambda_B, this revision names the margin MU and the coefficients
    A_DCPA / B_TCPA. Tabulated values (VC3 day: 0.2641, 0.1151, 3.4408) are
    used only to demonstrate the SHAPE; they are NOT FC30-calibrated and MU
    only enters through the normalisation, so Cs depends on the ratio a:b.

    KNOWN LIMITATION (head-on scenario): the paper's DCPA is a true distance
    and varies continuously; here the commanded-trajectory min_q_collision is
    either ~0 (touching) or >1 (clear), so the danger term is close to binary in
    this scenario. The time term tq is what carries the gradient here.
    """
    if res is not None:
        mq=float(res['min_q_collision'])
    else:
        mq=float(gate(c,state)[3])
    danger=float(np.clip(1.-mq,0.,1.))
    tc=float(gate(c,state)[1] or 0.)
    tq=float(np.clip(max(tc,0.)/float(value(c,'collision_probability','prediction_horizon_s')),0.,1.))
    a=float(c['implementation']['paper_linear_a_dcpa']);b=float(c['implementation']['paper_linear_b_tcpa'])
    return float(np.clip((a*danger+b*tq)/(a+b),0.,1.))

def outcomes(c,s,Z,asym=False):
    sp,sv=sigma_of(c)
    form=safety_loss_form(c);pref=float(value(c,'collision_probability','probability_reference'))
    out={};cache={}
    for i in 'AB':
        pst=perceived(c,s,i,asym);key=tuple(np.r_[pst['pA'],pst['pB'],pst['vA'],pst['vB'],pst['qA'],pst['qB']])
        if key not in cache:cache[key]={pair:trajectory(c,pst,pair) for pair in PAIRS}
        out[i]={}
        for pair,res in cache[key].items():
            p,lo,hi,tp=probability(c,res,sp,sv,Z);m=res['metrics'][i]
            if form=='paper_linear':
                Cs=paper_linear_safety(c,pst,res)
            else:
                Cs=safety_loss(p,form,pref)
            Ce=float(value(c,'utility','efficiency_path_weight'))*m['C_length']+float(value(c,'utility','efficiency_smooth_weight'))*m['C_smooth']
            U=float(value(c,'utility','safety_weight'))*(1-Cs)+float(value(c,'utility','efficiency_weight'))*(1-Ce)
            out[i][pair]={'U':U,'Pcol_max':p,'Pcol_pointwise_CI_low':lo,'Pcol_pointwise_CI_high':hi,'Pcol_peak_tau_s':tp,'C_safety':Cs,'C_efficiency':Ce,**m,'min_q_safe':res['min_q_safe']}
    return out

def _qre_roots(c,A,B,grid=2001,xtol=1e-13):
    """Enumerate candidate fixed points of the composed best-response map.
    Returns (roots, method). Method is reported so the caller can state that the
    found roots are CANDIDATES on a grid, not a proof that all equilibria were
    captured (Q11: do not call detected roots 'all equilibria')."""
    la=float(value(c,'qre','lambda_A'));lb=float(value(c,'qre','lambda_B'))
    def FA(y):return float(expit(la*(y*(A['Y,Y']-A['N,Y'])+(1-y)*(A['Y,N']-A['N,N']))))
    def FB(x):return float(expit(lb*(x*(B['Y,Y']-B['Y,N'])+(1-x)*(B['N,Y']-B['N,N']))))
    f=lambda x: FA(FB(x))-x
    xx=np.linspace(0.,1.,grid);ff=np.array([f(x) for x in xx])
    roots=[]
    for k in range(len(xx)-1):
        if abs(ff[k])<1e-12:roots.append(float(xx[k]))
        if ff[k]*ff[k+1]<0:roots.append(float(brentq(f,xx[k],xx[k+1],xtol=xtol)))
    if abs(ff[-1])<1e-12:roots.append(1.)
    # Tangent (sign-change-free) roots: refine any grid point where |f| is small
    # but no sign change was detected, so near-multiple roots are not dropped.
    for k in range(1,len(xx)-1):
        if ff[k-1]*ff[k+1]<0 and abs(ff[k])<1e-4:
            g=lambda x: abs(f(x))
            try:
                lo,hi=xx[k-1],xx[k+1]
                for _ in range(60):
                    m1=lo+(hi-lo)/3.;m2=hi-(hi-lo)/3.
                    if g(m1)<g(m2):hi=m2
                    else:lo=m1
                roots.append(float((lo+hi)/2))
            except Exception:pass
    roots=sorted({round(x,12) for x in roots if 0.<=x<=1.})
    return [(r,FB(r)) for r in roots],FA,FB,'grid_sign_change_plus_tangent_refinement'

def qre(c,out,previous=None,forward_previous=None):
    A={k:v['U'] for k,v in out['A'].items()};B={k:v['U'] for k,v in out['B'].items()}
    roots,FA,FB,method=_qre_roots(c,A,B)
    if not roots:raise RuntimeError('QRE bracket search found no root')
    ref=(.5,.5) if previous is None else previous
    x=min([r for r,_ in roots],key=lambda x:(x-ref[0])**2+(FB(x)-ref[1])**2);y=FB(x)
    residual=max(abs(x-FA(y)),abs(y-FB(x)))
    if residual>1e-8:raise RuntimeError(f'QRE equation residual {residual}')
    row={'pA_Y':x,'pB_Y':y,'pA_N':1-x,'pB_N':1-y,'pNN':(1-x)*(1-y),'residual':residual,
         'root_count_found':len(roots),'root_search_method':method}
    # Q11: path-dependence probe. Re-solving with no previous (cold start at0.5)
    # must reproduce the warm-start root; if not, the selection is path dependent.
    if forward_previous is not None:
        xc=min([r for r,_ in roots],key=lambda x:(x-.5)**2+(FB(x)-.5)**2)
        row['qre_root_path_consistent']=bool(abs(xc-x)<1e-9 or len(roots)>1)
    return row,roots,FA,FB

def evaluate(c,s,t,Z,asym=False,previous=None):
    st=at_time(s,t);out=outcomes(c,st,Z,asym);q,roots,FA,FB=qre(c,out,previous);g,tc,dc,mq=gate(c,st)
    row={'t_s':float(t),'TCPA_remaining_s':tc,'DCPA_m':dc,'G_current':g,'nominal_min_q_safe':mq}
    row.update(q)
    row['qre_root_x_values']=str([r for r,_ in roots])
    for i in 'AB':
        for pair in PAIRS:
            for field in ('U','Pcol_max','C_safety','C_efficiency'):
                row[i+'_'+field+'_'+pair.replace(',','')]=out[i][pair][field]
    return row,out,q

def t_conflict_formation(c,s):
    """Start of the decision process: conflict formation / monitoring trigger.

    v6 set t_c = max(0, T - H), i.e. it was derived BACKWARDS from the collision
    probability horizon. That coupled the stage-1 start to a numerical parameter
    and produced a knife edge (at t=10 s the CPA lay exactly on the H boundary,
    so pNN jumped by a factor of ~3 between t=9.5 s and t=10.0 s). Q7 requires t_c
    to be an independently declared quantity. It is now read from
    implementation.t_conflict_formation_s; if absent it falls back to the v6
    formula and the caller is told which rule was used.
    """
    declared=c.get('implementation',{}).get('t_conflict_formation_s')
    T=float(s['initial_tcpa'])
    if declared is None:
        H=float(value(c,'collision_probability','prediction_horizon_s'))
        return max(0.,T-H),'derived_from_H_fallback'
    return float(declared['value'] if isinstance(declared,dict) and 'value' in declared else declared),'declared'

def stage_bounds(c,s,last):
    T=float(s['initial_tcpa']);margin=T-last;k1=float(value(c,'dynamic_decision','kappa_1'));k2=float(value(c,'dynamic_decision','kappa_2'))
    if not k1>k2>1:raise ValueError('Require kappa1>kappa2>1')
    tc,tc_rule=t_conflict_formation(c,s);b1=T-k1*margin;b2=T-k2*margin
    if not tc<b1<b2<last:raise ValueError('Three nonempty stages require tc<b1<b2<tf (got tc=%.3f b1=%.3f b2=%.3f tf=%.3f)'%(tc,b1,b2,last))
    return {'t_c_s':tc,'t_c_rule':tc_rule,'t1_end_s':b1,'t2_end_s':b2,'t_f_s':last,'boundary_margin_DeltaT_f_s':margin}

def individual_mean_factor(samples):
    if not samples:raise ValueError('Empty stage: cannot replace it with another stage factor')
    return float(np.mean([r['pA_N'] for r in samples])),float(np.mean([r['pB_N'] for r in samples])),len(samples)

def stage_individual_means(measured,predicted):
    """Individual means of the selected window, before player multiplication.
    Current stage includes the measured current point and future predictions;
    excludes all current-stage observations strictly before NOW.
    """
    if not measured and not predicted:raise ValueError('Empty probability window')
    def segment(samples):
        return individual_mean_factor(samples)[:2] if samples else (None,None)
    oa,ob=segment(measured);pa,pb=segment(predicted)
    ma,mb,n=individual_mean_factor(measured+predicted)
    return {'A_mean_N':ma,'B_mean_N':mb,'sample_count':n,
            'observed_sample_count':len(measured),'forecast_sample_count':len(predicted),
            'observed_A_mean_N':oa,'observed_B_mean_N':ob,
            'forecast_A_mean_N':pa,'forecast_B_mean_N':pb}

def stage_grid(left,right,dt):
    grid=np.arange(left,right-1e-9,dt)
    return grid if len(grid) else np.array([left])

def history_indicator(events,t):
    """Absorbing branch indicator. Intent, pending, or failed Y cannot set it to 0.
    Event time is the observed/validated effective-execution confirmation time.
    The flag NEVER writes to or alters any physical-probability field.
    """
    for e in events:
        if float(e['time_s'])<=t+1e-10 and e.get('action')=='Y' and e.get('executed',False) and e.get('effective',False):return 0
    return 1

def dynamic(c,s,b,Z,asym=False,events=None):
    """Three-stage means according to user-approved Fig.5/Eq.17 logic.
    Past: completed individual historical means; current: remaining window
    [NOW, stage_end), with the current point once and forecasts strictly after it;
    future: model-predicted individual stage means on NN continuation.
    The latter two online rules are explicit migration choices, not a verified
    reconstruction of the original authors' implementation.
    """
    dt=float(value(c,'dynamic_decision','decision_update_interval_s'))
    endpoints=[b['t_c_s'],b['t1_end_s'],b['t2_end_s'],b['t_f_s']]
    grids=[stage_grid(l,r,dt) for l,r in zip(endpoints[:-1],endpoints[1:])]
    events=events or [];rows=[];summaries=[];previous=None;history={1:[],2:[],3:[]};cache={}
    # Cache ONLY model evaluations keyed by reconstructed predicted state, never
    # use future observed rows to construct any forecast remainder.
    def evaluate_state(st,abs_t,ref):
        psA=perceived(c,st,'A',asym);psB=perceived(c,st,'B',asym)
        key=tuple(np.round(np.concatenate([st['pA'],st['pB'],st['vA'],st['vB'],st['qA'],st['qB'],psA['pB'],psA['vB'],psB['pB'],psB['vB']]),8))
        if key not in cache:cache[key]=outcomes(c,st,Z,asym)
        ev=cache[key];qq,roots,FA,FB=qre(c,ev,ref);g,tc,dc,mq=gate(c,st)
        row={'t_s':float(abs_t),'TCPA_remaining_s':tc,'DCPA_m':dc,'G_current':g,'nominal_min_q_safe':mq}
        row.update(qq)
        row['qre_root_x_values']=str([r for r,_ in roots])
        for i in 'AB':
            for pair in PAIRS:
                for field in ('U','Pcol_max','C_safety','C_efficiency'):row[i+'_'+field+'_'+pair.replace(',','')]=ev[i][pair][field]
        return row,qq
    def predict_samples(current_state,current_t,sample_times,ref):
        predicted=[];prev=ref
        for tau in sample_times:
            if tau<=current_t+1e-9:raise ValueError('Forecast grid must exclude current and past observations')
            st=at_time(current_state,float(tau-current_t))
            r,q=evaluate_state(st,float(tau),prev);prev=(q['pA_Y'],q['pB_Y']);predicted.append(r)
        return predicted
    for stage,times in enumerate(grids,1):
        for t in times:
            current_state=at_time(s,float(t));row,q=evaluate_state(current_state,float(t),previous)
            previous=(q['pA_Y'],q['pB_Y']);row['stage']=stage;history[stage].append(row)
            factors=[]
            for k in (1,2,3):
                if k<stage:
                    fused=stage_individual_means(history[k],[]);provenance='completed_observed_mean'
                    window_start=endpoints[k-1];excluded=0
                elif k==stage:
                    future_times=grids[k-1][grids[k-1]>float(t)+1e-9]
                    predicted=predict_samples(current_state,float(t),future_times,previous)
                    fused=stage_individual_means([row],predicted)
                    provenance='current_remaining_window_from_now_NN_forecast_mean'
                    window_start=float(t);excluded=len(history[k])-1
                    v5_full_a,v5_full_b,_=individual_mean_factor(history[k]+predicted)
                else:
                    predicted=predict_samples(current_state,float(t),grids[k-1],previous)
                    fused=stage_individual_means([],predicted);provenance='future_NN_model_forecast_mean'
                    window_start=endpoints[k-1];excluded=0
                ma,mb,n=fused['A_mean_N'],fused['B_mean_N'],fused['sample_count']
                expected_count=len(grids[k-1])-excluded
                if n!=expected_count:raise RuntimeError('Remaining window must exclude consumed current-stage samples')
                row[f'stage{k}_A_mean_N']=ma;row[f'stage{k}_B_mean_N']=mb;row[f'stage{k}_q_mean_product']=ma*mb
                row[f'stage{k}_sample_count']=n;row[f'stage{k}_provenance']=provenance;factors.append(ma*mb)
                row[f'stage{k}_window_start_s']=window_start;row[f'stage{k}_window_end_s']=endpoints[k]
                row[f'stage{k}_excluded_history_count']=excluded
                for key in ('observed_sample_count','forecast_sample_count','observed_A_mean_N','observed_B_mean_N','forecast_A_mean_N','forecast_B_mean_N'):
                    row[f'stage{k}_'+key]=fused[key]
            I=history_indicator(events,float(t));row['I_effective_yield_history']=I
            row['P_three_stage_ungated']=float(np.prod(factors))
            row['P_three_stage_dynamic']=I*row['P_three_stage_ungated']
            # v4 prefix-only definition retained as an explicitly labeled comparison.
            prefix_a,prefix_b,_=individual_mean_factor(history[stage])
            comparison=factors.copy();comparison[stage-1]=prefix_a*prefix_b
            row['comparison_v4_prefix_P_three_stage']=I*float(np.prod(comparison))
            comparison=factors.copy();comparison[stage-1]=v5_full_a*v5_full_b
            row['comparison_v5_full_stage_P_three_stage']=I*float(np.prod(comparison))
            # Macro-stage conditional index: drops completed stage factors; not
            # an exact continuous-time survival probability without more assumptions.
            row['R_remaining_stage_index']=I*float(np.prod(factors[stage-1:]))
            row['decision_margin_s']=b['t_f_s']-float(t)
            row['physical_probability_state']='nominal_NN_counterfactual_audit' if I==0 else 'nominal_NN_current_state'
            # Old formula retained only for comparing changes; not the main result.
            prior_joint=[float(np.mean([r['pNN'] for r in history[k]])) for k in range(1,stage)]
            row['legacy_past_joint_means_times_instant_NN']=float(np.prod(prior_joint))*row['pNN']
            rows.append(row)
        prefix=history[stage];ma,mb,n=individual_mean_factor(prefix)
        old=float(np.mean([r['pNN'] for r in prefix]))
        summaries.append({'stage':stage,'start_s':endpoints[stage-1],'end_s':endpoints[stage],'sample_count':len(prefix),'mean_A_N':ma,'mean_B_N':mb,'q_mean_product':ma*mb,'comparison_mean_instant_joint_NN':old,'difference_old_minus_new':old-ma*mb})
    final_raw=float(np.prod([x['q_mean_product'] for x in summaries]))
    for row in rows:
        # Complete-history output is explicitly retrospective, never an online input.
        row['P_three_stage_final_retrospective']=final_raw
        row['retrospective_available_after_s']=b['t_f_s']
    # Boundary behaviour near tf is a numerical diagnostic only (Q10), not a
    # reported risk value and not part of any stage mean.
    terminal=cutoff_left_limit(c,s,b,summaries,Z,asym,events,previous)
    return rows,summaries,terminal

def cutoff_left_limit(c,s,b,stages,Z,asym=False,events=None,previous=None):
    """Q10: the v6 'mathematical left limit' claim and its naming are REMOVED.

    What was previously called a left limit was measured to be the saturating
    clip Cs=min(Pcol/Pref,1) switching state near tf, not a limit of the mean.
    It is retained only as a labelled NUMERICAL BOUNDARY DIAGNOSTIC: a sweep of
    several times, step sizes and tolerances around tf, recording where and why
    the value jumps. It is not a risk result and not a mathematical limit.
    """
    tf=float(b['t_f_s'])
    probes=[]
    for eps in [1e-1,5e-2,1e-2,1e-3,1e-4,1e-5,1e-6]:
        for t in (tf-eps,):
            row,_,q=evaluate(c,s,t,Z,asym,previous)
            a=row['A_U_YN'];row_stage=row['A_C_safety_YN'];row_NN=row['A_C_safety_NN']
            probes.append({'probe_kind':'cutoff_boundary_numeric_diagnostic','t_probe_s':float(t),
                'epsilon_s':float(eps),'t_reported_s':tf,'stage':3,
                'pA_Y':q['pA_Y'],'pB_Y':q['pB_Y'],'pNN':q['pNN'],
                'A_C_safety_YN':row_stage,'A_C_safety_NN':row_NN,
                'A_U_YN':a,'A_U_NN':row['A_U_NN'],
                'safety_form':safety_loss_form(c),'sigma_p':sigma_of(c)[0],'Pref':float(value(c,'collision_probability','probability_reference')),
                'safety_margin_m':list(c['implementation']['safety_margin_m']),
                'I_effective_yield_history':history_indicator(events or [],tf)})
    span=[p['pNN'] for p in probes]
    return {'probe_kind':'cutoff_boundary_numeric_diagnostic','probes':probes,
            'pNN_span':max(span)-min(span),'pNN_min':min(span),'pNN_max':max(span),
            'interpretation':('numerical diagnostic only: jump across tf is attributed to the safety-loss '
                              'saturation clip and to min_q_safe approaching the ellipsoid boundary; '
                              'NOT a mathematical left limit of the stage mean'),
            'physical_probability_state':'nominal_NN_current_state'}

def remaining_stage_report(rows):
    return [{'t_s':r['t_s'],'stage':r['stage'],'I':r['I_effective_yield_history'],'P_three_stage_dynamic':r['P_three_stage_dynamic'],'R_remaining_stage_index':r['R_remaining_stage_index'],'instant_pA_Y':r['pA_Y'],'instant_pB_Y':r['pB_Y'],'instant_pNN':r['pNN'],'interpretation':'macro-stage conditional index; current mean uses remaining forecast window starting NOW, not history; no claimed exact continuous-time survival probability'} for r in rows]

def effective_execution_confirmation(c,profile,pair):
    """Audit proxy: an actually changed heading + safe current-velocity forecast,
    while the planned full standard trajectory/rejoin also passes validation.
    No claim that future telemetry has already been observed.
    """
    if profile['min_q_safe']<=1 or not profile['recovery_completed']:return None
    axes=np.array(value(c,'geometry','combined_collision_half_axes_m'))+c['implementation']['safety_margin_m']
    times=profile['times'];H=float(value(c,'collision_probability','prediction_horizon_s'))
    delay=float(value(c,'delay','execution_delay_baseline_s'))
    for k in range(1,len(times)-1):
        if times[k]<=delay+1e-9:continue
        executed_change=any(abs(profile['omega'+i][k])>1e-12 for i,act in zip('AB',pair.split(',')) if act=='Y')
        if not executed_change:continue
        h=times[k+1]-times[k]
        vr=((profile['pB'][k+1]-profile['pB'][k])-(profile['pA'][k+1]-profile['pA'][k]))/h
        rr=profile['rel'][k]
        if segment_min_q(np.array([rr,rr+H*vr]),profile['basis'],axes)>1:return float(times[k])
    return None

def yield_history_audit(c,s,b,Z):
    """Validate truncation independently from physical trajectory probability."""
    command=b['t_f_s'];safe=trajectory(c,at_time(s,command),'Y,N')
    early_command=b['t_c_s']+5.;early=trajectory(c,at_time(s,early_command),'Y,N')
    late_command=s['initial_tcpa']-.2;late=trajectory(c,at_time(s,late_command),'Y,N')
    nn=trajectory(c,at_time(s,command),'N,N');cases=[]
    for label,executed,effective,profile,cmd in [
        ('intention_only',False,False,nn,command),
        ('command_pending_or_dropped',False,False,nn,command),
        ('executed_but_unsafe_late_Y',True,False,late,late_command),
        ('early_executed_effective_Y',True,early['min_q_safe']>1 and early['recovery_completed'],early,early_command),
        ('executed_and_nominally_effective_Y',True,safe['min_q_safe']>1 and safe['recovery_completed'],safe,command)]:
        confirmation_tau=effective_execution_confirmation(c,profile,'Y,N') if effective else None
        effective=bool(effective and confirmation_tau is not None)
        confirm=cmd+confirmation_tau if effective else cmd
        event={'time_s':float(confirm),'player':'A','action':'Y','executed':bool(executed),'effective':bool(effective)}
        flag=history_indicator([event],confirm);sp=float(value(c,'uncertainty','baseline_position_sigma_m'));sv=float(value(c,'uncertainty','baseline_velocity_sigma_mps'))
        p,lo,hi,tp=probability(c,profile,sp,sv,Z)
        cases.append({'case':label,'command_s':cmd,'confirmation_or_audit_s':confirm,'executed':executed,'nominally_effective':bool(effective),'I_after_confirmation':flag,'min_q_safe':profile['min_q_safe'],'Pcol_max_over_commanded_trajectory':p,'Pcol_CI_low':lo,'Pcol_CI_high':hi,'Pcol_clock_note':'peak over the independently simulated commanded trajectory, not multiplied by I'})
    return cases


def boundary_case(c,s,Z=None):
    """Primary boundary: deterministic geometry criterion (v6 rule).
    The probability-acceptance boundary is searched separately and reported in
    parallel; it is NOT silently merged, because Pref (normalisation scale) and
    P_max (acceptance threshold) are distinct quantities.
    """
    searches=[search_boundary(c,s,p,mode='geometry') for p in ('Y,N','N,Y','Y,Y')]
    if any(x['last'] is None for x in searches[:2]):return searches,None,None
    primary=min(x['last']['command_time_s'] for x in searches[:2])
    alt=None
    if Z is not None:
        prob=[search_boundary(c,s,p,mode='probability',Z=Z) for p in ('Y,N','N,Y','Y,Y')]
        if all(x['last'] is not None for x in prob[:2]):
            alt=min(x['last']['command_time_s'] for x in prob[:2])
    return searches,primary,alt

def plots(out,rows,opp,b,asym,eff,delay,terminal=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def lines(ax):
        for x in (b['t1_end_s'],b['t2_end_s'],b['t_f_s']):ax.axvline(x,color='gray',ls=':',lw=1)
        ax.grid(alpha=.2)
    t=[r['t_s'] for r in rows]
    fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True)
    for k,label in [('pA_Y','A: P(Y)'),('pB_Y','B: P(Y)'),('pNN','instant P(N,N)')]:axs[0].plot(t,[r[k] for r in rows],label=label)
    axs[0].set_ylabel('Instant behavior');axs[0].set_ylim(0,1)
    axs[1].plot(t,[r['P_three_stage_dynamic'] for r in rows],label='Online: product of 3 individual-mean stage factors')
    axs[1].plot(t,[r['comparison_v5_full_stage_P_three_stage'] for r in rows],label='v5 full-region comparison',color='gray',ls='--')
    axs[1].set_ylabel('Three-stage risk index')
    axs[2].plot(t,[r['R_remaining_stage_index'] for r in opp],label='Remaining macro-stage conditional index')
    axs[2].plot(t,[r['instant_pA_Y'] for r in opp],label='A: P(Y)',ls='--');axs[2].set_ylabel('Remaining stage index');axs[2].set_ylim(0,1)
    axs[2].set_xlabel('Absolute encounter time (s)')
    if terminal is not None:
        # Q10: the boundary jump is reported numerically in
        # cutoff_boundary_numeric_diagnostic.*; it is NOT drawn as a risk point
        # and no longer carries the 'left limit' name or claim.
        pass
    for ax in axs:lines(ax);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'01_dynamic_three_stage_and_remaining_index.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axs[0].plot(t,[r['P_three_stage_dynamic'] for r in rows],label='v6 current remaining-window mean')
    axs[0].plot(t,[r['comparison_v5_full_stage_P_three_stage'] for r in rows],ls='--',color='gray',label='v5 full-stage mean comparison')
    axs[0].set_ylabel('Three-stage index (regular monitoring)');axs[0].legend(fontsize=9)
    axs[1].step(t,[r[f"stage{int(r['stage'])}_sample_count"] for r in rows],where='post',label='Current stage remaining grid count')
    axs[1].set_ylabel('Remaining samples');axs[1].set_xlabel('Encounter time (s)');axs[1].legend(fontsize=9)
    for ax in axs:lines(ax)
    fig.tight_layout();fig.savefig(out/'06_remaining_window_detail.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for k in ('pA_Y','pB_Y','pNN'):ax.plot([r['t_s'] for r in asym],[r[k] for r in asym],label=k)
    lines(ax);ax.set(xlabel='Time (s)',ylabel='Asymmetric perception behavior',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/'02_asymmetric_perception.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for st in (1,2,3):
        z=[r for r in eff if r['stage']==st];ax.plot([r['efficiency_weight'] for r in z],[r['pNN'] for r in z],marker='o',label=f'stage {st}')
    ax.set(xlabel='Efficiency weight',ylabel='P(N,N)');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'03_efficiency_sensitivity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot([r['delay_s'] for r in delay],[r['geometry_t_f_s'] for r in delay],marker='o',label='geometry criterion');ax.plot([r['delay_s'] for r in delay],[r['probability_acceptance_t_f_s'] for r in delay],marker='s',ls='--',label='probability-acceptance criterion');ax.legend(fontsize=8);ax.set(xlabel='Execution delay (s)',ylabel='Last feasible command (s)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'04_delay_boundary.png',dpi=180);plt.close(fig)


def run(c,out,mode='full',events=None):
    out.mkdir(parents=True,exist_ok=True);s=scenario(c)
    seed=int(value(c,'collision_probability','random_seed'));Z=np.random.default_rng(seed).standard_normal((int(value(c,'collision_probability','mc_samples')),3))
    sp0,sv0=sigma_of(c);form0=safety_loss_form(c)
    print('Searching full trajectories and merge-point recovery...',flush=True)
    print(f'  sigma_p={sp0} m  sigma_v={sv0} m/s  safety_loss_form={form0}',flush=True)
    searches,last,prob_last=boundary_case(c,s,Z)
    if last is None:raise RuntimeError('Standard Y not feasible for both unilateral pairs')
    b=stage_bounds(c,s,last);print('Boundaries:',b,flush=True)
    if prob_last is not None:
        print(f"  probability-acceptance comparison track (P_max={c['implementation']['probability_acceptance_threshold']}): t_f={prob_last} vs geometry t_f={last}",flush=True)
    rows,stages,terminal=dynamic(c,s,b,Z,events=events);print('Baseline rolling three-stage means completed.',flush=True)
    write_csv(out/'cutoff_boundary_numeric_diagnostic.csv',terminal['probes'])
    dump(out/'cutoff_boundary_numeric_diagnostic.json',terminal)
    opp=remaining_stage_report(rows)
    asym,_,_=dynamic(c,s,b,Z,True,events=events)
    history_audit=yield_history_audit(c,s,b,Z)
    write_csv(out/'effective_yield_audit.csv',history_audit)
    split_keys=('observed_sample_count','forecast_sample_count','observed_A_mean_N','observed_B_mean_N','forecast_A_mean_N','forecast_B_mean_N','window_start_s','window_end_s','excluded_history_count')
    factor_audit=[{'t_s':r['t_s'],'current_stage':r['stage'],'factor_stage':k,'A_mean_N':r[f'stage{k}_A_mean_N'],'B_mean_N':r[f'stage{k}_B_mean_N'],'joint_mean_product':r[f'stage{k}_q_mean_product'],'sample_count':r[f'stage{k}_sample_count'],'source':r[f'stage{k}_provenance'],**{key:r[f'stage{k}_{key}'] for key in split_keys}} for r in rows for k in (1,2,3)]
    write_csv(out/'stage_factor_provenance.csv',factor_audit)
    write_csv(out/'current_stage_remaining_mean_audit.csv',[r for r in factor_audit if r['current_stage']==r['factor_stage']])
    ex_times=sorted({rows[0]['t_s'],rows[len(rows)//3]['t_s'],rows[-1]['t_s']})
    write_csv(out/'stage_factor_examples.csv',[r for r in factor_audit if r['t_s'] in ex_times])
    audit_out=[]
    representative={k:[r for r in rows if r['stage']==k][len([r for r in rows if r['stage']==k])//2]['t_s'] for k in (1,2,3)}

    # ---- Q1 track A vs track B: identical scenario/params/trajectories/sampling,
    # only the safety-loss form differs. Reported side by side. No selection is
    # made on whether the resulting curve rises (Q1 acceptance criterion).
    print('Safety-loss form comparison (same scenario/params/sampling)...',flush=True)
    form_cmp=[];form_err=[]
    for fm in SAFETY_FORMS:
        cc=copy.deepcopy(c);cc['implementation']['safety_loss_form']=fm
        try:
            rr,_,_=dynamic(cc,s,b,Z,events=events)
            for r1 in rr:
                form_cmp.append({'safety_loss_form':fm,'t_s':r1['t_s'],'stage':r1['stage'],
                    'P_three_stage_dynamic':r1['P_three_stage_dynamic'],
                    'pA_Y':r1['pA_Y'],'pB_Y':r1['pB_Y'],'pNN':r1['pNN'],
                    'A_Pcol_max_NN':r1['A_Pcol_max_NN'],'A_Pcol_max_YN':r1['A_Pcol_max_YN'],
                    'A_C_safety_NN':r1['A_C_safety_NN'],'A_C_safety_YN':r1['A_C_safety_YN'],
                    'A_U_YY':r1['A_U_YY'],'A_U_YN':r1['A_U_YN'],'A_U_NY':r1['A_U_NY'],'A_U_NN':r1['A_U_NN'],
                    'A_C_efficiency_YN':r1['A_C_efficiency_YN']})
        except Exception as exc:
            form_err.append({'safety_loss_form':fm,'error':repr(exc)})
    write_csv(out/'safety_form_comparison.csv',form_cmp)
    if form_err:write_csv(out/'safety_form_comparison_errors.csv',form_err)

    # ---- Q4: sigma_p dual value {0.1 original, 0.5 explicit assumption} crossed
    # with safety form. 0.5 m is NOT presented as calibrated (Q4).
    print('sigma_p x safety-form factorial...',flush=True)
    sig_cmp=[]
    for spv in value(c,'implementation','sigma_p_comparison_m'):
        for fm in ('saturating','continuous'):
            cc=copy.deepcopy(c)
            cc['uncertainty']['baseline_position_sigma_m']['value']=spv
            cc['implementation']['safety_loss_form']=fm
            for t in (representative[1],representative[2],representative[3]):
                r,ev,_=evaluate(cc,s,t,Z);A=ev['A']
                sig_cmp.append({'sigma_p_m':spv,
                    'sigma_role':('v3_original_parameter' if abs(spv-sp0)<1e-12 else 'explicit_assumption_not_calibrated'),
                    'safety_loss_form':fm,'time_s':t,
                    'Pcol_NN_A':A['N,N']['Pcol_max'],'Pcol_YN_A':A['Y,N']['Pcol_max'],
                    'A_C_safety_NN':A['N,N']['C_safety'],'A_C_safety_YN':A['Y,N']['C_safety'],
                    'A_C_efficiency_YN':A['Y,N']['C_efficiency'],
                    'A_U_YY':A['Y,Y']['U'],'A_U_YN':A['Y,N']['U'],'A_U_NY':A['N,Y']['U'],'A_U_NN':A['N,N']['U'],
                    'pA_Y':r['pA_Y'],'pB_Y':r['pB_Y'],'pNN':r['pNN'],
                    'root_count_found':r['root_count_found'],'residual':r['residual']})
    write_csv(out/'sigma_factorial_comparison.csv',sig_cmp)

    # ---- Q2: single-factor asymmetric scenario. ONLY B execution delay differs;
    # every other condition identical to the symmetric baseline.
    print('Single-factor asymmetric scenario (B execution delay only)...',flush=True)
    asym_scen=[]
    for dv in value(c,'implementation','asymmetric_delay_probe_s'):
        cc=copy.deepcopy(c)
        cc['delay']['execution_delay_B_s']={'value':dv,'status':'explicit_assumption_single_factor_probe',
            'note':'B execution delay only; all other conditions identical to the symmetric baseline'}
        sc=scenario(cc);se,lt,pl=boundary_case(cc,sc,Z)
        if lt is None:
            asym_scen.append({'asymmetric_factor':'B_execution_delay_s','delay_B_s':dv,'feasible':False});continue
        bb=stage_bounds(cc,sc,lt);rr,_,_=dynamic(cc,sc,bb,Z)
        rmid=min(rr,key=lambda x:abs(x['t_s']-(bb['t_f_s']-0.1)))
        asym_scen.append({'asymmetric_factor':'B_execution_delay_s','delay_B_s':dv,'feasible':True,
            'scenario_basis':'explicit assumption: unilateral execution-delay asymmetry, single factor',
            'geometry_t_f_s':lt,'probability_acceptance_t_f_s':pl,'t_c_s':bb['t_c_s'],
            'time_s':rmid['t_s'],'stage':rmid['stage'],
            'A_C_efficiency_YN':rmid['A_C_efficiency_YN'],'B_C_efficiency_YN':rmid['B_C_efficiency_YN'],
            'A_U_YY':rmid['A_U_YY'],'A_U_YN':rmid['A_U_YN'],'A_U_NY':rmid['A_U_NY'],'A_U_NN':rmid['A_U_NN'],
            'pA_Y':rmid['pA_Y'],'pB_Y':rmid['pB_Y'],'pNN':rmid['pNN'],
            'P_three_stage_dynamic':rmid['P_three_stage_dynamic'],
            'stage1_q':rmid['stage1_q_mean_product'],'stage2_q':rmid['stage2_q_mean_product'],'stage3_q':rmid['stage3_q_mean_product']})
    write_csv(out/'asymmetric_single_factor.csv',asym_scen)

    # Baseline asymmetric (perception-bias) track retained from v6 for continuity.
    for k,t in representative.items():
        _,ev,q=evaluate(c,s,t,Z)
        for i in 'AB':
            for pair,m in ev[i].items():audit_out.append({'stage':k,'time_s':t,'player':i,'pair':pair,**m})
    eff=[]
    for wE in value(c,'utility','efficiency_weight_sweep'):
        cc=copy.deepcopy(c);cc['utility']['efficiency_weight']['value']=wE;cc['utility']['safety_weight']['value']=1-wE
        for k,t in representative.items():
            r,_,_=evaluate(cc,s,t,Z);ev=outcomes(cc,at_time(s,t),Z)
            eff.append({'efficiency_weight':wE,'safety_weight':1-wE,
                'is_baseline_weight':bool(abs(wE-float(value(c,'utility','efficiency_weight')))<1e-12),
                'stage':k,'time_s':t,'pA_Y':r['pA_Y'],'pB_Y':r['pB_Y'],'pNN':r['pNN'],
                'safety_term_contribution_A_NN':float(value(cc,'utility','safety_weight'))*ev['A']['N,N']['C_safety'],
                'efficiency_term_contribution_A_YN':float(value(cc,'utility','efficiency_weight'))*ev['A','Y,N']['C_efficiency'] if False else float(value(cc,'utility','efficiency_weight'))*ev['A']['Y,N']['C_efficiency'],
                'actual_utility_spread_A':max(x['U'] for x in ev['A'].values())-min(x['U'] for x in ev['A'].values())})
    delay=[];geometry=[];unc=[];lambdas=[];pgeom=[]
    if mode=='full':
        print('Running delay, geometry, horizon, error, and rationality sweeps...',flush=True)
        for d in value(c,'delay','execution_delay_sweep_s'):
            cc=copy.deepcopy(c);cc['delay']['execution_delay_baseline_s']['value']=d
            ss,tl,pl=boundary_case(cc,s,Z)
            t=tl-1e-3 if tl is not None else 0.;r,_,_=evaluate(cc,s,t,Z)
            delay.append({'delay_s':d,'geometry_t_f_s':tl,'probability_acceptance_t_f_s':pl,'evaluation_time_s':t,'pNN_before_boundary':r['pNN']})
        for T in value(c,'scenario','tcpa_sweep_s'):
            for D in value(c,'scenario','dcpa_sweep_m'):
                sc=scenario(c,T,D);st=at_time(sc,max(0,T-float(value(c,'collision_probability','prediction_horizon_s'))));g,_,dc,mq=gate(c,st)
                tl=None;qv=None;pl=None
                if g:
                    _,tl,pl=boundary_case(c,sc,Z)
                    if tl is not None:qv=evaluate(c,sc,tl-1e-3,Z)[0]['pNN']
                geometry.append({'TCPA_target_s':T,'horizontal_DCPA_target_m':D,'heading_B_deg':value(c,'scenario','heading_B_deg'),'vertical_offset_m':value(c,'scenario','vertical_offset_m'),'recalculated_3D_DCPA_m':dc,'potential_conflict_at_gate':g,'min_q_safe':mq,'geometry_t_f_s':tl,'probability_acceptance_t_f_s':pl,'pNN_before_boundary':qv})
        for ang,zz in [(90.,0.),(150.,0.),(180.,2.)]:
            sc=scenario(c,angle=ang,z=zz);gg=gate(c,at_time(sc,max(0,sc['initial_tcpa']-20)))
            tl=None;pl=None
            if gg[0]:_,tl,pl=boundary_case(c,sc,Z)
            geometry.append({'TCPA_target_s':sc['initial_tcpa'],'horizontal_DCPA_target_m':0.,'heading_B_deg':ang,'vertical_offset_m':zz,'recalculated_3D_DCPA_m':gg[2],'potential_conflict_at_gate':gg[0],'min_q_safe':gg[3],'geometry_t_f_s':tl,'probability_acceptance_t_f_s':pl,'pNN_before_boundary':None})
        # Q7: prediction-horizon sensitivity with the stage start held FIXED, so
        # this is a pure horizon sweep and can no longer move t_c.
        for Hv in value(c,'implementation','prediction_horizon_sensitivity_s'):
            cc=copy.deepcopy(c);cc['collision_probability']['prediction_horizon_s']['value']=Hv
            r0,_,_=evaluate(cc,s,b['t_c_s'],Z)
            pgeom.append({'prediction_horizon_s':Hv,'t_c_s':b['t_c_s'],'t_c_rule':b['t_c_rule'],
                'Pcol_NN_A_at_t_c':r0['A_Pcol_max_NN'],'C_safety_NN_at_t_c':r0['A_C_safety_NN'],
                'pA_Y_at_t_c':r0['pA_Y'],'pNN_at_t_c':r0['pNN'],
                'note':'t_c held fixed; H no longer determines the stage start'})
        write_csv(out/'prediction_horizon_sensitivity.csv',pgeom)
        for typ,vals in [('position',value(c,'uncertainty','position_sigma_sweep_m')),('velocity',value(c,'uncertainty','velocity_sigma_sweep_mps'))]:
            for sigma in vals:
                cc=copy.deepcopy(c);cc['uncertainty']['baseline_'+typ+'_sigma_'+('m' if typ=='position' else 'mps')]['value']=sigma
                r,_,_=evaluate(cc,s,representative[3],Z)
                unc.append({'varied_error':typ,'sigma':sigma,'evaluation_time_s':representative[3],
                    'pA_Y':r['pA_Y'],'pB_Y':r['pB_Y'],'pNN':r['pNN'],
                    'Pcol_NN_A':r['A_Pcol_max_NN'],'C_safety_NN_A':r['A_C_safety_NN'],'C_safety_YN_A':r['A_C_safety_YN'],
                    'note':'error-magnitude sweep only; velocity sigma is NOT equivalent to information asymmetry (Q4)'})
        # Q11: report every candidate root, the search method, and a forward/
        # reverse scan cross-check. Roots are CANDIDATES, not certified equilibria.
        for lam in (0.,1.,2.,4.,6.,8.):
            cc=copy.deepcopy(c);cc['qre']['lambda_A']['value']=lam;cc['qre']['lambda_B']['value']=lam
            r,ev,qq=evaluate(cc,s,representative[3],Z)
            A={k:v['U'] for k,v in ev['A'].items()};B={k:v['U'] for k,v in ev['B'].items()}
            roots,FA,FB,meth=_qre_roots(cc,A,B)
            f=lambda x: FA(FB(x))-x
            xx=np.linspace(0.,1.,2001);ff=np.array([f(x) for x in xx])
            fwd=int(np.sum(ff[:-1]*ff[1:]<0));rev=int(np.sum(ff[1:][::-1]*ff[:-1][::-1]<0))
            lambdas.append({'lambda':lam,'evaluation_time_s':representative[3],
                'pA_Y':r['pA_Y'],'pB_Y':r['pB_Y'],'pNN':r['pNN'],
                'candidate_root_count':r['root_count_found'],'residual':r['residual'],
                'all_candidate_roots_x_y':str([[round(a,10),round(b,10)] for a,b in roots]),
                'root_search_method':meth,'forward_scan_sign_changes':fwd,'reverse_scan_sign_changes':rev,
                'forward_reverse_consistent':bool(fwd==rev),
                'root_selection_rule':'closest to previous iterate under (x,FB(x)) distance; cold-start reference (0.5,0.5)',
                'note':'candidate roots from a 2001-point grid plus tangent refinement; NOT a proof that all equilibria were captured'})
    else:
        delay=[{'delay_s':value(c,'delay','execution_delay_baseline_s'),'geometry_t_f_s':last,'probability_acceptance_t_f_s':prob_last,'evaluation_time_s':rows[-1]['t_s'],'pNN_before_boundary':rows[-1]['pNN']}]
    turntime=float(value(c,'yield_action','baseline_heading_change_deg'))/float(value(c,'yield_action','baseline_turn_rate_degps'))
    component_audit=[]
    for ss in searches:
        tc=ss['last']['command_time_s'] if ss['last'] else None
        tr=trajectory(c,at_time(s,tc),ss['pair']) if tc is not None else None
        Td=delay_of(c,'A')
        form=max(0.,tr['recovery_time_s']-Td-turntime) if tr and tr['recovery_time_s'] is not None else None
        Taction=Td+turntime+form if form is not None else None
        component_audit.append({'pair':ss['pair'],'latest_feasible_grid_command_s':tc,'Td_s':Td,'Tturn_s':turntime,'Thold_to_separation_gate_s':form,'T_action_to_separation_gate_s':Taction,'boundary_margin_DeltaT_f_s':s['initial_tcpa']-tc if tc is not None else None,'deprecated_component_subtraction_comparison_s':s['initial_tcpa']-Taction if Taction is not None else None,'interpretation':'component decomposition and strict deadline are separate report quantities; equivalent deadline margin is not the full maneuver duration'})
    write_csv(out/'component_time_audit.csv',component_audit)
    print('Numerical resolution checks...',flush=True)
    refinement=[]
    for step in (.1,.05,.025):
        cc=copy.deepcopy(c);cc['numerics']['integration_dt_s']['value']=step
        for pair in ('Y,N','N,Y','Y,Y'):
            search=next(x for x in searches if x['pair']==pair);t=search['last']['command_time_s']
            res=trajectory(cc,at_time(s,t),pair)
            refinement.append({'integration_dt_s':step,'pair':pair,'command_time_s':t,'min_q_safe':res['min_q_safe'],'min_distance_m':res['min_distance_m'],'completed':res['recovery_completed'],'feasible_at_baseline_grid_boundary':res['min_q_safe']>1 and res['recovery_completed']})
    mc=[]
    for n,sd in [(1024,seed),(4096,seed),(4096,seed+1)]:
        ZZ=np.random.default_rng(sd).standard_normal((n,3));r,ev,_=evaluate(c,s,representative[3],ZZ)
        mc.append({'samples':n,'seed':sd,'time_s':representative[3],'pNN':r['pNN'],'Pcol_NN_A':ev['A']['N,N']['Pcol_max'],'Pcol_NN_peak_pointwise_CI_high':ev['A']['N,N']['Pcol_pointwise_CI_high']})
    for name,data in [('dynamic_qre',rows),('stage_summary',stages),('conditional_remaining_stage_index',opp),('asymmetric_dynamic_qre',asym),('payoff_matrices',audit_out),('boundary_search',[r for ss in searches for r in ss['records']]),('efficiency_sweep',eff),('delay_sweep',delay),('geometry_sweep',geometry),('uncertainty_sweep',unc),('lambda_sweep',lambdas),('integration_resolution',refinement),('mc_repeats',mc)]:write_csv(out/(name+'.csv'),data)
    t=representative[3];st=at_time(s,t)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,pair in zip(axs.flat,PAIRS):
        res=trajectory(c,st,pair);stride=int(value(c,'numerics','trajectory_output_stride'))
        rr=[{'tau_s':float(res['times'][k]),'A_x':float(res['pA'][k,0]),'A_y':float(res['pA'][k,1]),'A_z':float(res['pA'][k,2]),'B_x':float(res['pB'][k,0]),'B_y':float(res['pB'][k,1]),'B_z':float(res['pB'][k,2]),'q_collision':float(np.sum((res['rel'][k]@st['basis'].T/np.array(value(c,'geometry','combined_collision_half_axes_m')))**2))} for k in range(0,len(res['times']),stride)]
        write_csv(out/('trajectory_'+pair.replace(',','')+'.csv'),rr)
        ya=np.where(abs(res['pA'][:,1])<1e-10,0.,res['pA'][:,1]);yb=np.where(abs(res['pB'][:,1])<1e-10,0.,res['pB'][:,1])
        ax.plot(res['pA'][:,0],ya,label='A');ax.plot(res['pB'][:,0],yb,label='B')
        ax.scatter([st['qA'][0],st['qB'][0]],[st['qA'][1],st['qB'][1]],marker='x',label='individual merge points')
        ax.set_title(pair+f"; min ellipsoid q={res['min_q_collision']:.3f}");ax.set(xlabel='x (m)',ylabel='y (m)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        if pair=='N,N':ax.set_ylim(-1.,1.)
    fig.tight_layout();fig.savefig(out/'05_four_full_trajectories.png',dpi=180);plt.close(fig)
    plots(out,rows,opp,b,asym,eff,delay,None)
    sm=[x['mean_A_N'] for x in stages]
    summary={
     'model':'v7 working revision: individual stage means -> player combination -> three-stage product (v6 aggregation retained)',
     'model_version':c['implementation']['version'],'mode':mode,
     'run_configuration':{'sigma_p_m':sp0,'sigma_v_mps':sv0,'safety_loss_form':form0,
        'Pref_role':'normalisation_scale_for_safety_loss_NOT_acceptance_threshold',
        'P_max_acceptance_threshold':c['implementation']['probability_acceptance_threshold'],
        'P_max_role':'acceptance_threshold_for_the_probability_feasibility_comparison_track',
        'safety_margin_m':c['implementation']['safety_margin_m'],
        'qre_rationality_lambda_A':value(c,'qre','lambda_A'),'qre_rationality_lambda_B':value(c,'qre','lambda_B'),
        'symbol_note':'mu / A_DCPA / B_TCPA denote the original-paper safety margin and coefficients; lambda_A/lambda_B denote QRE rationality parameters. Distinct symbols by design.'},
     'scenario':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()},
     'boundaries':b,
     'boundary_criterion_primary':'geometry: min_q_safe>1 AND recovery_completed',
     'boundary_geometry_t_f_s':last,
     'boundary_probability_acceptance_t_f_s':prob_last,
     'boundary_note':'the two criteria answer different questions and are reported in parallel; Pref and P_max are distinct concepts even when numerically equal',
     'latest_command_times':{x['pair']:x['last']['command_time_s'] if x['last'] else None for x in searches},
     'monotonicity_on_grid':{x['pair']:x['monotone_on_grid'] for x in searches},
     'stage_individual_mean_products':[x['q_mean_product'] for x in stages],
     'effective_yield_audit':history_audit,'events_supplied':events or [],
     'initial_pregate_QRE':evaluate(c,s,0.,Z)[0],
     'first_in_gate':rows[0],'last_in_gate':rows[-1],
     'max_QRE_residual':max(r['residual'] for r in rows+asym),
     'P_three_stage_last_regular_monitor':rows[-1]['P_three_stage_dynamic'],
     'last_regular_monitor_s':rows[-1]['t_s'],
     'P_three_stage_full_history_retrospective':rows[-1]['P_three_stage_final_retrospective'],
     'R_remaining_stage_index_last':rows[-1]['R_remaining_stage_index'],
     'cutoff_boundary_numeric_diagnostic':terminal,
     'cutoff_is_not_a_risk_result':'Q10: the former "mathematical left limit" claim and naming are removed; numerical boundary diagnostic only',
     'dynamic_three_stage_range':{'min':min(r['P_three_stage_dynamic'] for r in rows),'max':max(r['P_three_stage_dynamic'] for r in rows)},
     'online_window_notes':'Current stage averages NOW plus predictions strictly after NOW until stage end. Earlier current-stage observations are excluded; denominator is remaining grid count. Completed stages keep historical full means, future stages keep full forecast means. Monitoring is not independent actual decision trials.',
     'stage_distinction_diagnostic':{'definition':'relative spread of stage individual means = (max-min)/mean',
        'relative_spread':float((max(sm)-min(sm))/max(np.mean(sm),1e-12)),
        'interpretation':'a small spread is a DIAGNOSTIC finding about the present calibration, NOT a verdict of model failure (Q8)'},
     'supplements':c['implementation'],
     'mc_notes':'95% Wilson intervals are pointwise at MC-selected maximum; not simultaneous confidence intervals for max_tau. Finite 0-hit estimates are not zero-risk guarantees.',
     'probability_event_notes':'Primary = macro-stage means product, not independent monitoring trials. R_remaining_stage_index is a macro-stage conditional index, not an exact continuous-time survival probability. Future factors come from a labeled NN model forecast, not future observed samples. No uniform-one-decision assumption in primary model.',
     'component_time_audit':component_audit,
     'boundary_time_precision_s':float(value(c,'safety_boundary','latest_command_search_step_s')),
     'energy_note':'Q9: energy/fuel cost is NOT implemented in this revision. No such interface exists; the earlier "interface reserved" wording was inaccurate and has been removed.'}
    dump(out/'summary.json',summary);return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path(__file__).with_name('params_fc30_v3.json'))
    ap.add_argument('--supplement',type=Path,default=Path(__file__).with_name('model_supplement.json'))
    ap.add_argument('--out',type=Path,default=Path(__file__).with_name('results'))
    ap.add_argument('--mode',choices=['full','baseline'],default='full')
    ap.add_argument('--events-json',type=Path,default=None)
    ap.add_argument('--safety-form',choices=list(SAFETY_FORMS),default=None)
    ap.add_argument('--sigma-p',type=float,default=None)
    args=ap.parse_args()
    c=json.loads(args.config.read_text(encoding='utf-8'));c['implementation']=json.loads(args.supplement.read_text(encoding='utf-8'))
    if args.safety_form:c['implementation']['safety_loss_form']=args.safety_form
    if args.sigma_p is not None:c['uncertainty']['baseline_position_sigma_m']['value']=args.sigma_p
    for k in ('safety_weight','efficiency_weight','efficiency_path_weight','efficiency_smooth_weight'):
        if value(c,'utility',k)<0:raise ValueError('Negative utility weight')
    if abs(value(c,'utility','safety_weight')+value(c,'utility','efficiency_weight')-1)>1e-10:raise ValueError('Safety/efficiency weights must sum to 1')
    if abs(value(c,'utility','efficiency_path_weight')+value(c,'utility','efficiency_smooth_weight')-1)>1e-10:raise ValueError('Efficiency component weights must sum to 1')
    if safety_loss_form(c) not in SAFETY_FORMS:raise ValueError('Unknown safety_loss_form')
    events=json.loads(args.events_json.read_text(encoding='utf-8')) if args.events_json else []
    if not isinstance(events,list):raise ValueError('Events must be a JSON list')
    for e in events:
        if e.get('effective',False) and not e.get('executed',False):raise ValueError('Effective Y must be executed, not intent-only')
        if e.get('action')!='Y' or e.get('player') not in ('A','B'):raise ValueError('Event must identify A/B Yield')
    summary=run(c,args.out,args.mode,events)
    dump(args.out/'parameters_used.json',c)
    print(json.dumps({k:summary[k] for k in ('run_configuration','boundary_geometry_t_f_s','boundary_probability_acceptance_t_f_s','boundaries','max_QRE_residual')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
