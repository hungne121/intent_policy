"""C1 nominal interactive expert preview; cloth is a rigid unfolding proxy, not dataset-ready."""
from pathlib import Path
import argparse
import json
import hashlib
import mujoco
import numpy as np
from layout_gallery import make
from i3_collect import make_env, smooth, geom_sets, JOINTS, DT, TOL
from i3_self_check import I3SelfCheck
from intent_policy.sim.scene_builder_v2 import weld_name
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule
from intent_policy.benchmark.video import VideoWriter
from intent_policy.experts.human_paced import observed_hand_speed, forecast_hand_speed, HumanPace, CartesianFollower, work_speed, WORK_ACCEL
from contact_context import overlap_tolerance
ROOT=Path(__file__).resolve().parents[1]


def run(mode, seed, predictor, out):
    scene, positions, xml=make('C1')
    scene['human']['stand_xy']=[.28,-1.0]
    for key in ('clothL','clothR'):
        positions[key]=(positions[key][0],-.29,positions[key][2])
    from c_preview_visual import connected_human
    xml=connected_human(scene,xml)
    env,human=make_env(scene,positions,xml);m,d=env.model,env.data
    baseJ={k:v.copy() for k,v in human.J.items()};baseP={k:v.copy() for k,v in human.palm_T.items()}
    root0=np.array([.28,-1.,0.]); root=root0.copy()
    order=list(np.random.default_rng(seed).permutation(['L','R']))
    initial={k:np.array(v) for k,v in positions.items()}
    round_i=0;hp='walk';ht=0;hstart=human.hanging_palm(human.active);palm=hstart.copy();root_start=root.copy()
    held=None;human_events=[];events=[];trace=[];history=[];done_cups=[]
    phase='prepare';selected=None;start=d.site_xpos[env.tcp].copy();rt=0;count=1;prev=None
    pace=HumanPace(DT);follower=CartesianFollower(DT,acceleration=WORK_ACCEL)
    peak_speed=0.;forecast=None;observed_root=None;calibration=None
    _,robot,humans,world=geom_sets(m,positions)
    audit=DemoContactAudit(m,[PairRule(a,b,0.,0.) for a in humans for b in robot+world]+[PairRule(a,b,0.,0.) for a in robot for b in world])
    self_check=I3SelfCheck(m);violations={};self_hits={}
    writer=VideoWriter(out/f'C1_{mode.title()}.mp4',width=640,height=480)
    def pos(name):return d.xpos[m.body(name).id].copy()
    def human_stage(name,tick):
        nonlocal hp,ht,hstart,root_start
        hp=name;ht=tick;hstart=palm.copy();root_start=root.copy()
        human_events.append({'event':name,'t':tick*DT,'round':round_i})
    def robot_stage(name,tick):
        nonlocal phase,rt,start,prev
        phase=name;rt=tick;start=d.site_xpos[env.tcp].copy();prev=None
        events.append({'event':name,'t':tick*DT,'cup':selected})
    def grasp_cloth(name):
        eid=m.equality(weld_name(name)).id;anchor=m.eq_obj1id[eid]
        m.eq_data[eid,:3]=0;m.eq_data[eid,3:6]=d.xmat[anchor].reshape(3,3).T@(pos(name)-d.xpos[anchor])
        inv=np.zeros(4);rel=np.zeros(4);mujoco.mju_negQuat(inv,d.xquat[anchor]);mujoco.mju_mulQuat(rel,inv,d.xquat[m.body(name).id]);m.eq_data[eid,6:10]=rel;d.eq_active[eid]=True
    try:
        for tick in range(1600):
            side=order[min(round_i,1)];cup='cup'+side;cloth='cloth'+side;target=initial[cup]
            u=(tick-ht)*DT
            if hp=='walk':
                root=root_start+(np.array([target[0]-.08,-.59,0])-root_start)*smooth(u/(2.4 if round_i==0 else 1.6))
            offset=root-root0
            human.J={k:v+offset for k,v in baseJ.items()};human.palm_T={k:v+offset for k,v in baseP.items()}
            home=human.hanging_palm(human.active)
            if hp=='walk':
                palm=home
                if u>=(2.4 if round_i==0 else 1.6):human_stage('reach_cloth',tick)
            elif hp=='reach_cloth':
                palm=hstart+(pos(cloth)+[0,0,.055]-hstart)*smooth(u/1.0)
                if u>=1.:human_stage('grasp_cloth',tick)
            elif hp=='grasp_cloth':
                if u>=.35 and held is None:grasp_cloth(cloth);held=cloth;human_stage('lift_cloth',tick)
            elif hp=='lift_cloth':
                palm=hstart+(np.array([target[0],-.27,.77])-hstart)*smooth(u/.7)
                if u>=.7:human_stage('wait_cup_clear',tick)
            elif hp=='wait_cup_clear':
                if pos(cup)[2]>.77:human_stage('spread_cloth',tick)
            elif hp=='spread_cloth':
                palm=hstart+(np.array([target[0],target[1],.680])-hstart)*smooth(u/1.2)
                if u>=1.2:
                    d.eq_active[m.equality(weld_name(cloth)).id]=False;held=None
                    human_stage('withdraw',tick)
            elif hp=='withdraw':
                q=min(1.,u/1.0)
                palm=hstart+(home-hstart)*smooth(q)+np.array([0.,0.,.10*np.sin(np.pi*q)])
                if u>=1.:human_stage('wait_cup_return',tick)
            elif hp=='wait_cup_return':
                palm=home
                if np.linalg.norm(pos(cup)[:2]-target[:2])<.03 and pos(cup)[2]<.71 and env.attached is None and np.linalg.norm(d.site_xpos[env.tcp]-pos(cup))>.13:
                    round_i+=1
                    if round_i==2:human_stage('done',tick)
                    else:human_stage('walk',tick)
            else:palm=home
            grasp_strength=smooth(u/.35) if hp=='grasp_cloth' else float(held is not None)
            if hp=='withdraw':grasp_strength=1-smooth(u/.7)
            pose=human.apply(d,palm=palm,hand_shape='grasp',shape_from='relaxed',shape_progress=grasp_strength)
            mujoco.mj_forward(m,d)
            kp=human.keypoints(pose,JOINTS);history=(history+[kp.copy()])[-9:]
            if calibration is None:calibration=kp[:2].mean(axis=0)-root
            observed_root=kp[:2].mean(axis=0)-calibration
            speed=observed_hand_speed(history,5,DT);future_root=None
            if mode=='intent':
                from intent_policy.intent.manicast_intent import forecast_xi
                forecast=forecast_xi(np.stack(history),predictor,20.,np.arange(1,11)/10)[-1]
                future_root=observed_root[None,:]+forecast[:,:2].mean(axis=1)
                speed=max(speed,forecast_hand_speed(forecast,5,np.arange(1,11)/10))
            pace.update(observed_hand_speed(history,5,DT),speed if mode=='intent' else 0.)
            peak_speed=max(peak_speed,speed)
            available=[n for n in ('cupL','cupR') if n not in done_cups]
            if phase in ('prepare','wait_person') and available:
                cue=observed_root if mode=='observed' else np.mean(future_root[-4:],axis=0)
                candidate=min(available,key=lambda n:abs(cue[0]+.08-initial[n][0]))
                if abs(cue[0]+.08-initial[candidate][0])<.065 and cue[1]>-.68:
                    selected=candidate;robot_stage('approach',tick)
            goal=None;grip=.085
            if phase=='prepare':goal=np.array([.28,.02,.84])
            elif phase=='approach':goal=np.r_[pos(selected)[:2],.84]
            elif phase=='descend':goal=np.r_[initial[selected][:2],.688]
            elif phase=='close':goal=np.r_[initial[selected][:2],.688];grip=0.
            elif phase=='lift':goal=np.array([initial[selected][0],initial[selected][1]+.09,.87]);grip=0.
            elif phase=='hold':
                grip=0.;c='cloth'+selected[-1];cp=pos(c)
                if np.linalg.norm(cp[:2]-initial[selected][:2])<.035 and cp[2]<.65 and np.linalg.norm(kp[5]-cp)>.19:robot_stage('replace',tick)
            elif phase=='replace':goal=np.r_[initial[selected][:2],.696];grip=0.
            elif phase=='release':goal=np.r_[initial[selected][:2],.696]
            elif phase=='retreat':goal=np.r_[initial[selected][:2],.85]
            if goal is not None:
                if prev!=phase:
                    prev=phase;count=30 if phase=='close' else 12
                r=min(1.,(tick-rt+1)/count)
                # C1 has no forecast-following motion phase yet: all robot work is independent.
                command=np.r_[env.solve_ik(follower.step(d.site_xpos[env.tcp],goal,work_speed(phase))),grip]
                if phase not in ('close','release'):r=1. if np.linalg.norm(d.site_xpos[env.tcp]-goal)<.004 else 0.
            else:r=0.;command=np.r_[d.qpos[env.qids],0.] if env.attached is not None else env.hold()
            env.step(command);mujoco.mj_forward(m,d)
            if phase=='close':
                if env.attached==m.body(selected).id:robot_stage('lift',tick+1)
                elif tick-rt>70:events.append({'event':'grasp_failed','t':tick*DT});break
            elif goal is not None and r>=1:
                nxt={'prepare':'wait_person','approach':'descend','descend':'close','lift':'hold','replace':'release','release':'retreat','retreat':'wait_person'}[phase]
                if phase=='retreat':done_cups.append(selected)
                robot_stage(nxt,tick+1)
            audit.observe(m,d)
            for idx,dist,_,_ in audit.step_violations:
                pair=audit.rules[idx];ba=m.body(m.geom_bodyid[pair.geom_a]).name;bb=m.body(m.geom_bodyid[pair.geom_b]).name
                # Gripping the held cloth is intentional; all other human contacts remain audited.
                if ba.startswith('human_seg_') and bb==held:continue
                tol=overlap_tolerance(robot_body=ba,other_body=bb,other_geom=m.geom(pair.geom_b).name,attached=None if env.attached is None else m.body(env.attached).name,selection=selected,phase=phase,static_robot=bool(m.body_weldid[m.geom_bodyid[pair.geom_a]]==0),human_pair=ba.startswith('human_seg_'),tolerances=TOL)
                if tol is not None and -dist>tol:violations[f'{ba}|{bb}']=max(violations.get(f'{ba}|{bb}',0.),-float(dist))
            for pair,depth in self_check.measure(d).items():self_hits[str(pair)]=max(self_hits.get(str(pair),0.),float(depth))
            trace.append({'t':tick*DT,'human_phase':hp,'robot_phase':phase,'root_observed':observed_root.tolist(),'palm':palm.tolist(),'tcp':d.site_xpos[env.tcp].tolist(),'selected':selected,'forecast':None if forecast is None else forecast.tolist(),'objects':{k:pos(k).tolist() for k in positions}})
            writer.add(env.render('high',640,480))
            if hp=='done' and len(done_cups)==2:break
        final={k:pos(k).tolist() for k in positions}
        success=round_i==2 and len(done_cups)==2 and all(np.linalg.norm(pos('cloth'+s)[:2]-initial['cup'+s][:2])<.035 and np.linalg.norm(pos('cup'+s)[:2]-initial['cup'+s][:2])<.03 for s in ('L','R'))
        result={'task':'C1','case':'nominal','mode':mode,'seed':seed,'success':bool(success),'duration_s':(tick+1)*DT,'events':events,'human_events':human_events,'final_positions':final,'contact_violations':violations,'human_self_violations':self_hits,'dataset_accepted':False,'limitations':['scripted human root/hand trajectory, not HY-Motion','rigid cloth proxy; no fabric unfolding physics','control-frame contact checks only, swept audit pending','expert preview, not trained ACT'],'forecast_source':'ManiCast actual checkpoint' if mode=='intent' else 'none'}
        (out/f'C1_{mode.title()}_summary.json').write_text(json.dumps(result,indent=2));(out/f'C1_{mode.title()}_trace.json').write_text(json.dumps(trace));print(mode,success,result['duration_s'],len(violations),flush=True)
    finally:writer.close();env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=300100);p.add_argument('--out',type=Path,default=ROOT/'previews');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    from intent_policy.intent.manicast_intent import load_forecaster
    predictor=load_forecaster()
    for mode in ('observed','intent'):run(mode,a.seed,predictor,a.out)
