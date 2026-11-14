"""V2 I2 one-cake/three-plate paired MuJoCo pilot (scripted expert, not ACT)."""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from layout_gallery import make
from i3_collect import make_env, smooth, smooth_motion_path, motion_deviation, geom_sets, TOL
from i3_self_check import I3SelfCheck
from contact_context import overlap_tolerance
from i3_observed import select_pointed
from intent_policy.benchmark.video import VideoWriter
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule
from intent_policy.sim.motion_bank_v2 import to_scene, palm_path, segment_ranges, warp, sample, clip_id
from intent_policy.experts.human_paced import (observed_hand_speed, forecast_hand_speed,
                                                HumanPace, CartesianFollower, work_speed, WORK_ACCEL)
from v2_preview_common import apply_station

ROOT = Path(__file__).resolve().parents[1]
DT = .05
PLATES = {f'plate{i}': np.array([x, -.11, .64]) for i, x in enumerate((.105, .30, .495))}
JOINTS = ['l_shoulder','r_shoulder','l_elbow','r_elbow','l_wrist','r_wrist']


def human_clip(seed, scene, positions, xml):
    rng = np.random.default_rng(seed)
    chosen = str(rng.choice(list(PLATES)))
    side = 'l' if chosen == 'plate0' else 'r'
    onset = float(rng.uniform(2.3, 2.8))
    src = ROOT/'outputs/hymotion_v2_candidates/hymotion_out/i2_point_plate.npz'
    with np.load(src, allow_pickle=False) as raw:
        keypoints = raw['keypoints3d'][0].copy()
    world = to_scene(keypoints, np.array([*scene['human']['stand_xy'], 0]),
                     scene['human']['facing_deg'], scene['floor_z'])
    path = palm_path(world)
    parts = segment_ranges(path, tails=True)
    a, b = parts[0] if parts else (0, len(path))
    motion = smooth_motion_path(path[a:b])
    env, human = make_env(scene, positions, xml)
    try:
        human.set_active(side)
        home = human.hanging_palm(human.active)
    finally:
        env.close()
    forward = np.array([PLATES[chosen][0], -.29, .86])
    rows = []
    for tick in range(400):
        t = tick*DT
        out = smooth((t-onset)/2.2)
        palm = home + (forward-home)*out
        palm += motion_deviation(motion,out)*.15
        rows.append((palm, PLATES[chosen] if out>.001 else None, out))
    return chosen, onset, rows, {'clip_id':clip_id(keypoints),'source':str(src.relative_to(ROOT)),'arm':side}


def run(mode, seed, scene, positions, xml, clip, predictor, out):
    chosen, onset, human_rows, source = clip
    env, human = make_env(scene, positions, xml)
    # Preserve the contact width when the assistive weld engages. The current
    # ellipsoid/finger contact still exceeds the provisional geometry gate.
    env.hold_squeeze = 0.0
    human.set_active(source['arm'])
    m,d=env.model,env.data
    _,robot_geoms,human_geoms,world_geoms=geom_sets(m,positions)
    audit=DemoContactAudit(m,[PairRule(a,b,0.,0.) for a in human_geoms for b in robot_geoms+world_geoms]
                           +[PairRule(a,b,0.,0.) for a in robot_geoms for b in world_geoms])
    human_self=I3SelfCheck(m)
    contact_violations={};contact_details={};self_violations={}
    source_xyz=np.asarray(positions['cake'],float)
    phase='above_cake'; stage_start=0; start=d.site_xpos[env.tcp].copy()
    selected=None; observed=[]; history=[]; events=[]; frames=[]; last_forecast=None
    ready_samples=0
    predicted=None; withdraw_tick=None; home_palm=human.hanging_palm(human.active)
    wrist_history=[]; peak_hand_speed=0.
    pace=HumanPace(DT); follower=CartesianFollower(DT,acceleration=WORK_ACCEL)
    writer=VideoWriter(out/f'I2_{"Intent" if mode=="intent" else "Observed"}.mp4',width=640,height=480)
    stages=['above_cake','wait_cue','descend','close','lift','wait_target','move_plate','lower_plate','release','retreat','settle']
    goals={'above_cake':(np.r_[source_xyz[:2],.80],30,.085),
           'descend':(np.r_[source_xyz[:2],.667],25,.085),
           'close':(np.r_[source_xyz[:2],.667],20,0.),
           'lift':(np.r_[source_xyz[:2],.80],25,0.)}
    try:
        for tick in range(700):
            t=tick*DT
            palm,point,strength=human_rows[min(tick,len(human_rows)-1)]
            if withdraw_tick is not None:
                fade=smooth((tick-withdraw_tick)*DT/.8)
                palm=palm*(1-fade)+home_palm*fade
                strength*=1-fade
                if strength<.001:point=None
            pose=human.apply(d,palm=palm,point_at=point,
                             hand_shape='point',shape_from='relaxed',shape_progress=strength,aim_blend=strength)
            mujoco.mj_forward(m,d)
            ray=human.keypoints(pose,[source['arm']+'_index_base',source['arm']+'_index_tip'])
            visible=select_pointed(ray[0],ray[1],PLATES,max_distance_m=.065,margin_m=.01)
            observed=(observed+[visible])[-5:]
            if selected is None and len(observed)==5 and observed[0] is not None and all(x==observed[0] for x in observed):
                selected=observed[0];events.append({'t':t,'event':'target_observed','plate':selected})
            forecast=None
            kp=human.keypoints(pose,JOINTS)
            wrist=4 if source['arm']=='l' else 5
            wrist_history=(wrist_history+[kp.copy()])[-9:]
            measured_speed=observed_hand_speed(wrist_history,wrist,DT)
            predicted_speed=0.
            if mode=='intent':
                from intent_policy.intent.manicast_intent import forecast_xi
                history=(history+[kp])[-9:]
                forecast=forecast_xi(np.stack(history),predictor,20.,np.arange(1,11)/10)[-1]
                predicted_speed=forecast_hand_speed(forecast,wrist,np.arange(1,11)/10)
                last_forecast=(kp[wrist]+forecast[-1,wrist]).tolist()
                # Re-evaluate the whole predicted wrist path at every tick; the
                # candidate is preparation only until the observed ray confirms it.
                future=kp[wrist][None,:]+forecast[:,wrist,:]
                # The person stands well behind the plates, so wrist-to-plate
                # Euclidean distance is dominated by depth. Lateral forecast
                # trajectory carries the early left/centre/right cue.
                scores={name:float(np.mean(np.abs(future[-4:,0]-xyz[0])))
                        for name,xyz in PLATES.items()}
                candidate=min(scores,key=scores.get)
                motion=float(np.linalg.norm(kp[wrist]-history[0][wrist]))
                if motion>.018 and scores[candidate]<.12:
                    predicted=candidate
            ready_samples=ready_samples+1 if selected is not None and measured_speed<.045 else 0
            if phase=='wait_cue' and ((selected is not None and ready_samples>=3) or (mode=='intent' and predicted is not None)):
                phase='descend';stage_start=tick;start=d.site_xpos[env.tcp].copy()
                events.append({'t':t,'event':'pick_start','basis':'observed' if selected else 'forecast',
                               'forecast_candidate':predicted})
            if phase=='wait_target' and selected is None and mode=='intent' and predicted is not None:
                # Forecast can position the held cake above the predicted route,
                # but final placement still waits for an observed target.
                guess=PLATES[predicted]
                # Intent-only forecast following: drift toward the predicted plate at human pace.
                dest=np.array([guess[0],-.02,.80])
                early_command=np.r_[env.solve_ik(follower.step(d.site_xpos[env.tcp],dest,pace.speed(.24,following=True))),0.]
            else:early_command=None
            pace.update(measured_speed,predicted_speed)
            peak_hand_speed=max(peak_hand_speed,measured_speed,predicted_speed)
            if phase=='wait_target' and selected is not None and np.linalg.norm(kp[wrist]-PLATES[selected])>.22:
                phase='move_plate';stage_start=tick;start=d.site_xpos[env.tcp].copy()
                events.append({'t':t,'event':'transport_start','plate':selected})
            if phase in goals:
                goal,count,grip=goals[phase]
            elif phase=='move_plate':goal,count,grip=np.r_[PLATES[selected][:2],.80],32,0.
            elif phase=='lower_plate':goal,count,grip=np.r_[PLATES[selected][:2],.674],25,0.
            elif phase=='release':goal,count,grip=np.r_[PLATES[selected][:2],.674],20,.085
            elif phase=='retreat':goal,count,grip=np.r_[PLATES[selected][:2],.80],25,.085
            else:goal=count=grip=None
            if phase=='lift' and mode=='intent' and (selected or predicted) is not None:
                destination=PLATES[selected or predicted]
                goal=np.r_[source_xyz[:2]+.35*(destination[:2]-source_xyz[:2]),.80]
            if goal is not None:
                u=min(1.,(tick-stage_start+1)/count)
                # Human-paced following only for the Intent lift that drifts toward the
                # forecast plate before the pointing is confirmed; all else is independent work.
                following=(mode=='intent' and phase=='lift' and selected is None)
                speed=pace.speed(.24,following=True) if following else work_speed(phase)
                target=follower.step(d.site_xpos[env.tcp],goal,speed)
                command=np.r_[env.solve_ik(target),grip]
                if phase not in ('close','release'):
                    u=1. if np.linalg.norm(d.site_xpos[env.tcp]-goal)<.004 else 0.
            elif early_command is not None:
                command=early_command
            else:
                u=0.;command=env.hold()
            frames.append({'t':t,'phase':phase,'selected':selected,'human_hand':kp[wrist].tolist(),
                           'tcp':d.site_xpos[env.tcp].tolist(),'target':command.tolist(),
                           'forecast_wrist_1s':last_forecast if mode=='intent' else None,
                           'forecast_candidate':predicted,'human_withdraw_tick':withdraw_tick,
                           'observed_hand_speed_m_s':measured_speed,
                           'forecast_hand_speed_m_s':predicted_speed if mode=='intent' else None,
                           'robot_command_speed_m_s':float(np.linalg.norm(follower.velocity)),
                           'human_filtered_speed_m_s':pace.filtered})
            env.step(command)
            if phase in ('lift','move_plate') and env.attached is not None and withdraw_tick is None and strength>.95:
                before=np.linalg.norm(np.asarray(frames[-1]['tcp'])[:2]-PLATES[chosen][:2])
                after=np.linalg.norm(d.site_xpos[env.tcp,:2]-PLATES[chosen][:2])
                if before-after>.001 and d.site_xpos[env.tcp,2]>.72:withdraw_tick=tick+1;events.append({'t':(tick+1)*DT,'event':'human_withdraw_after_robot_direction'})
            mujoco.mj_forward(m,d);audit.observe(m,d)
            for index,distance,_,_ in audit.step_violations:
                pair=audit.rules[index];ga,gb=pair.geom_a,pair.geom_b
                ba=m.body(m.geom_bodyid[ga]).name;bb=m.body(m.geom_bodyid[gb]).name
                tol=overlap_tolerance(robot_body=ba,other_body=bb,other_geom=m.geom(gb).name,
                    attached=None if env.attached is None else m.body(env.attached).name,
                    selection='cake',phase=phase,
                    static_robot=bool(m.body_weldid[m.geom_bodyid[ga]]==0),
                    human_pair=ba.startswith('human_seg_'),tolerances=TOL)
                if tol is not None and -distance>tol:
                    key=f'{ba}/{m.geom(ga).name}|{bb}/{m.geom(gb).name}'
                    depth=-float(distance)
                    if depth>contact_violations.get(key,0.):
                        contact_violations[key]=depth
                        contact_details[key]={'t_s':t,'phase':phase,'attached':env.attached is not None,
                                              'gripper_gap_m':float(d.qpos[env.gid])}
            for pair,depth in human_self.measure(d).items():
                self_violations[str(pair)]=max(self_violations.get(str(pair),0.),float(depth))
            writer.add(env.render('high',640,480))
            if goal is not None and u>=1.:
                if phase=='lift':phase='wait_target'
                elif phase=='retreat':phase='settle'
                else:phase=stages[stages.index(phase)+1]
                stage_start=tick+1;start=d.site_xpos[env.tcp].copy()
                events.append({'t':(tick+1)*DT,'event':'stage','stage':phase})
            if phase=='settle' and tick-stage_start>=12:break
        mujoco.mj_forward(m,d)
        cake=d.xpos[m.body('cake').id].copy()
        correct=selected==chosen and selected is not None and np.all(np.abs(cake[:2]-PLATES[chosen][:2])<[.035,.03])
        correct=bool(correct and env.attached is None and .63<cake[2]<.71)
        result={'task':'I2','mode':mode,'seed':seed,'target_evaluator_only':chosen,
                'selected_from_observed_pointing':selected,'human_clip':source,
                'success':correct,'duration_s':round((tick+1)*DT,3),
                'cake_final_xyz':cake.tolist(),'events':events,
                'contact_violations':contact_violations,'contact_violation_details':contact_details,
                'human_self_violations':self_violations,
                'geometry_gate_pass':not contact_violations and not self_violations,
                'limitations':['one-seed scripted expert preview','human path uses retargeted HY-Motion palm deviation',
                               'contact geometry acceptance pending','ACT not trained']}
        (out/f'I2_{"Intent" if mode=="intent" else "Observed"}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/f'I2_{"Intent" if mode=="intent" else "Observed"}_trace.json').write_text(json.dumps(frames)+'\n')
        return result
    finally:
        writer.close();env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=300106)
    p.add_argument('--out',type=Path,default=ROOT/'previews');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    scene,positions,xml=make('I2')
    xml=apply_station(scene,xml)        # shared V2 station (base across the table, skin, side camera)
    clip=human_clip(args.seed,scene,positions,xml)
    from intent_policy.intent.manicast_intent import load_forecaster
    predictor=load_forecaster()
    for mode in ('observed','intent'):
        result=run(mode,args.seed,scene,positions,xml,clip,predictor,args.out)
        print(mode,result['success'],result['duration_s'],flush=True)

if __name__=='__main__':main()
