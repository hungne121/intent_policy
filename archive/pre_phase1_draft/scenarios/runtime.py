"""Four physical HRI tasks; teacher/GT separated from camera policy observations."""
from pathlib import Path
import json
import numpy as np
import mujoco
import yaml
from env.base_env import ManipulationEnv, ROOT

class Scenario:
    def __init__(self,name,seed=0,object_key=None,destination=None,gui=False):
        self.cfg=yaml.safe_load((ROOT/'configs/scenarios/tasks.yaml').read_text())
        self.name=name;self.spec=self.cfg['scenarios'][name];self.seed=seed
        rng=np.random.default_rng(seed)
        self.object_key=object_key or str(rng.choice(self.spec['objects']))
        if self.object_key not in self.spec['objects']:raise ValueError('Object not in scenario')
        zones=self.spec.get('zones',[])
        self.destination=destination or (str(rng.choice(zones)) if zones else None)
        if self.destination is not None and self.destination not in zones:raise ValueError('Invalid destination')
        self.env=ManipulationEnv(gui=gui,model_path=ROOT/f'assets/scenarios/{name}.xml')
        self.env.reset(seed);self.env.receiver_enabled=False
        self.events=[];self.trace=[];self.success=False;self.failure=None;self.stable=0
        self.phase='request';self.phase_time=0.;self.lifted=False;self.grasped=False;self.stop_active=False;self.clear_since=None
        self.intrusion_start=None;self.intrusion_anchor=None;self.intrusion_finished=False;self.safety_stops=0
        self.human_holding=False;self.human_dropped=False;self.human_origin=None;self.drop_time=None
        self.bowl_key=self.cfg['bowl_mapping'].get(self.object_key)
        self.target_body=self.cfg['objects'][self.object_key]['body']
        self.manipulated=self.cfg['bowls'][self.bowl_key]['body'] if name=='bowl_assist' else self.target_body
        # Seeded layouts remain observable; selected object is not always in same slot.
        positions=[self.cfg['objects'][k]['position'] for k in self.spec['objects']]
        rng.shuffle(positions)
        for key,position in zip(self.spec['objects'],positions):
            obj=self.cfg['objects'][key];j=self.env.model.joint(obj['body']+'_free');q=j.qposadr[0]
            xyz=np.array(position,dtype=float);xyz[:2]+=rng.uniform(-.008,.008,2)
            xyz[2]=.62+float(obj['size'][-1])+.003
            self.env.data.qpos[q:q+3]=xyz;self.env.data.qpos[q+3:q+7]=[1,0,0,0]
            self.env.data.qvel[self.env.model.jnt_dofadr[j.id]:self.env.model.jnt_dofadr[j.id]+6]=0
        if name=='bowl_assist':
            # Objects in front of human; robot handles the bowl, not the human-held object.
            for i,key in enumerate(self.spec['objects']):
                q=self.env.model.joint(self.cfg['objects'][key]['body']+'_free').qposadr[0]
                self.env.data.qpos[q:q+2]=[.5,-.07+i*.14]
        self._select_body(self.manipulated)
        self.set_hand([.58,.22,.97])
        self._request_card()
        mujoco.mj_forward(self.env.model,self.env.data)
        hold=self.env.hold()
        for _ in range(15):self.env.step(hold)
        self.env.data.time=0.;self.initial_positions={self.cfg['objects'][k]['body']:self.pos(self.cfg['objects'][k]['body']).copy() for k in self.spec['objects']}
        self.initial_bowls={v['body']:self.pos(v['body']).copy() for v in self.cfg['bowls'].values()} if name=='bowl_assist' else {}
        self.robot_geoms=[]
        root_id=self.env.model.body('robot_mount').id
        for g in range(self.env.model.ngeom):
            b=int(self.env.model.geom_bodyid[g])
            while b>0 and b!=root_id:b=int(self.env.model.body_parentid[b])
            if b==root_id and self.env.model.geom_contype[g]:self.robot_geoms.append(g)
        self.human_geoms=[self.env.model.geom(k).id for k in ('human_palm','human_arm_geom')]
        self.log('episode_start',scenario=name,seed=seed,requested_object=self.object_key,destination=self.destination)

    def _select_body(self,body):
        e=self.env;j=e.model.joint(body+'_free')
        e.cube_id=e.model.body(body).id;e.cube_q=int(j.qposadr[0]);e.cube_d=int(j.dofadr[0])
        e.receiver_eq=e.model.equality('receiver_grasp' if body=='cube' else 'receiver_grasp_'+body).id
    def pos(self,body):return self.env.data.xpos[self.env.model.body(body).id].copy()
    def speed(self,body):
        d=self.env.model.joint(body+'_free').dofadr[0]
        return float(np.linalg.norm(self.env.data.qvel[d:d+3]))
    def log(self,event,**fields):self.events.append(dict(timestamp=float(self.env.data.time),event=event,**fields))
    def set_hand(self,position):
        e=self.env;position=np.array(position)
        e.data.mocap_pos[e.model.body('human_hand').mocapid[0]]=position
        e.data.mocap_pos[e.model.body('human_forearm').mocapid[0]]=position+[.12,0,.015]
    def hand(self):return self.env.data.mocap_pos[self.env.model.body('human_hand').mocapid[0]].copy()
    def _request_card(self):
        e=self.env;obj=self.cfg['objects'][self.object_key]
        for shape in ('box','cylinder','sphere'):
            e.model.geom_rgba[e.model.geom('request_'+shape).id]=obj['color'] if shape==obj['shape'] else [0,0,0,0]
        color=self.cfg['zones'][self.destination]['color'] if self.destination else (self.cfg['bowls'][self.bowl_key]['color'] if self.name=='bowl_assist' else [.9,.65,.45,1])
        e.model.geom_rgba[e.model.geom('request_destination').id]=color
        if self.name in ('intruder','bowl_assist'):
            # No request card reveals the human's object selection in bowl task.
            for k in ('request_board','request_destination','request_box','request_cylinder','request_sphere'):e.model.geom_rgba[e.model.geom(k).id,3]=0
    def attach_to_human(self,body):
        e=self.env;eid=e.model.equality('receiver_grasp' if body=='cube' else 'receiver_grasp_'+body).id
        p=self.pos(body);e.data.mocap_pos[e.receiver_mocap]=p
        e.model.eq_data[eid,3:6]=0;e.model.eq_data[eid,6:10]=e.data.xquat[e.model.body(body).id]
        e.data.eq_active[eid]=True
        self.log('human_acquired',body=body,method='explicit_simulated_hand_weld')
    def target(self):
        if self.name=='handover':return np.array(self.cfg['human']['receive'])
        if self.name=='bowl_assist':return np.array(self.cfg['human']['bowl_delivery'])
        return np.array(self.cfg['zones'][self.destination]['position'])+np.array([0,0,.025])
    def human_update(self):
        t=float(self.env.data.time)
        if self.name=='intruder':
            if self.lifted and self.intrusion_start is None:
                self.intrusion_start=t;self.intrusion_anchor=self.env.observe().ee_pos+np.array([0,.11,.02]);self.log('intrusion_scheduled')
            if self.intrusion_start is not None:
                dt=t-self.intrusion_start;rest=np.array([.6,.3,1.0]);anchor=self.intrusion_anchor
                if dt<1:self.set_hand(rest+(anchor-rest)*dt)
                elif dt<3:self.set_hand(anchor)
                elif dt<4:self.set_hand(anchor+(rest-anchor)*(dt-3))
                else:
                    self.set_hand(rest)
                    if not self.intrusion_finished:self.log('human_withdrawn');self.intrusion_finished=True
            return
        if self.name=='bowl_assist':
            if not self.human_holding and t>=.8:
                self.human_origin=self.pos(self.target_body);self.set_hand(self.human_origin);self.attach_to_human(self.target_body);self.human_holding=True
            if self.human_holding and not self.human_dropped:
                ready=self.target()+[0,0,.16]
                u=min(1.,max(0.,(t-.8)/1.2));p=self.human_origin+(ready-self.human_origin)*u
                self.set_hand(p);self.env.data.mocap_pos[self.env.receiver_mocap]=p
                bowl_ready=np.linalg.norm(self.pos(self.manipulated)[:2]-self.target()[:2])<.035 and self.speed(self.manipulated)<.025 and not self.env.observe().holding
                if t>3 and bowl_ready:
                    eid=self.env.model.equality('receiver_grasp' if self.target_body=='cube' else 'receiver_grasp_'+self.target_body).id
                    self.env.data.eq_active[eid]=False;self.human_dropped=True;self.drop_time=t;self.log('human_released_into_bowl')
            elif self.human_dropped:self.set_hand(self.target()+[.12,0,.22])
        elif self.name=='handover':self.set_hand(self.target())
        else:
            # Point first at requested object, then the destination; card persists.
            target=self.pos(self.target_body) if t<1.5 else self.target()
            self.set_hand(target+[.11,0,.12])

    def human_distance(self):
        smallest=1.
        for r in self.robot_geoms:
            for h in self.human_geoms:
                smallest=min(smallest,float(mujoco.mj_geomDistance(self.env.model,self.env.data,r,h,1.,None)))
        return smallest

    def step(self,action):
        self.human_update();mujoco.mj_forward(self.env.model,self.env.data)
        t=float(self.env.data.time);requested=np.asarray(action,dtype=float).copy();distance=self.human_distance()
        if self.name=='intruder':
            if distance<self.cfg['safety']['stop_distance'] and not self.stop_active:
                self.stop_active=True;self.clear_since=None;self.safety_stops+=1;self.log('safety_stop',distance=distance)
            if self.stop_active:
                if distance>self.cfg['safety']['resume_distance']:
                    if self.clear_since is None:self.clear_since=t
                    if t-self.clear_since>=self.cfg['safety']['clear_dwell_s']:
                        self.stop_active=False;self.log('safety_resume',distance=distance)
                else:self.clear_since=None
            if self.stop_active:action=self.env.hold()
        old=self.env.observe();new=self.env.step(action)
        if new.holding and not self.grasped:self.grasped=True;self.log('robot_grasp',body=self.manipulated)
        if self.grasped and self.pos(self.manipulated)[2]>.72 and not self.lifted:self.lifted=True;self.log('object_lifted',body=self.manipulated)
        if self.name=='handover' and self.lifted and action[6]>.05 and not new.holding and np.linalg.norm(self.pos(self.target_body)-self.hand())<.055:
            if not self.human_holding:self.attach_to_human(self.target_body);self.human_holding=True
        for c in self.env.data.contact:
            if (c.geom1 in self.human_geoms and c.geom2 in self.robot_geoms) or (c.geom2 in self.human_geoms and c.geom1 in self.robot_geoms):
                self.failure='robot_human_contact'
        for body,initial in self.initial_positions.items():
            if body!=self.target_body and self.pos(body)[2]>initial[2]+.045:self.failure='wrong_object_lifted'
        if self.pos(self.manipulated)[2]<.57:self.failure='object_dropped_off_table'
        valid=self._success_predicate()
        self.stable=self.stable+1 if valid else 0
        if self.stable>=int(self.cfg['success']['stable_seconds']/self.env.dt) and not self.success and not self.failure:
            self.success=True;self.log('task_success')
        self.trace.append(dict(timestamp=t,phase=self.phase,action_requested=requested.tolist(),action_executed=self.env.data.ctrl.tolist(),robot_state=np.r_[old.q,old.gripper].tolist(),tcp=new.ee_pos.tolist(),object_position=self.pos(self.manipulated).tolist(),human_position=self.hand().tolist(),human_distance=distance,safety_stop=self.stop_active))
        if t>=self.cfg['success']['timeout_s'] and not self.success:self.failure=self.failure or 'timeout'
        return self.success or self.failure is not None

    def _success_predicate(self):
        if self.name=='handover':return self.lifted and self.human_holding and not self.env.observe().holding
        if self.name=='bowl_assist':
            if not self.human_dropped:return False
            bowl=self.pos(self.manipulated);obj=self.pos(self.target_body)
            others=[b for b in self.initial_bowls if b!=self.manipulated]
            unchanged=all(np.linalg.norm(self.pos(b)-self.initial_bowls[b])<.025 for b in others)
            return unchanged and np.linalg.norm(bowl[:2]-self.target()[:2])<.035 and np.linalg.norm(obj[:2]-bowl[:2])<.041 and .006<obj[2]-bowl[2]<.07 and self.speed(self.target_body)<.04 and not self.env.observe().holding
        p=self.pos(self.target_body);zone=self.cfg['zones'][self.destination]
        placed=np.max(np.abs(p[:2]-np.array(zone['position'][:2])))<zone['half_size']-.025 and .63<p[2]<.70 and self.speed(self.target_body)<.04 and not self.env.observe().holding
        extra=self.name!='intruder' or (self.safety_stops>0 and self.intrusion_finished and any(e['event']=='safety_resume' for e in self.events))
        return self.lifted and placed and extra

    def observation(self,cameras=('front','wrist'),width=640,height=480):
        state=self.env.observe()
        return {'observation.state':np.r_[state.q,state.gripper].astype(np.float32),**{f'observation.images.{c}':self.env.render(c,width,height) for c in cameras}}
    def report(self):
        return dict(schema='hri_scenes_v1',scenario=self.name,role=self.spec['role'],seed=self.seed,requested_object=self.object_key,destination=self.destination,bowl=self.bowl_key if self.name=='bowl_assist' else None,controller='scripted_expert_or_external_action_not_ACT',success=self.success,failure=self.failure,events=self.events,trace=self.trace)
    def close(self):self.env.close()
