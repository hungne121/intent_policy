"""New V2 I1 nominal preview; reuse compatible mechanics, never old demonstrations."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import yaml
from intent_policy.scenarios.config import ScenarioConfig, _deep_merge
from intent_policy.scenarios.i1_coaster_sequence import CoasterSequenceTask
from intent_policy.benchmark.runner import ExpertAgent, ObservationConfig, run_episode, save_record
from intent_policy.benchmark.video import VideoWriter
from intent_policy.sim.restricted_action import RestrictedActionConfig

ROOT=Path(__file__).resolve().parents[1]; OLD=ROOT
parser=argparse.ArgumentParser()
parser.add_argument('--mode', choices=['observed','intent'], required=True)
parser.add_argument('--out', type=Path, default=ROOT/'previews')
args=parser.parse_args()
OUT=args.out; OUT.mkdir(parents=True,exist_ok=True)
NAME='I1_Intent' if args.mode=='intent' else 'I1_Observed'
base=yaml.safe_load((OLD/'configs/scenarios/i1_coaster_sequence.yaml').read_text())
mechanics=yaml.safe_load((OLD/'configs/experiments/ten_task_v11/I1.yaml').read_text())['overrides']
raw=_deep_merge(base,mechanics)
raw=_deep_merge(raw,{'collection_ready':False,'task_code':'T3','intention_change':{'probability':0.},
 'goal':'Three identical cups on three coasters placed by the human, left middle right.',
 'expert':{'rule':'early' if args.mode=='intent' else 'observed','standoff_m':.18},'timing':{'timeout_s':100.},
 'scene':{'grasp_hold_squeeze_m':0.,'objects':{f'C{i}':{'color':'white','label':'identical white cup'} for i in (1,2,3)},
          # 13/11 (user): the three cups stand in one row on one plate (visual slab, collision off in on_reset)
          'decor':[{'name':'plate','type':'slab','xy':[.30,.09],'size':[.11,.10,.001],'color':'tray'}],
          'cameras':{'high':{'pos':[.4,1.15,1.85],'target':[.4,-.04,.85],'fovy':65},
                     'top':{'pos':[.4,.05,1.85],'target':[.4,.05,.62],'fovy':65}}},
 # 13/11 (user): the human never waits for the robot; the robot keeps clear of the (predicted) hand
 'human_behavior':{'yield_to_robot':{'before_start':False,'max_wait_s':0.0},'motion_bank':None,
                   'set_down_aim_m':-.0002},
 'scene_variation':{'sequence':{'pause_before_s':[0.,0.],
     # 13/11 (user): three cups clustered in a triangle on one plate, close in front; nearest-first from the
     # robot's rest TCP (0.15, 0.00): 0.10 / 0.21 / 0.21 m
     'items':[{'cup':'C1','coaster':'CO1','cup_xy':[.24,.04],'coaster_xy':[.60,-.26]},
              {'cup':'C2','coaster':'CO2','cup_xy':[.30,.15],'coaster_xy':[.70,-.26]},
              {'cup':'C3','coaster':'CO3','cup_xy':[.36,.04],'coaster_xy':[.80,-.26]}]}}})
raw['scene_variation']['sequence'].pop('switch',None)
raw['scene_variation']['sequence'].pop('forecast_fault',None)

class V2I1(CoasterSequenceTask):
    def coaster_bottom_gap(self, key):
        from intent_policy.sim.scene_builder import object_half_height
        return float(self.pos(key)[2] - object_half_height(self.scene['objects'][key]) - self.table_z)

    @property
    def expert_class(self):
        from intent_policy.experts.i1_v2_expert import SmoothI1Expert
        return SmoothI1Expert

    def round_released(self, cup):
        # Do not interrupt the upward retreat to start the next cup early.
        height=self.table_z + float(self.cfg.expert.get('travel_dz', .2))
        return super().round_released(cup) and self.tcp()[2] >= height-.005

    def hand_clearance_lift(self, path, shapes, palm_up=0.0, point_at=None,
                            ignore=(), holding=None, **kwargs):
        # Coasters still welded to this hand move with it; they are not static
        # obstacles to route the palm over. The current coaster remains the
        # held body checked against the table by the original geometry code.
        moving={item['coaster'] for item in getattr(self,'rounds',())
                if self.env.data.eq_active[self.weld_id[item['coaster']]]}
        return super().hand_clearance_lift(path,shapes,palm_up,point_at,
                                            set(ignore)|moving,holding,**kwargs)

    def on_reset(self):
        import mujoco
        from intent_policy.sim.i1_v2_human import StackedCoasterHuman
        super().on_reset()
        old = self.human
        human = StackedCoasterHuman(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(human, key, getattr(old, key))
        self.human = human
        human.rest = np.array([.51, -.36, .76])
        human.pos = human.rest.copy()
        human.shape = 'grasp'
        human.holding = self.rounds[0]['coaster']
        human.initial_still_s = float(np.random.default_rng([int(self.seed), 916]).uniform(2.6, 3.6))
        self.initial_still_s = human.initial_still_s
        self._apply_human_pose()
        human.pause_rng = np.random.default_rng([int(self.seed), 917])
        for g in range(self.env.model.ngeom):
            if self.env.model.geom(g).name.startswith('decor_plate_'):
                self.env.model.geom_contype[g] = 0; self.env.model.geom_conaffinity[g] = 0
        for i, item in enumerate(self.rounds):
            key = item['coaster']
            adr = self.qadr[key]
            self.env.data.qpos[adr:adr+3] = [human.pos[0], human.pos[1]+.015,
                                             human.pos[2]-float(self.cfg.human_behavior['grip_dz_m'])+i*.008]
            self.env.data.qpos[adr+3:adr+7] = [1.,0.,0.,0.]
        mujoco.mj_forward(self.env.model, self.env.data)
        for item in self.rounds:
            key = item['coaster']
            self.attach_to_hand(key)
            self.start_pos[key] = self.pos(key).copy()
        mujoco.mj_forward(self.env.model, self.env.data)

    def update_task(self):
        # The robot's preparation pose above the cup it will take is the observable cue the human waits for.
        cup = self.robot_target()
        if ('robot_prepared_' + cup) not in self.protocol_done and 'cup_grasped_' + cup not in self.protocol_done:
            tcp = self.tcp(); above = self.grasp_point(cup)
            if np.linalg.norm(tcp[:2] - above[:2]) < .02 and tcp[2] > above[2] + .03 and np.linalg.norm(self.env.data.qvel[self.env.dids]) < .3:
                self.protocol('robot_prepared_' + cup, 'robot', object=cup)
        super().update_task()

    def step(self, command):
        import mujoco
        previous = getattr(self.human, 'human_round', 0)
        done = super().step(command)
        current = self.human.human_round
        if current > previous and current < len(self.rounds):
            # The remaining discs settle down by one disc thickness after the
            # bottom coaster is released. Re-anchor their hand welds at rest.
            for item in self.rounds[current:]:
                key = item['coaster']
                adr = self.qadr[key]
                self.env.data.qpos[adr+2] -= .008
            mujoco.mj_forward(self.env.model, self.env.data)
            for item in self.rounds[current:]:
                key = item['coaster']
                self.attach_to_hand(key)
            mujoco.mj_forward(self.env.model, self.env.data)
            self.__dict__.setdefault('stack_settles', []).append({'t':self.time,'round':current,'dz_m':-.008})
        return done

    def sample_episode_variation(self, seed, episode_spec=None, _layout_retry=0):
        var=super().sample_episode_variation(seed,episode_spec)
        # Human generator only: destinations never supplied as future policy input.
        destinations=[[.155,-.13],[.305,-.13],[.455,-.13]]
        order=np.random.default_rng([int(seed),811]).permutation(3)
        for row, idx in zip(var['coaster_sequence'],order): row['destination_xy']=destinations[int(idx)]
        return var

cfg=ScenarioConfig.from_dict(raw,source=str(ROOT/'specs/I1_OBSERVED.md'))
sc=V2I1(cfg); writers={}
try:
    def frame(obs, decision, target, scenario):
        for cam in ('high',):
            if cam not in writers: writers[cam]=VideoWriter(OUT/f'{NAME}.mp4',width=640,height=480)
            writers[cam].add(obs[f'observation.images.{cam}'])
    controller=RestrictedActionConfig.load(OLD/'configs/controller/continuous_motion.yaml')
    controller.continuous.update(tcp_speed=.39,tcp_accel=1.0)
    record=run_episode(sc,ExpertAgent(),300100,
        controller_cfg=controller,
        obs_cfg=ObservationConfig(cameras=('high',),width=640,height=480),on_frame=frame,
        episode_id=f'I1_V2_preview_{args.mode}_300100')
    save_record(record,OUT/f'{NAME}_episode.json.gz')
    summary={k:record[k] for k in ('success','failure','duration_s','wall_time_s','protocol_steps_completed')}
    summary.update(mode=args.mode, forecast_source='manicast' if args.mode=='intent' else 'none', source_spec='specs/01_INSTRUCTOR/I1_OBSERVED.md',
        human_source='seeded I1 stacked-coaster scripted human; waits for the robot preparation pose above the next cup',initial_still_s=sc.initial_still_s,
        stack_settles=getattr(sc,'stack_settles',[]),
        status='visual review preview; not an accepted dataset episode',
        limitations=['assisted robot grasp','seeded scripted human rather than HY-Motion replay',
                     'remaining coasters re-anchored by 8 mm after each release; not accepted dataset motion',
                     'scripted expert, not a trained ACT policy','geometry acceptance not evaluated'])
    from intent_policy.benchmark.collection_integrity import motion_source_hash
    summary['motion_source_hash']=motion_source_hash(record)
    if args.mode=='intent':
        from pif.config import load_config
        from pif.utils import resolve
        checkpoint=resolve(load_config(str(OLD/'configs/pif/pif.yaml'),[])['v6_expert']['checkpoint_path'])
        summary['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (OUT/f'{NAME}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/f'{NAME}_config.yaml').write_text(yaml.safe_dump(raw,allow_unicode=True))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
finally:
    for w in writers.values():w.close()
    sc.close()
