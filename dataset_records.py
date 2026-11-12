"""Validate and atomically store *quality-approved* V2 intermediate episodes.

This is an NPZ staging format, not LeRobot. Metadata/events stay separate from
policy features. Geometry and quality checks must be supplied by a real auditor.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from quality_gate import evaluate as quality_errors

CAMERAS=('high','top','wrist')
BASE_FIELDS={'timestamp','robot_qpos','robot_qvel','gripper','human_joints',
             'action_target','action_executed','control_dt',*(f'image_{c}' for c in CAMERAS)}
FORECAST_FIELDS={'forecast','forecast_valid','forecast_generated_at','forecast_observation_end'}


def validate_frames(frames,mode):
    if mode not in ('observed','intent') or not frames: raise ValueError('mode and nonempty frames required')
    fields=BASE_FIELDS | (FORECAST_FIELDS if mode=='intent' else set())
    arrays={}
    for i,frame in enumerate(frames):
        if not isinstance(frame,dict) or set(frame)!=fields: raise ValueError(f'frame {i}: unexpected fields')
    for key in fields:
        try: arrays[key]=np.stack([frame[key] for frame in frames])
        except (ValueError,TypeError) as exc: raise ValueError(f'{key}: inconsistent shape') from exc
    n=len(frames)
    required={'timestamp':(n,), 'robot_qpos':(n,6), 'robot_qvel':(n,6), 'gripper':(n,),
              'human_joints':(n,6,3), 'action_target':(n,7), 'action_executed':(n,7), 'control_dt':(n,)}
    if mode=='intent': required.update(forecast=(n,10,6,3),forecast_valid=(n,10),
                                       forecast_generated_at=(n,),forecast_observation_end=(n,))
    for key,shape in required.items():
        value=arrays[key]
        if value.shape!=shape or value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError(f'{key}: invalid shape or values')
    for camera in CAMERAS:
        arr=arrays[f'image_{camera}']
        if arr.dtype!=np.uint8 or arr.ndim!=4 or arr.shape[-1]!=3 or min(arr.shape[1:3])<1:
            raise ValueError(f'{camera}: expected uint8 RGB')
    ts=arrays['timestamp']
    if (ts<0).any() or np.any(np.diff(ts)<=0) or (arrays['control_dt']<=0).any():
        raise ValueError('timestamps must increase and control_dt must be positive')
    if mode=='intent':
        if arrays['forecast_valid'].dtype.kind!='b': raise ValueError('forecast_valid must be boolean')
        if np.any(arrays['forecast_observation_end']>arrays['forecast_generated_at']) or np.any(arrays['forecast_generated_at']>ts):
            raise ValueError('forecast is not causal')
    return arrays


def write_episode(root, metadata, frames, quality):
    errors=quality_errors(quality)
    if errors: raise ValueError('quality rejected: '+'; '.join(errors))
    if not isinstance(metadata,dict): raise ValueError('metadata required')
    episode_id=metadata.get('episode_id')
    if not isinstance(episode_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',episode_id): raise ValueError('invalid episode_id')
    for key in ('episode_id','task_id','case_id','scene_hash'):
        if metadata.get(key)!=quality.get(key): raise ValueError(f'quality mismatch: {key}')
    if metadata.get('seed')!=quality.get('motion_seed'): raise ValueError('quality mismatch: motion seed')
    for key in ('pair_id','human_clip_hash','motion_family_id','split','source_commit','expert_mode','events'):
        if key not in metadata or metadata[key] is None: raise ValueError(f'missing metadata: {key}')
    mode=metadata['expert_mode']
    if not isinstance(metadata['events'],(list,dict)): raise ValueError('events must be structured')
    if mode=='observed' and metadata.get('forecast_source') not in (None,'none'):
        raise ValueError('observed episode must not use a forecast')
    if mode=='intent':
        if metadata.get('forecast_source')!='manicast': raise ValueError('intent episode requires real ManiCast output')
        if not all(metadata.get(k) for k in ('predictor_version','checkpoint_sha256','forecast_horizons_s','forecast_joint_order')):
            raise ValueError('forecast provenance required')
        if not isinstance(metadata['checkpoint_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',metadata['checkpoint_sha256']):
            raise ValueError('invalid forecast checkpoint sha256')
        expected=['l_shoulder','r_shoulder','l_elbow','r_elbow','l_wrist','r_wrist']
        horizon=np.asarray(metadata['forecast_horizons_s'],dtype=float)
        if horizon.shape!=(10,) or not np.isfinite(horizon).all() or np.any(horizon<=0) or np.any(np.diff(horizon)<=0):
            raise ValueError('invalid forecast horizons')
        if metadata['forecast_joint_order']!=expected: raise ValueError('invalid forecast joint order')
    arrays=validate_frames(frames,mode)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    dest=root/episode_id
    if dest.exists(): raise FileExistsError(dest)
    stage=Path(tempfile.mkdtemp(prefix='.partial-',dir=root))
    try:
        with (stage/'frames.npz').open('wb') as stream:
            np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        payload={'metadata':metadata,'quality':quality,'frames_sha256':hashlib.sha256((stage/'frames.npz').read_bytes()).hexdigest()}
        with (stage/'record.json').open('w',encoding='utf-8') as stream:
            json.dump(payload,stream,ensure_ascii=False,allow_nan=False,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        if dest.exists(): raise FileExistsError(dest)
        stage.rename(dest)
        return dest
    finally:
        if stage.exists(): shutil.rmtree(stage)
