"""Explicit event metrics for matched preliminary expert rollouts.

No event is fabricated from missing observations, and failed rollouts have no
completion time. TSync is the project's definition, not an HRIBench reproduction.
"""
import math


def episode_metrics(episode):
    events=episode.get('events',{})
    success=episode.get('success') is True
    def time(name):
        t=events.get(name)
        if t is None: return None
        if not isinstance(t,(int,float)) or isinstance(t,bool) or not math.isfinite(t) or t<0:
            raise ValueError(f'invalid event time {name}')
        return float(t)
    start,end=time('task_active'),time('completed')
    h,r=time('human_ready'),time('robot_ready')
    if start is not None and end is not None and end < start: raise ValueError('completed before task active')
    return {'success':success, 'completion_s':end-start if success and start is not None and end is not None else None,
            'wait_s':max(0.,r-h) if h is not None and r is not None else None,
            'tsync_s':abs(r-h) if h is not None and r is not None else None}


def compare_pair(observed,intent):
    for k in ('task_id','case_id','scene_hash','human_clip_hash','seed'):
        if observed.get(k) is None or observed.get(k)!=intent.get(k): raise ValueError(f'unmatched {k}')
    if intent.get('forecast_source')!='manicast': raise ValueError('intent must use ManiCast')
    if observed.get('forecast_source') not in (None,'none'): raise ValueError('observed cannot use forecast')
    a,b=episode_metrics(observed),episode_metrics(intent)
    delta={}
    for k in ('completion_s','wait_s','tsync_s'):
        delta[k]=a[k]-b[k] if a[k] is not None and b[k] is not None else None
    return {'observed':a,'intent':b,'observed_minus_intent':delta}
