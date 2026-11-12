"""Observed-only pointing geometry for the I3 expert, not an ACT policy.

The caller supplies current *observed* object coordinates and index keypoints.
No target, future trajectory, task metadata or case metadata is accepted.
The result is provisional until perception noise and ambiguous gestures are tested.
"""
from math import isfinite, sqrt


def select_pointed(index_base, index_tip, visible_objects, *, max_distance_m, margin_m):
    if len(index_base) != 3 or len(index_tip) != 3 or not visible_objects:
        raise ValueError('3D index keypoints and visible object coordinates required')
    if not isfinite(max_distance_m) or max_distance_m <= 0 or not isfinite(margin_m) or margin_m < 0:
        raise ValueError('invalid decision thresholds')
    if not all(isfinite(x) for x in (*index_base,*index_tip)):
        raise ValueError('nonfinite keypoints')
    vec=[b-a for a,b in zip(index_base,index_tip)]
    norm=sqrt(sum(x*x for x in vec))
    if norm < 1e-8: return None
    direction=[x/norm for x in vec]
    scores=[]
    for key,pos in visible_objects.items():
        if len(pos)!=3 or not all(isfinite(v) for v in pos): raise ValueError('invalid object position')
        relative=[v-a for v,a in zip(pos,index_base)]
        along=sum(a*b for a,b in zip(relative,direction))
        if along <= 0: continue
        distance=sqrt(sum((a-along*b)**2 for a,b in zip(relative,direction)))
        scores.append((distance,key))
    scores.sort()
    if not scores or scores[0][0] > max_distance_m: return None
    if len(scores)>1 and scores[1][0]-scores[0][0] < margin_m: return None
    return scores[0][1]
