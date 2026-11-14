import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from fpylll import FPLLL,GSO,IntegerMatrix,LLL

ROOT=Path(__file__).resolve().parent
from qary_embedding import center
from progressive import beam_points

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--out',required=True)
p.add_argument('--width',type=int,default=4096)
p.add_argument('--no-beam',action='store_true')
a=p.parse_args()
dest=Path(a.out)
dest.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
raw=Path(a.checkpoint).read_bytes()
(dest/'source_checkpoint.json').write_bytes(raw)
s=json.loads(raw)
mat=np.array(s['basis'],dtype=np.int64)
k=mat[:,-1]//8
assert not np.any(mat[:,-1]%8)
pivots=np.flatnonzero(abs(k)==1)
assert len(pivots),'No unit embedding pivot in this checkpoint'
pivot=int(min(pivots,key=lambda j:sum(int(v)**2 for v in mat[j,:-1])))
rows=[mat[i,:-1]-int(k[i]*k[pivot])*mat[pivot,:-1] for i in range(len(mat)) if i!=pivot]
B=IntegerMatrix.from_matrix(np.array(rows).tolist())
LLL.reduction(B)
n,w=128,s['weight']
d=np.load(ROOT/'denoised_lift.npz')
which=s['which']
mu=np.array(s['shift'],dtype=np.int64)
y=np.concatenate((w*mu,center(d['y'][which],8191)))
matrix=np.array([[int(B[i,j]) for j in range(B.ncols)] for i in range(B.nrows)],dtype=np.int64)
assert not np.any(matrix[:,:n]%w)
assert not np.any((matrix[:,n:]-(matrix[:,:n]//w)@d['F'][which].T)%8191)
FPLLL.set_precision(212)
g=GSO.Mat(B,float_type='mpfr',update=True)
logdet=sum(np.log(g.get_r(i,i)) for i in range(B.nrows))/2
expected=n*np.log(w)+len(which)*np.log(8191)
assert abs(logdet-expected)<1e-6
meta={'source':a.checkpoint,'source_sha256':hashlib.sha256(raw).hexdigest(),'source_stage':s['stage'],'unit_pivot':pivot,
    'rank':B.nrows,'exact_membership':True,'log_determinant':float(logdet),'expected_log_determinant':float(expected),'which':which,'weight':w,'shift':s['shift']}
(dest/'metadata.json').write_text(json.dumps(meta,indent=2))
(dest/'checkpoint.json').write_text(json.dumps({'basis':matrix.tolist(),'stage':'lll_kernel','mode':'weighted_decode',**meta}))
original=np.load(ROOT/'system_A.npz')
best=None
candidates=[]
tested=0
small=0
def accept(point,source):
    global best,tested,small
    point=np.array(point,dtype=object)
    tested+=1
    error=point-y.astype(object)
    norm=float(np.sqrt(sum(int(v)**2 for v in error)))
    best=norm if best is None else min(best,norm)
    assert not np.any(point[:n]%w)
    t=np.array(point[:n]//w,dtype=np.int64)
    if max(abs(t))>3:
        return
    small+=1
    x=d['x0']+d['K']@t
    if max(abs(x))>3:
        return
    residual=center(original['F'].astype(object)@x.astype(object)-original['y'].astype(object),8191)
    if max(abs(residual))<=14:
        candidates.append({'x':x.tolist(),'max_residual':int(max(abs(residual))),'rms_residual':float(np.sqrt(np.mean(np.array(residual,dtype=float)**2))),'source':source})

accept(B.multiply_left(g.babai(tuple(map(int,y)))),'babai')
info={}
if not a.no_beam and not candidates:
    points,info=beam_points(B,y,a.width,adaptive=True,limit=32)
    for point in points:
        accept(point,'adaptive_beam')
result={'seconds':time.monotonic()-start,'candidates':candidates,'best_weighted_error_norm':best,'tested':tested,'box_candidates':small,'source_stage':s['stage'],**info}
(dest/'result.json').write_text(json.dumps(result,indent=2))
if candidates:
    (dest/'candidate.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
