import argparse
import json
from pathlib import Path
import time
import numpy as np
from fpylll import BKZ, IntegerMatrix, LLL

def center(v,q):
    return (v+q//2)%q-q//2

def inverse(a,q):
    n=len(a)
    w=np.hstack((a%q,np.eye(n,dtype=np.int64)))
    for j in range(n):
        piv=np.flatnonzero(w[j:,j])
        if not len(piv):
            raise ValueError('singular pivot matrix')
        k=j+int(piv[0])
        w[[j,k]]=w[[k,j]]
        w[j]=w[j]*pow(int(w[j,j]),-1,q)%q
        factors=w[:,j].copy()
        factors[j]=0
        w=(w-factors[:,None]*w[j])%q
    return w[:,n:]

def run(args):
    started=time.monotonic()
    d=np.load(args.system)
    f,y=d['F'].astype(np.int64),d['y'].astype(np.int64)
    q,eta=int(d['q']),int(d['eta'])
    bound=int(d['error_bound']) if 'error_bound' in d else args.bound
    m,n=args.m,f.shape[1]
    if m<n:
        raise ValueError('m must be >= n')
    which=np.random.default_rng(args.seed).choice(len(y),m,replace=False)
    fs,ys=f[which],y[which]
    inv=inverse(fs[:n],q)
    assert np.array_equal(fs[:n]@inv%q,np.eye(n,dtype=np.int64))
    C=fs[n:]@inv%q
    B=IntegerMatrix(m+1,m+1)
    for j in range(n):
        B[j,j]=1
        for i in range(m-n):
            B[j,n+i]=int(center(C[i,j],q))
    for j in range(n,m):
        B[j,j]=q
    for j in range(m):
        B[m,j]=int(center(ys[j],q))
    B[m,m]=args.embedding
    # Independently check membership for synthetic coefficients before reduction.
    t=np.random.default_rng(991).integers(-3,4,n)
    v=fs@t%q
    assert np.all((v[n:]-C@v[:n])%q==0)
    print(json.dumps({'start':vars(args),'dimension':m+1,'q_determinant_exponent':m-n,'construction_tests':'PASS'}),flush=True)
    if args.lll=='wrapper':
        LLL.reduction(B)
    else:
        LLL.reduction(B,method='proved',float_type='mpfr',precision=212)
    print(json.dumps({'lll_seconds':time.monotonic()-started}),flush=True)
    candidates=[]
    history=[]
    stages=[2]+([args.block] if args.block>2 else [])
    for stage in stages:
        if stage>2:
            BKZ.reduction(B,BKZ.Param(block_size=stage,max_loops=args.loops),float_type='mpfr',precision=212)
        eligible=0
        best=None
        for i in range(m+1):
            last=int(B[i,m])
            if abs(last)!=args.embedding:
                continue
            r=np.array([int(B[i,j])*(last//args.embedding) for j in range(m)],dtype=np.int64)
            if np.max(abs(r))>bound:
                continue
            t=center(inv@(ys[:n]-r[:n]),q)
            if np.max(abs(t))>eta:
                continue
            eligible+=1
            residual=center(f@t-y,q)
            mx=int(np.max(abs(residual)))
            best=mx if best is None else min(best,mx)
            if mx>bound:
                continue
            x=d['x0']+d['K']@t if 'K' in d else t
            if 'K' in d and np.max(abs(x))>3:
                continue
            if 'original_F' in d:
                residual=center(d['original_F']@x-d['original_y'],q)
                mx=int(np.max(abs(residual)))
                if mx>14:
                    continue
            entry={'x':x.tolist(),'max_residual':mx,'rms_residual':float(np.sqrt(np.mean(residual.astype(float)**2))),'stage':stage}
            if not any(c['x']==entry['x'] for c in candidates):
                candidates.append(entry)
        h={'stage':stage,'seconds':time.monotonic()-started,'eligible':eligible,'best_max_residual':best,'accepted':len(candidates)}
        history.append(h)
        print(json.dumps(h),flush=True)
        if candidates:
            break
    result={'args':vars(args),'dimension':m+1,'seconds':time.monotonic()-started,'rows_checked':len(d['original_y']) if 'original_y' in d else len(y),'history':history,'candidates':candidates}
    Path(args.out).write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('system')
    p.add_argument('--m',type=int,default=240)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--embedding',type=int,default=8)
    p.add_argument('--block',type=int,default=20)
    p.add_argument('--loops',type=int,default=2)
    p.add_argument('--bound',type=int,default=14)
    p.add_argument('--lll',choices=['wrapper','proved'],default='wrapper')
    p.add_argument('--out',required=True)
    print(json.dumps(run(p.parse_args())),flush=True)
