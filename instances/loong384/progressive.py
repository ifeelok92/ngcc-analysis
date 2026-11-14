import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from fpylll import BKZ, FPLLL, GSO, IntegerMatrix, LLL

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'method_C'))
from qary_embedding import center, inverse

def beam_points(B,target,width,adaptive=False,limit=None):
    FPLLL.set_precision(212)
    gso=GSO.Mat(B,float_type='mpfr',update=True)
    n=B.nrows
    mu=np.array([[gso.get_mu(i,j) if j<i else 0 for j in range(n)] for i in range(n)])
    rr=np.array([gso.get_r(i,i) for i in range(n)])
    states=np.array([gso.from_canonical(tuple(map(int,target)))])
    coeff=np.zeros((1,n),dtype=np.int64)
    costs=np.zeros(1)
    for i in range(n-1,-1,-1):
        middle=np.rint(states[:,i]).astype(np.int64)
        if adaptive:
            span=max(1,min(16,int(np.ceil(8/np.sqrt(rr[i])))))
            offsets=np.arange(-span,span+1)
            choices=(middle[:,None]+offsets).reshape(-1)
            branches=len(offsets)
        else:
            other=middle+np.where(states[:,i]>=middle,1,-1)
            choices=np.column_stack((middle,other)).reshape(-1)
            branches=2
        new_cost=np.repeat(costs,branches)+(np.repeat(states[:,i],branches)-choices)**2*rr[i]
        keep=np.argsort(new_cost)[:width]
        parent=keep//branches
        vals=choices[keep]
        states=states[parent].copy()
        coeff=coeff[parent].copy()
        costs=new_cost[keep]
        coeff[:,i]=vals
        states[:,:i]-=vals[:,None]*mu[i,:i]
    mat=np.array([[int(B[i,j]) for j in range(n)] for i in range(n)],dtype=object)
    return coeff[:limit].astype(object)@mat, {'beam_width':width,'adaptive':adaptive,'gso_min_length':float(np.sqrt(min(rr))),'gso_max_length':float(np.sqrt(max(rr))),'beam_best_cost':float(min(costs))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('system')
    p.add_argument('--mode',choices=['embedding','decode'],required=True)
    p.add_argument('--m',type=int,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--embedding',type=int,default=8)
    p.add_argument('--schedule',default='20:1,25:1,30:2,35:2,40:3,45:3,50:3')
    p.add_argument('--beam',type=int,default=128)
    p.add_argument('--out',required=True)
    p.add_argument('--resume')
    p.add_argument('--variant',choices=['standard','selfdual'],default='standard')
    p.add_argument('--gh-bound',action='store_true')
    a=p.parse_args()
    started=time.monotonic()
    dest=Path(a.out)
    dest.mkdir(parents=True,exist_ok=True)
    def log(obj):
        obj['seconds']=time.monotonic()-started
        line=json.dumps(obj)
        print(line,flush=True)
        with (dest/'events.jsonl').open('a') as stream:
            stream.write(line+'\n')
    d=np.load(a.system)
    f,y=d['F'],d['y']
    q,eta,bound=int(d['q']),int(d['eta']),int(d['error_bound'])
    m,n=a.m,f.shape[1]
    which=np.random.default_rng(a.seed).choice(len(y),m,replace=False)
    fs,ys=f[which],center(y[which],q)
    inv=inverse(fs[:n],q)
    C=fs[n:]@inv%q
    dim=m+(a.mode=='embedding')
    meta={'args':vars(a),'q':q,'n':n,'m':m,'dimension':dim,'local_bound':bound,'input_sha256':hashlib.sha256(Path(a.system).read_bytes()).hexdigest(),'strategy':BKZ.DEFAULT_STRATEGY.decode(),'fpylll_precision':212}
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2))
    log({'event':'start',**meta})
    if a.resume:
        saved=json.loads(Path(a.resume).read_text())
        assert saved['input_sha256']==meta['input_sha256'] and saved['which']==which.tolist() and saved['mode']==a.mode
        B=IntegerMatrix.from_matrix(saved['basis'])
        assert B.nrows==dim and B.ncols==dim
    else:
        B=IntegerMatrix(dim,dim)
        for i in range(n):
            B[i,i]=1
            for j in range(m-n):
                B[i,n+j]=int(center(C[j,i],q))
        for i in range(n,m):
            B[i,i]=q
        if a.mode=='embedding':
            for j in range(m):
                B[m,j]=int(ys[j])
            B[m,m]=a.embedding
        LLL.reduction(B)
        log({'event':'lll_done'})
    def save(stage):
        record={'basis':[[int(B[i,j]) for j in range(dim)] for i in range(dim)],'stage':stage,'which':which.tolist(),'mode':a.mode,'input_sha256':meta['input_sha256']}
        tmp=dest/'checkpoint.tmp'
        tmp.write_text(json.dumps(record))
        tmp.replace(dest/'checkpoint.json')
    original=np.load(ROOT/'data/system_A.npz')
    best_all=None
    def accept(point,source):
        nonlocal best_all
        point=np.asarray(point,dtype=object)
        residual=point-ys.astype(object)
        norm=float(np.sqrt(sum(int(v)**2 for v in residual)))
        if best_all is None or norm<best_all:
            best_all=norm
        t=np.array([int(v) for v in center(inv.astype(object)@point[:n],q)],dtype=np.int64)
        if np.max(abs(t))>eta:
            return False
        local=center(f@t-y,q)
        if np.max(abs(local))>bound:
            return False
        x=d['x0']+d['K']@t if 'K' in d else t
        if len(x)!=160 or np.max(abs(x))>3:
            return False
        rr=center(original['F'].astype(object)@x.astype(object)-original['y'].astype(object),q)
        mx=int(max(abs(rr)))
        if mx>14:
            return False
        result={'candidates':[{'x':x.tolist(),'max_residual':mx,'rms_residual':float(np.sqrt(np.mean(np.array(rr,dtype=float)**2)))}],'source':source,'original_rows_checked':len(rr)}
        (dest/'candidate.json').write_text(json.dumps(result,indent=2))
        log({'event':'candidate',**result})
        return True
    def search(stage):
        tested=0
        info={}
        if a.mode=='decode':
            FPLLL.set_precision(212)
            g=GSO.Mat(B,float_type='mpfr',update=True)
            coeff=g.babai(tuple(map(int,ys)))
            point=B.multiply_left(coeff)
            tested+=1
            if accept(point,stage+':babai'):
                return True
            points,info=beam_points(B,ys,a.beam)
            for point in points:
                tested+=1
                if accept(point,stage+':beam'):
                    return True
        else:
            mat=np.array([[int(B[i,j]) for j in range(dim)] for i in range(dim)],dtype=np.int64)
            for i in range(dim):
                last=int(mat[i,m])
                if abs(last)==a.embedding:
                    tested+=1
                    if accept(ys-mat[i,:m]*(last//a.embedding),stage+':row'):
                        return True
            # Check sums and differences with the required last coordinate.
            for i in range(dim):
                for sign in [1,-1]:
                    idx=np.flatnonzero(mat[i,m]+sign*mat[:,m]==a.embedding)
                    idx=idx[idx>i]
                    if not len(idx):
                        continue
                    vectors=mat[i,:m]+sign*mat[idx,:m]
                    for v in vectors:
                        if np.max(abs(v))<=bound:
                            tested+=1
                            if accept(ys-v,stage+':pair'):
                                return True
            info={'shortest_basis_norm':float(min(np.sqrt(np.sum(mat.astype(float)**2,axis=1))))}
        log({'event':'search','stage':stage,'tested':tested,'best_selected_error_norm':best_all,**info})
        return False
    save('lll_or_resume')
    if search('lll_or_resume'):
        return
    for specification in a.schedule.split(','):
        beta,tours=map(int,specification.split(':'))
        for tour in range(1,tours+1):
            stage=f'bkz{beta}_tour{tour}'
            log({'event':'tour_start','stage':stage})
            flags=BKZ.SD_VARIANT if a.variant=='selfdual' else BKZ.DEFAULT
            if a.gh_bound:
                flags|=BKZ.GH_BND
            BKZ.reduction(B,BKZ.Param(block_size=beta,strategies=BKZ.DEFAULT_STRATEGY,max_loops=1,flags=flags),float_type='mpfr',precision=212)
            save(stage)
            log({'event':'tour_done','stage':stage})
            if search(stage):
                return
    log({'event':'schedule_complete','candidate_found':False})

if __name__=='__main__':
    main()
