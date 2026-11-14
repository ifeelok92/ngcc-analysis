import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--seconds',type=int,required=True)
p.add_argument('--record',required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
t=time.monotonic()
log=Path(a.record).with_suffix('.log')
with log.open('w') as stream:
    try:
        r=subprocess.run(a.command,stdout=stream,stderr=subprocess.STDOUT,timeout=a.seconds)
        status={'exit_code':r.returncode,'timed_out':False}
    except subprocess.TimeoutExpired:
        status={'exit_code':None,'timed_out':True}
status.update(command=a.command,seconds=time.monotonic()-t,limit=a.seconds,log=str(log))
Path(a.record).write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
