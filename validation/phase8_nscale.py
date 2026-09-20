"""At FIXED 1 ns stride, how does constructibility scale with frame count?
This isolates the one axis mdCATH changes (5x more time at the same stride)."""
import sys, json, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/claude/SciML-MD'); warnings.filterwarnings("ignore")
from validation.phase8_recurrence import load_atlas, _pairwise_rmsd_ang, _levels
MIN_GAP_NS=10.0
out={}
for d in sorted(p for p in Path('/mnt/user-data/uploads/atlas').iterdir() if p.is_dir()):
    got=load_atlas(d)
    if not got: continue
    traj,base=got
    R=_pairwise_rmsd_ang(traj)[::10,::10]      # deliver at 1 ns
    out[d.name]={}
    for n in [26,51,76,101]:
        Rs=R[:n]; Rs=Rs[:,:n]
        ii,jj=np.triu_indices(n,k=10)
        ref=np.array([Rs[t,t+1] for t in range(n-1)])
        L=_levels(Rs[ii,jj],ref,ii,f"{n}f",n)
        out[d.name][n]={"P50":L["levels"]["50"]["constructible"],
                        "n_adm":L["levels"]["50"]["n_admissible"],
                        "pct":L["closest_pair_percentile"]}
print(f"{'frames':>7} {'ns':>6} {'frac P50 ok':>12} {'median admissible':>18} {'median pct':>11}")
for n in [26,51,76,101]:
    v=[r[n] for r in out.values()]
    print(f"{n:7d} {n:6d} {np.mean([x['P50'] for x in v]):12.2f}"
          f" {np.median([x['n_adm'] for x in v]):18.0f} {np.median([x['pct'] for x in v]):11.1f}")
json.dump(out,open('/home/claude/SciML-MD/validation/phase8_nscale.json','w'),indent=2,default=str)
