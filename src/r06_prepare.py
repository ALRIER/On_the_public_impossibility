import csv
from pathlib import Path

FEATURES=[f"X{i:03d}" for i in range(1,11)]
BOUNDS=[(0.01,0.99),(0.1,1.5),(0.0,2.0),(0.01,0.99),(0.0,36.0),(0.0,1.0),(0.2,3.0),(0.0,1.5),(0.01,0.99),(0.0,1.5)]
CENTERS=["L02950","L00326","L02352","L01136","L02225","L02776"]
RADII=[0.005,0.01,0.02,0.04]
PRIMES=[1009,1013,1019,1021,1031,1033,1039,1049,1051,1061]

src={r["CASE_ID"]:r for r in csv.DictReader(open("outputs/r05_local_map/top_candidates.csv",encoding="utf-8"))}
out=Path("inputs/r06_finalist"); out.mkdir(parents=True,exist_ok=True)
rows=[]; trace=[]; n=0

for ci,cid in enumerate(CENTERS,1):
    c=src[cid]
    center=[float(c[x]) for x in FEATURES]
    for ri,rad in enumerate(RADII,1):
        group=f"F{(ci-1)*4+ri:02d}"
        for k in range(1024):
            n+=1; vals=[]
            for j,x in enumerate(FEATURES):
                u=0.5 if k==0 else (((k+1)*(j+3)*PRIMES[j]+ci*97+ri*53)%4096)/4096.0
                lo,hi=BOUNDS[j]
                v=center[j]+(2*u-1)*rad*(hi-lo)
                vals.append(min(hi,max(lo,v)))
            row={"CASE_ID":f"F{n:05d}","GROUP_ID":group,"P00":c["P00"],"REPLICATES":4,"SEED_BLOCK":200000+n}
            row.update(dict(zip(FEATURES,vals))); rows.append(row)
            trace.append({"CASE_ID":row["CASE_ID"],"GROUP_ID":group,"CENTER_CASE":cid,"RADIUS_FRACTION":rad,"POINT_INDEX":k})

fields=["CASE_ID","GROUP_ID","P00","REPLICATES","SEED_BLOCK",*FEATURES]
with open(out/"stage1_cases.csv","w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
with open(out/"design_trace.csv","w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=list(trace[0])); w.writeheader(); w.writerows(trace)
with open(out/"manifest.csv","w",encoding="utf-8",newline="") as h:
    f=["schema","centers","radii","stage1_cases","stage1_reps","stage2_cases","stage2_reps","stage3_cases","stage3_reps"]
    w=csv.DictWriter(h,fieldnames=f); w.writeheader()
    w.writerow({"schema":"r06_finalist_v1","centers":6,"radii":"0.005;0.01;0.02;0.04","stage1_cases":24576,"stage1_reps":4,"stage2_cases":768,"stage2_reps":24,"stage3_cases":96,"stage3_reps":64})
