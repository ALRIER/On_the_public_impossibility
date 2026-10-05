import csv, glob, heapq
from collections import Counter
from pathlib import Path

POOL_CAP=50000
PROMOTE=2048
heap=[]
serial=0
counts=Counter()
total=0

def rank(r):
    return (
        int(r["PASS_COUNT"]),
        -float(r["MAX_GAP"]),
        -float(r["SUM_GAP"]),
        -float(r["CENTER_DISTANCE"]),
    )

for p in sorted(glob.glob("/tmp/network_n1/shard_*.csv")):
    with open(p,encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h):
            total+=1
            pc=int(r["PASS_COUNT"])
            counts[pc]+=1
            serial+=1
            item=(rank(r),serial,r)
            if len(heap)<POOL_CAP:
                heapq.heappush(heap,item)
            elif item[0]>heap[0][0]:
                heapq.heapreplace(heap,item)

pool=[x[2] for x in heap]
pool.sort(key=lambda r:(-int(r["PASS_COUNT"]),float(r["MAX_GAP"]),float(r["SUM_GAP"]),float(r["CENTER_DISTANCE"]),r["NID"]))

# Diversity-preserving promotion: one best point per 4^6 neutral hypercell,
# then fill remaining slots from the global ranking.
cells={}
for r in pool:
    key=tuple(min(3,int(float(r[f"Z{i:03d}"])*4.0)) for i in range(1,7))
    if key not in cells:
        cells[key]=r
diverse=sorted(cells.values(),key=lambda r:(-int(r["PASS_COUNT"]),float(r["MAX_GAP"]),float(r["SUM_GAP"]),float(r["CENTER_DISTANCE"]),r["NID"]))

selected=[]
seen=set()
for r in diverse:
    if len(selected)>=PROMOTE: break
    selected.append(r); seen.add(r["NID"])
for r in pool:
    if len(selected)>=PROMOTE: break
    if r["NID"] not in seen:
        selected.append(r); seen.add(r["NID"])

out=Path("outputs/network_closure_n1")
out.mkdir(parents=True,exist_ok=True)

with open(out/"pass_count_distribution.csv","w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=["PASS_COUNT","CASES","SHARE"])
    w.writeheader()
    for pc in range(7):
        n=counts.get(pc,0)
        w.writerow({"PASS_COUNT":pc,"CASES":n,"SHARE":n/max(total,1)})

for name,rows in [("top_pool.csv",pool[:10000]),("promotion_queue.csv",selected)]:
    with open(out/name,"w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

best=pool[0]
with open(out/"summary.csv","w",encoding="utf-8",newline="") as h:
    fields=["TOTAL_CASES","BEST_PASS_COUNT","SIX_OF_SIX_CASES","PROMOTION_CASES","BEST_MAX_GAP","BEST_SUM_GAP","BEST_CENTER_DISTANCE"]
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
    w.writerow({
        "TOTAL_CASES":total,
        "BEST_PASS_COUNT":best["PASS_COUNT"],
        "SIX_OF_SIX_CASES":counts.get(6,0),
        "PROMOTION_CASES":len(selected),
        "BEST_MAX_GAP":best["MAX_GAP"],
        "BEST_SUM_GAP":best["SUM_GAP"],
        "BEST_CENTER_DISTANCE":best["CENTER_DISTANCE"],
    })
