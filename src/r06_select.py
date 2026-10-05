import csv,glob,sys
from collections import defaultdict
from pathlib import Path

stage=int(sys.argv[1])
if stage==2:
    input_cases="/tmp/design/stage1_cases.csv"
    result_glob="/tmp/s1/s1_*.csv"
    out=Path("/tmp/sel2"); out.mkdir(parents=True,exist_ok=True)
    n_select=768; floor_per_group=16; reps=24; seed_add=2000000
    cases_name="stage2_cases.csv"; summary_name="stage1_summary.csv"
else:
    input_cases="/tmp/sel2/stage2_cases.csv"
    result_glob="/tmp/s2/s2_*.csv"
    out=Path("/tmp/sel3"); out.mkdir(parents=True,exist_ok=True)
    n_select=96; floor_per_group=2; reps=64; seed_add=4000000
    cases_name="stage3_cases.csv"; summary_name="stage2_summary.csv"

inp={r["CASE_ID"]:r for r in csv.DictReader(open(input_cases,encoding="utf-8"))}
rows=[]
for p in glob.glob(result_glob):
    rows.extend(csv.DictReader(open(p,encoding="utf-8")))
rows.sort(key=lambda r:(-int(r["DEV_PASS"]),-int(r["SEP_PASS"]),-int(r["CAL_PASS"]),r["CASE_ID"]))
by=defaultdict(list)
for r in rows: by[r["GROUP_ID"]].append(r)

selected=[]; seen=set()
for g in sorted(by):
    for r in by[g][:floor_per_group]:
        selected.append(r); seen.add(r["CASE_ID"])
for r in rows:
    if len(selected)>=n_select: break
    if r["CASE_ID"] not in seen:
        selected.append(r); seen.add(r["CASE_ID"])

cases=[]
for r in selected:
    z=dict(inp[r["CASE_ID"]]); z["REPLICATES"]=str(reps); z["SEED_BLOCK"]=str(int(z["SEED_BLOCK"])+seed_add); cases.append(z)
with open(out/cases_name,"w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=list(cases[0])); w.writeheader(); w.writerows(cases)

reduced=[{"CASE_ID":r["CASE_ID"],"GROUP_ID":r["GROUP_ID"],"P00":r["P00"],"FINITE_STATE":r["FINITE_STATE"],"CAL_PASS":r["CAL_PASS"],"DEV_PASS":r["DEV_PASS"],"SEP_PASS":r["SEP_PASS"]} for r in rows]
with open(out/summary_name,"w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=list(reduced[0])); w.writeheader(); w.writerows(reduced)
