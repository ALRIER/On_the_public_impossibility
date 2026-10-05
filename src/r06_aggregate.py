import csv,glob,os
from collections import defaultdict
from pathlib import Path

out=Path("outputs/r06_finalist"); out.mkdir(parents=True,exist_ok=True)
copies=[
("/tmp/design/manifest.csv","design_manifest.csv"),
("/tmp/design/design_trace.csv","design_trace.csv"),
("/tmp/sel2/stage1_summary.csv","stage1_summary.csv"),
("/tmp/sel2/stage2_cases.csv","stage2_cases.csv"),
("/tmp/sel3/stage2_summary.csv","stage2_summary.csv"),
("/tmp/sel3/stage3_cases.csv","stage3_cases.csv")]
for a,b in copies: (out/b).write_bytes(Path(a).read_bytes())

rows=[]; fields=[]
for p in glob.glob("/tmp/s3/s3_*.csv"):
    with open(p,encoding="utf-8",newline="") as h:
        for r in csv.DictReader(h):
            rows.append(r)
            for k in r:
                if k not in fields: fields.append(k)
rows.sort(key=lambda r:(-int(r["DEV_PASS"]),-int(r["SEP_PASS"]),-int(r["CAL_PASS"]),r["CASE_ID"]))

with open(out/"stage3_results.csv","w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)

small=[{"CASE_ID":r["CASE_ID"],"GROUP_ID":r["GROUP_ID"],"P00":r["P00"],"FINITE_STATE":r["FINITE_STATE"],"CAL_PASS":int(r["CAL_PASS"]),"DEV_PASS":int(r["DEV_PASS"]),"SEP_PASS":int(r["SEP_PASS"])} for r in rows]
with open(out/"finalist_summary.csv","w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=list(small[0])); w.writeheader(); w.writerows(small)

by=defaultdict(list)
for r in small: by[r["GROUP_ID"]].append(r)
region=[]
for g,rs in sorted(by.items()):
    region.append({"GROUP_ID":g,"CASES":len(rs),"MEAN_DEV_PASS":sum(x["DEV_PASS"] for x in rs)/len(rs),"MAX_DEV_PASS":max(x["DEV_PASS"] for x in rs),"COUNT_GE_40":sum(x["DEV_PASS"]>=40 for x in rs),"COUNT_GE_41":sum(x["DEV_PASS"]>=41 for x in rs),"COUNT_GE_42":sum(x["DEV_PASS"]>=42 for x in rs),"MEAN_SEP_PASS":sum(x["SEP_PASS"] for x in rs)/len(rs),"FINITE_CASES":sum(str(x["FINITE_STATE"]).lower()=="true" for x in rs)})
with open(out/"finalist_region_summary.csv","w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=list(region[0])); w.writeheader(); w.writerows(region)

with open(out/"run_manifest.csv","w",encoding="utf-8",newline="") as h:
    f=["RUN_ID","STAGE1_CASES","STAGE1_REPS","STAGE2_CASES","STAGE2_REPS","STAGE3_CASES","STAGE3_REPS","TOTAL_TRUE_ABM_REPLICATIONS","ROLE"]
    w=csv.DictWriter(h,fieldnames=f); w.writeheader()
    w.writerow({"RUN_ID":os.environ.get("GITHUB_RUN_ID",""),"STAGE1_CASES":24576,"STAGE1_REPS":4,"STAGE2_CASES":768,"STAGE2_REPS":24,"STAGE3_CASES":96,"STAGE3_REPS":64,"TOTAL_TRUE_ABM_REPLICATIONS":122880,"ROLE":"developmental_semifinalist_local_confirmation"})
