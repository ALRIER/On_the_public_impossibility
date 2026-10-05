import csv
from pathlib import Path

SUMMARY=Path("outputs/r06_finalist/finalist_summary.csv")
CASES=Path("outputs/r06_finalist/stage3_cases.csv")
OUT=Path("inputs/r07_final_confirmation")

summary={r["CASE_ID"]:r for r in csv.DictReader(SUMMARY.open(encoding="utf-8"))}
cases={r["CASE_ID"]:r for r in csv.DictReader(CASES.open(encoding="utf-8"))}

selected=[]
for cid,s in summary.items():
    if (
        str(s["FINITE_STATE"]).lower()=="true"
        and int(s["DEV_PASS"])>=40
        and int(s["CAL_PASS"])>=4
        and int(s["SEP_PASS"])>=4
    ):
        r=dict(cases[cid])
        r["REPLICATES"]="256"
        r["SEED_BLOCK"]=str(int(r["SEED_BLOCK"])+9000000)
        selected.append(r)

selected.sort(key=lambda r:r["CASE_ID"])
if len(selected)!=22:
    raise SystemExit(f"Expected 22 frozen candidates, found {len(selected)}")

OUT.mkdir(parents=True,exist_ok=True)
fields=list(selected[0])
with (OUT/"cases.csv").open("w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(selected)

with (OUT/"manifest.csv").open("w",encoding="utf-8",newline="") as h:
    fields=["schema","selection_rule","candidates","replications_per_candidate","total_replications","seed_policy","adaptive_search"]
    w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
    w.writerow({
        "schema":"r07_final_confirmation_v1",
        "selection_rule":"R06_STAGE3_FINITE_AND_DEV_GE40_AND_CAL_GE4_AND_SEP_GE4",
        "candidates":22,
        "replications_per_candidate":256,
        "total_replications":5632,
        "seed_policy":"R06_seed_block_plus_9000000",
        "adaptive_search":"false",
    })
