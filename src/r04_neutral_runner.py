from __future__ import annotations

import argparse
import base64
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
RUNTIME_ROOT=ROOT/".r04_runtime_tmp"

CAL_IDS={"T002","T003","T004","T005","T006"}

def deep_merge(base,overlay):
    out=dict(base)
    for k,v in (overlay or {}).items():
        if k in {"cycle_id","cycle_version","baseline_cycle"}:
            continue
        if isinstance(v,dict) and isinstance(out.get(k),dict):
            out[k]=deep_merge(out[k],v)
        else:
            out[k]=v
    return out

def materialize():
    raw=base64.b64decode((ROOT/"runtime/r04/runtime.b64").read_text(encoding="ascii"))
    payload=json.loads(raw.decode("utf-8"))
    if payload.get("schema")!="opaque_runtime_r04_v1":
        raise RuntimeError("runtime schema mismatch")
    for rel,content in payload["files"].items():
        p=RUNTIME_ROOT/rel
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(content,encoding="utf-8")
    sys.path.insert(0,str(RUNTIME_ROOT/"code"))

def load_maps():
    m=pd.read_csv(RUNTIME_ROOT/"results/Test3/private_public_mapping_r04.csv")
    x_rev=dict(zip(m.loc[m.kind.eq("FEATURE"),"public_id"],m.loc[m.kind.eq("FEATURE"),"private_name"]))
    p_rev=dict(zip(m.loc[m.kind.eq("PROFILE"),"public_id"],m.loc[m.kind.eq("PROFILE"),"private_name"]))
    t_fwd=dict(zip(m.loc[m.kind.eq("TARGET"),"private_name"],m.loc[m.kind.eq("TARGET"),"public_id"]))
    return x_rev,p_rev,t_fwd

def load_cfg(profile_private):
    base=yaml.safe_load((RUNTIME_ROOT/"code/stages/Test3/config/test3.yaml").read_text(encoding="utf-8"))
    over=yaml.safe_load((RUNTIME_ROOT/"code/stages/Test3/remediation_cycle_14/config_override.yaml").read_text(encoding="utf-8"))
    cfg=deep_merge(base,over)
    if profile_private!="BASE":
        cand=over.get("candidate_overrides",{}).get(profile_private,{})
        cfg=deep_merge(cfg,cand)
    return cfg

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--start",type=int,required=True)
    ap.add_argument("--count",type=int,required=True)
    a=ap.parse_args()

    materialize()
    x_rev,p_rev,t_fwd=load_maps()

    model=importlib.import_module("stages.Test3.test3_model")
    data=importlib.import_module("stages.Test3.test3_data")
    validation=importlib.import_module("stages.Test3.validation")

    cases=pd.read_csv(a.cases).iloc[a.start:a.start+a.count]
    cal_rows={t_fwd[r["target_id"]]:r for r in data.target_rows("CALIBRATION") if t_fwd.get(r["target_id"]) in CAL_IDS}
    dev_rows=data.target_rows("INDEPENDENT_VALIDATION")
    separate={t_fwd[r["target_id"]] for r in dev_rows if str(r.get("independence_level",""))=="SEPARATE_SOURCE"}

    out=[]
    for _,q in cases.iterrows():
        profile_private=p_rev[str(q["P00"])]
        cfg=load_cfg(profile_private)
        params={private:float(q[public]) for public,private in x_rev.items()}
        reps=int(q["REPLICATES"])
        seed_block=int(q["SEED_BLOCK"])
        master=int(cfg["reproducibility"]["master_seed"])
        sims=[
            model.simulate(
                params,
                int(cfg["simulation"]["periods"]),
                int(cfg["simulation"]["warmup_periods"]),
                master+91000000+seed_block*100003+r*104729,
                cfg["simulation"],
            )
            for r in range(reps)
        ]
        moms=[s.moments for s in sims]
        qlo=float(cfg["validation"]["model_interval_lower_quantile"])
        qhi=float(cfg["validation"]["model_interval_upper_quantile"])
        vals=validation.evaluate_validation(moms[0],moms,qlo,qhi)

        row={
            "CASE_ID":str(q["CASE_ID"]),
            "GROUP_ID":str(q["GROUP_ID"]),
            "P00":str(q["P00"]),
            "REPLICATES":reps,
            "FINITE_STATE":str(all(bool(s.diagnostics["finite_state"]) for s in sims)).lower(),
        }

        cal_pass=0
        for tid,r in cal_rows.items():
            private_tid=next(k for k,v in t_fwd.items() if v==tid)
            z=np.asarray([m[private_tid] for m in moms if private_tid in m and np.isfinite(float(m[private_tid]))],dtype=float)
            if not len(z): continue
            v=float(z.mean()); lo=float(np.quantile(z,qlo)); hi=float(np.quantile(z,qhi))
            elo=float(r["tolerance_lower"]); ehi=float(r["tolerance_upper"])
            rule=str(r.get("acceptance_rule",""))
            ok=(hi>=elo and lo<=ehi) if rule=="model_95pct_interval_overlaps_empirical_95pct_interval" else elo<=v<=ehi
            cal_pass+=int(ok)
            row[f"STATUS__{tid}"]="PASS" if ok else "FAIL"
            row[f"VALUE__{tid}"]=v; row[f"LO__{tid}"]=lo; row[f"HI__{tid}"]=hi

        dev_pass=0; sep_pass=0
        for vr in vals:
            tid=t_fwd[vr["target_id"]]
            ok=str(vr["status"])=="PASS"
            dev_pass+=int(ok)
            sep_pass+=int(ok and tid in separate)
            row[f"STATUS__{tid}"]=str(vr["status"])
            row[f"VALUE__{tid}"]=float(vr["model_value"])
            row[f"LO__{tid}"]=float(vr["model_interval_lower"])
            row[f"HI__{tid}"]=float(vr["model_interval_upper"])

        row["CAL_PASS"]=cal_pass; row["CAL_TOTAL"]=len(cal_rows)
        row["DEV_PASS"]=dev_pass; row["DEV_TOTAL"]=len(vals)
        row["SEP_PASS"]=sep_pass; row["SEP_TOTAL"]=len(separate)
        out.append(row)

    write_csv(Path(a.output),out)
    print(f"neutral cases completed: {len(out)}")

if __name__=="__main__":
    raise SystemExit(main())
