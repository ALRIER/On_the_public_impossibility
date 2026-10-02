from __future__ import annotations
import argparse,re
from pathlib import Path
import pandas as pd

FILES={"evidence.csv","scales.csv","screen_map.csv","parameter_bounds.csv","residual_targets.csv","bundle_manifest.csv"}
RX={
"X":re.compile(r"^X\d{3}$"),"Y":re.compile(r"^Y\d{3}$"),"T":re.compile(r"^T\d{3}$"),
"P":re.compile(r"^P\d{2}$"),"G":re.compile(r"^G\d{4}$"),"R":re.compile(r"^R\d{6}$")
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",default="inputs/r04"); a=ap.parse_args()
    root=Path(a.input_dir); found={p.name for p in root.glob("*.csv")}
    if found!=FILES: raise SystemExit(f"Bundle file mismatch: missing={sorted(FILES-found)} extra={sorted(found-FILES)}")
    ev=pd.read_csv(root/"evidence.csv",nrows=32)
    meta={"ROW_ID","GROUP_ID","LOCKBOX","P00"}
    for col in ev.columns:
        if col not in meta and not (RX["X"].match(col) or RX["Y"].match(col)):
            raise SystemExit(f"Non-neutral evidence column: {col}")
    if not ev["ROW_ID"].astype(str).map(lambda x:bool(RX["R"].match(x))).all(): raise SystemExit("ROW_ID invalid")
    if not ev["GROUP_ID"].astype(str).map(lambda x:bool(RX["G"].match(x))).all(): raise SystemExit("GROUP_ID invalid")
    if not ev["P00"].astype(str).map(lambda x:bool(RX["P"].match(x))).all(): raise SystemExit("P00 invalid")
    sc=pd.read_csv(root/"scales.csv")
    if list(sc.columns)!=["response_id","scale"]: raise SystemExit("scales schema invalid")
    sm=pd.read_csv(root/"screen_map.csv")
    if set(sm.columns)!={"target_id","value_response","lo_response","hi_response","lower","upper","rule"}: raise SystemExit("screen_map schema invalid")
    if not set(sm["rule"].astype(str)).issubset({"VALUE","INTERVAL"}): raise SystemExit("rule invalid")
    pb=pd.read_csv(root/"parameter_bounds.csv")
    if set(pb.columns)!={"feature_id","lower","upper"}: raise SystemExit("parameter_bounds schema invalid")
    rt=pd.read_csv(root/"residual_targets.csv")
    if list(rt.columns)!=["target_id"]: raise SystemExit("residual_targets schema invalid")
    print("Neutral bundle validation PASS")

if __name__=="__main__": raise SystemExit(main())
