from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
import pandas as pd

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)

def load_surface(path,label):
    df=pd.read_csv(Path(path)/"surface.csv")
    req={"virtual_index","P00","max_i","rms_i","uncertainty","residual_score","nroy"}
    missing=req-set(df.columns)
    if missing: raise RuntimeError(f"{label}: missing {sorted(missing)}")
    xcols=sorted(c for c in df.columns if c.startswith("X"))
    keep=["virtual_index","P00","max_i","rms_i","uncertainty","residual_score","nroy"]+xcols
    out=df[keep].copy()
    return out.rename(columns={
        "max_i":f"max_i__{label}","rms_i":f"rms_i__{label}",
        "uncertainty":f"uncertainty__{label}","residual_score":f"residual_score__{label}",
        "nroy":f"nroy__{label}"})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extratrees",required=True)
    ap.add_argument("--hgb",required=True)
    ap.add_argument("--catboost",required=True)
    ap.add_argument("--output-dir",required=True)
    a=ap.parse_args()
    et=load_surface(a.extratrees,"extratrees")
    hg=load_surface(a.hgb,"hgb")
    cb=load_surface(a.catboost,"catboost")
    xcols=sorted(c for c in et.columns if c.startswith("X"))
    for frame,label in ((hg,"hgb"),(cb,"catboost")):
        chk=et[["virtual_index","P00"]+xcols].merge(
            frame[["virtual_index","P00"]+xcols],
            on=["virtual_index","P00"],suffixes=("_a","_b"),validate="one_to_one")
        for c in xcols:
            if not np.allclose(chk[f"{c}_a"],chk[f"{c}_b"],rtol=0,atol=1e-12):
                raise RuntimeError(f"Virtual design mismatch: {label}:{c}")
    m=et.copy()
    hgcols=[c for c in hg.columns if "__hgb" in c]
    cbcols=[c for c in cb.columns if "__catboost" in c]
    m=m.merge(hg[["virtual_index","P00"]+hgcols],on=["virtual_index","P00"],validate="one_to_one")
    m=m.merge(cb[["virtual_index","P00"]+cbcols],on=["virtual_index","P00"],validate="one_to_one")
    A=m[["max_i__extratrees","max_i__hgb","max_i__catboost"]].to_numpy(float)
    m["nroy_count"]=(A<=3).sum(1)
    m["all_three_nroy"]=m["nroy_count"].eq(3)
    m["two_of_three_nroy"]=m["nroy_count"].ge(2)
    m["boosting_pair_nroy"]=(m["max_i__hgb"]<=3)&(m["max_i__catboost"]<=3)
    m["median_i"]=np.median(A,axis=1); m["mean_i"]=A.mean(1); m["disagreement_i"]=A.std(1)
    m["boosting_worst_i"]=np.maximum(m["max_i__hgb"],m["max_i__catboost"])
    m["boosting_mean_i"]=(m["max_i__hgb"]+m["max_i__catboost"])/2
    consensus=[]; chosen=set()
    def add(indices,bucket,limit=8):
        n=0
        for idx in indices:
            row=m.iloc[int(idx)]
            vi=int(row["virtual_index"])
            if vi in chosen: continue
            chosen.add(vi); d=row.to_dict(); d["selection_source"]="consensus"; d["bucket"]=bucket
            consensus.append(d); n+=1
            if n>=limit: break
    pair=np.where(m["boosting_pair_nroy"].to_numpy(bool))[0]
    if len(pair):
        pair=pair[np.lexsort((m.iloc[pair]["boosting_mean_i"],m.iloc[pair]["boosting_worst_i"]))]
    add(pair,"boosting_pair_plausible")
    maj=np.where(m["two_of_three_nroy"].to_numpy(bool))[0]
    if len(maj):
        maj=maj[np.lexsort((m.iloc[maj]["mean_i"],m.iloc[maj]["median_i"]))]
    add(maj,"two_of_three_plausible")
    add(np.argsort(-m["disagreement_i"].to_numpy(float)),"model_disagreement")
    add(np.argsort(np.abs(m["median_i"].to_numpy(float)-3)),"consensus_boundary")
    for i,r in enumerate(consensus,1): r["queue_order"]=i
    queues={}
    for label,path in (("extratrees",a.extratrees),("hgb",a.hgb),("catboost",a.catboost)):
        queues[label]=pd.read_csv(Path(path)/"queue.csv")
    membership=[]; candidates={}
    sources={**queues,"consensus":pd.DataFrame(consensus)}
    for source,q in sources.items():
        for _,r in q.iterrows():
            vi=int(r["virtual_index"])
            membership.append({"virtual_index":vi,"selection_source":source,
                "source_queue_order":int(r.get("queue_order",0)),"bucket":str(r.get("bucket",""))})
            candidates.setdefault(vi,r.to_dict())
    mem=pd.DataFrame(membership); unique=[]
    for order,vi in enumerate(sorted(candidates),1):
        r=dict(candidates[vi]); mm=mem[mem["virtual_index"].eq(vi)]
        r["queue_order"]=order; r["selection_source"]="wave04_unique"; r["bucket"]="deduplicated_multi_source"
        r["selection_sources"]=";".join(sorted(mm["selection_source"].unique()))
        unique.append(r)
    out=Path(a.output_dir)
    write_csv(out/"consensus_queue.csv",consensus)
    write_csv(out/"unique_queue.csv",unique)
    write_csv(out/"selection_membership.csv",membership)
    write_csv(out/"summary.csv",[{
        "virtual_points":len(m),"all_three_nroy":int(m["all_three_nroy"].sum()),
        "two_of_three_nroy":int(m["two_of_three_nroy"].sum()),
        "boosting_pair_nroy":int(m["boosting_pair_nroy"].sum()),
        "consensus_queue_points":len(consensus),"theoretical_points_without_dedup":len(membership),
        "unique_points_after_dedup":len(unique),"duplicate_candidates_avoided":len(membership)-len(unique),
        "status":"PUBLIC_WAVE04_SURROGATE_CONSENSUS_COMPLETE"}])
if __name__=="__main__": raise SystemExit(main())
