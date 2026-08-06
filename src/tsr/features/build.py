from __future__ import annotations

import csv, hashlib, json, os, shutil
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from ..config import config_hash
from ..data.stooq import StooqCatalog, sha256_file
from ..run import RunContext, atomic_json
from .engine import compute_features
from .schema import FEATURE_NAMES, MODULE_NAME, MODULE_VERSION, payload
from .storage import npz_bytes, shard_rel, write_npz

COLS=["instrument_id","ticker","exchange","instrument_class","source_member","row_count","first_date","last_date","segment_count","source_gap_reset_count","calendar_gap_reset_count","shard_relative_path","shard_sha256","valid_counts_json"]
ERR=["instrument_id","source_member","error_type","error_message"]

def append(path:Path,row:dict,cols:list[str]):
    exists=path.exists() and path.stat().st_size
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        if not exists:w.writeheader()
        w.writerow({k:row.get(k,"") for k in cols}); f.flush(); os.fsync(f.fileno())

def loadfp(p:Path)->dict:
    x=json.loads(p.read_text());
    for k in ("archive_sha256","dataset_fingerprint_sha256","module_version"):
        if k not in x: raise ValueError(f"Upstream fingerprint missing {k}")
    return x

def fingerprint(man:pd.DataFrame,schema_hash:str,up:str)->str:
    d=hashlib.sha256((schema_hash+"|"+up+"\n").encode())
    for r in man.sort_values("instrument_id").itertuples(): d.update(f"{r.instrument_id}|{int(r.row_count)}|{'' if pd.isna(r.shard_sha256) else r.shard_sha256}\n".encode())
    return d.hexdigest()

def build(config:dict,resume:str|Path|None=None,dry_run:bool=False,max_new:int|None=None)->Path:
    if config["module"]!={"name":MODULE_NAME,"version":MODULE_VERSION}: raise ValueError("Wrong module config")
    up=config["upstream"]; exe=config["execution"]; feat=config["features"]
    archive=Path(up["archive_path"]).resolve(); manifest=Path(up["symbol_manifest_path"]).resolve(); fp=loadfp(Path(up["dataset_fingerprint_path"]).resolve())
    if sha256_file(archive)!=fp["archive_sha256"]: raise ValueError("Archive hash differs from Module 01")
    run_cfg=dict(config); run_cfg["diagnostic_dry_run"]=dry_run
    ctx=RunContext.create(MODULE_NAME,MODULE_VERSION,run_cfg,exe["runs_root"],resume_dir=resume); rd=ctx.run_dir; assert rd
    sch=payload(); atomic_json(rd/"feature_schema.json",sch)
    cat=StooqCatalog(archive,manifest,invalid_policy="drop"); uni=cat.list_instruments(require_rows=False).sort_values("instrument_id").reset_index(drop=True)
    if dry_run:uni=uni.head(int(exe.get("dry_run_max_files",25)))
    partial=rd/"feature_manifest.partial.csv"; errors=rd/"file_errors.csv"
    done=set(pd.read_csv(partial).instrument_id.astype(str)) if partial.exists() and partial.stat().st_size else set()
    new=0
    for r in uni.itertuples(index=False):
        iid=str(r.instrument_id)
        if iid in done:continue
        try:
            if int(r.valid_row_count)==0:
                row={"instrument_id":iid,"ticker":r.ticker,"exchange":r.exchange,"instrument_class":r.instrument_class,"source_member":r.source_member,"row_count":0,"segment_count":0,"source_gap_reset_count":0,"calendar_gap_reset_count":0,"valid_counts_json":json.dumps({k:0 for k in FEATURE_NAMES},sort_keys=True)}
            else:
                bars=cat.load_instrument(iid); res=compute_features(bars,int(feat.get("max_calendar_gap_days",30))); rel=shard_rel(iid); sh=write_npz(rd/rel,res.arrays); vc={k:int(np.isfinite(res.arrays[k]).sum()) for k in FEATURE_NAMES}
                row={"instrument_id":iid,"ticker":r.ticker,"exchange":r.exchange,"instrument_class":r.instrument_class,"source_member":r.source_member,"row_count":len(bars),"first_date":bars.date.iloc[0].strftime("%Y-%m-%d"),"last_date":bars.date.iloc[-1].strftime("%Y-%m-%d"),"segment_count":res.segment_count,"source_gap_reset_count":res.source_gap_reset_count,"calendar_gap_reset_count":res.calendar_gap_reset_count,"shard_relative_path":str(rel),"shard_sha256":sh,"valid_counts_json":json.dumps(vc,sort_keys=True,separators=(",",":"))}
            append(partial,row,COLS)
        except Exception as e:
            append(errors,{"instrument_id":iid,"source_member":r.source_member,"error_type":type(e).__name__,"error_message":str(e)},ERR)
        new+=1
        if new%50==0:ctx.save_checkpoint({"completed_instruments":len(done)+new,"total_instruments":len(uni),"last_instrument_id":iid})
        if max_new is not None and new>=max_new:
            ctx.interrupt("bounded chunk completed"); return rd
    man=pd.read_csv(partial).sort_values("instrument_id").reset_index(drop=True); man.to_csv(rd/"feature_manifest.csv",index=False,lineterminator="\n")
    if not errors.exists():pd.DataFrame(columns=ERR).to_csv(errors,index=False)
    total=int(man.row_count.sum()); counts=Counter()
    for x in man.valid_counts_json.fillna("{}"):counts.update(json.loads(x))
    pd.DataFrame([{"feature_name":k,"valid_count":counts[k],"missing_count":total-counts[k],"valid_fraction":counts[k]/total if total else None} for k in FEATURE_NAMES]).to_csv(rd/"feature_quality.csv",index=False)
    fph=fingerprint(man,sch["schema_sha256"],fp["dataset_fingerprint_sha256"])
    atomic_json(rd/"dataset_fingerprint.json",{"module_version":MODULE_VERSION,"schema_sha256":sch["schema_sha256"],"upstream_dataset_fingerprint_sha256":fp["dataset_fingerprint_sha256"],"archive_sha256":fp["archive_sha256"],"feature_dataset_fingerprint_sha256":fph,"config_hash":ctx.manifest["config_hash"]})
    errn=len(pd.read_csv(errors)); upstream_rows=int(cat.manifest.valid_row_count.sum())
    summary={"module_name":MODULE_NAME,"module_version":MODULE_VERSION,"dry_run":dry_run,"expected_instruments":len(uni),"completed_instruments":len(man),"nonempty_instruments":int((man.row_count>0).sum()),"empty_instruments":int((man.row_count==0).sum()),"total_feature_rows":total,"upstream_valid_rows":upstream_rows,"row_count_matches_upstream":dry_run or total==upstream_rows,"operational_errors":errn,"feature_count":len(FEATURE_NAMES),"total_segments":int(man.segment_count.sum()),"source_gap_reset_count":int(man.source_gap_reset_count.sum()),"calendar_gap_reset_count":int(man.calendar_gap_reset_count.sum()),"max_calendar_gap_days":int(feat.get("max_calendar_gap_days",30)),"feature_dataset_fingerprint_sha256":fph,"feature_schema_sha256":sch["schema_sha256"]}
    atomic_json(rd/"summary.json",summary); ctx.complete(**summary); return rd

def verify(config:dict,reference:str|Path,output:str|Path,resume_state:str|Path|None=None,max_new:int|None=None)->Path:
    ref=Path(reference).resolve(); man=pd.read_csv(ref/"feature_manifest.csv"); up=config["upstream"]; cat=StooqCatalog(up["archive_path"],up["symbol_manifest_path"],invalid_policy="drop")
    state=Path(resume_state).resolve() if resume_state else Path(str(output)+".partial.csv"); done=set(pd.read_csv(state).instrument_id) if state.exists() and state.stat().st_size else set(); new=0
    for r in man.itertuples(index=False):
        if r.instrument_id in done:continue
        ok=True; actual=""
        if int(r.row_count)>0:
            arrays=compute_features(cat.load_instrument(r.instrument_id),int(config["features"].get("max_calendar_gap_days",30))).arrays
            actual=hashlib.sha256(npz_bytes(arrays)).hexdigest(); ok=actual==r.shard_sha256
        append(state,{"instrument_id":r.instrument_id,"expected_sha256":r.shard_sha256 if int(r.row_count)>0 else "","actual_sha256":actual,"identical":ok},["instrument_id","expected_sha256","actual_sha256","identical"])
        new+=1
        if max_new is not None and new>=max_new:return state
    checks=pd.read_csv(state); result={"reference_run":str(ref),"instruments_verified":len(checks),"expected_instruments":len(man),"all_shards_identical":bool(checks.identical.astype(str).str.lower().eq("true").all()),"deterministic":len(checks)==len(man) and bool(checks.identical.astype(str).str.lower().eq("true").all())}
    atomic_json(Path(output),result); return Path(output)

def publish(run:str|Path,target:str|Path)->Path:
    src=Path(run).resolve(); dst=Path(target).resolve(); s=json.loads((src/"summary.json").read_text()); m=json.loads((src/"run_manifest.json").read_text())
    if s["dry_run"] or s["operational_errors"] or not s["row_count_matches_upstream"] or m["status"]!="completed":raise ValueError("Run is not publishable")
    tmp=Path(str(dst)+".tmp"); shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True)
    for name in ("feature_schema.json","feature_manifest.csv","feature_quality.csv","file_errors.csv","dataset_fingerprint.json","summary.json","run_manifest.json"):shutil.copy2(src/name,tmp/name)
    man=pd.read_csv(src/"feature_manifest.csv")
    for r in man.itertuples(index=False):
        if int(r.row_count)==0:continue
        a=src/r.shard_relative_path;b=tmp/r.shard_relative_path;b.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(a,b)
        except OSError:shutil.copy2(a,b)
    (tmp/"SOURCE_RUN.txt").write_text(str(src)+"\n"); shutil.rmtree(dst,ignore_errors=True); os.replace(tmp,dst); return dst
