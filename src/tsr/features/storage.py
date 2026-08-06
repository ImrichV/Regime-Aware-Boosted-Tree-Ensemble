from __future__ import annotations

import hashlib, io, json, os, zipfile
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from .schema import FEATURE_NAMES, META_NAMES


def shard_rel(instrument_id:str)->Path:
    h=hashlib.sha256(instrument_id.encode()).hexdigest(); return Path("feature_shards")/h[:2]/f"{h}.npz"


def npz_bytes(arrays:Mapping[str,np.ndarray])->bytes:
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w") as z:
        for name in sorted(arrays):
            b=io.BytesIO(); np.lib.format.write_array(b,np.ascontiguousarray(arrays[name]),allow_pickle=False)
            info=zipfile.ZipInfo(f"{name}.npy",(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_STORED; info.create_system=3; info.external_attr=0o600<<16
            z.writestr(info,b.getvalue(),compress_type=zipfile.ZIP_STORED)
    return out.getvalue()


def write_npz(path:Path,arrays:Mapping[str,np.ndarray])->str:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp"); data=npz_bytes(arrays); tmp.write_bytes(data); os.replace(tmp,path); return hashlib.sha256(data).hexdigest()


def file_hash(path:Path)->str:
    d=hashlib.sha256()
    with path.open("rb") as f:
        while x:=f.read(8*1024*1024): d.update(x)
    return d.hexdigest()


class FeatureStore:
    def __init__(self,root:str|Path):
        self.root=Path(root).resolve(); self.manifest=pd.read_csv(self.root/"feature_manifest.csv"); self.rows=self.manifest.set_index("instrument_id",drop=False)
        self.schema=json.loads((self.root/"feature_schema.json").read_text())
    def load_arrays(self,instrument_id:str,verify:bool=False)->dict[str,np.ndarray]:
        row=self.rows.loc[instrument_id]; p=self.root/row.shard_relative_path
        if verify and file_hash(p)!=row.shard_sha256: raise ValueError("Shard hash mismatch")
        with np.load(p,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
        expected=set(META_NAMES)|set(FEATURE_NAMES)
        if expected-set(arrays): raise ValueError("Shard arrays missing")
        return arrays
    def load_instrument(self,instrument_id:str,verify:bool=False)->pd.DataFrame:
        a=self.load_arrays(instrument_id,verify); df=pd.DataFrame({k:a[k] for k in (*META_NAMES,*FEATURE_NAMES)}); df.insert(0,"instrument_id",instrument_id); df.date=pd.to_datetime(df.date.astype(str),format="%Y%m%d"); return df
