from pathlib import Path
import json, hashlib, zipfile
import numpy as np, pandas as pd
from tsr.features.engine import compute_features
from tsr.features.storage import npz_bytes, write_npz, FeatureStore
from tsr.features.schema import FEATURE_NAMES, payload


def bars(n=300):
    d=pd.bdate_range('2020-01-01',periods=n);c=100+np.arange(n,dtype=float)
    return pd.DataFrame({'date':d,'open':c-.5,'high':c+1,'low':c-1,'close':c,'volume':1000+np.arange(n),'source_row_number':np.arange(2,n+2)})

def test_schema_and_formulas():
    assert len(FEATURE_NAMES)==36 and len(set(FEATURE_NAMES))==36 and payload()['schema_sha256']
    b=bars();a=compute_features(b).arrays;i=260
    assert np.isclose(a['return_20'][i],b.close.iloc[i]/b.close.iloc[i-20]-1)
    assert np.isclose(a['sma_distance_20'][i],b.close.iloc[i]/b.close.iloc[i-19:i+1].mean()-1)
    assert a['return_20'].dtype==np.float32 and a['date'].dtype==np.int32

def test_prefix_causality():
    b=bars();f=compute_features(b).arrays;p=compute_features(b.iloc[:220]).arrays
    for k in f:
        if np.issubdtype(f[k].dtype,np.floating):np.testing.assert_allclose(f[k][:220],p[k],rtol=0,atol=0,equal_nan=True)
        else:np.testing.assert_array_equal(f[k][:220],p[k])

def test_segment_reset():
    b=bars(80);b.loc[40:,'source_row_number']+=1;a=compute_features(b).arrays
    assert a['segment_reset'][40]==1 and a['history_bars'][40]==1 and np.isnan(a['return_1'][40])

def test_deterministic_npz(tmp_path):
    a={'b':np.array([1.,np.nan],np.float32),'a':np.array([1,2],np.int32)}
    assert npz_bytes(a)==npz_bytes(a)
    p=tmp_path/'x.npz';h=write_npz(p,a);assert h==hashlib.sha256(p.read_bytes()).hexdigest()
