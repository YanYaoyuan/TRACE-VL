import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('stage1_data',Path(__file__).with_name('stage1_data.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_upstream_ids_preserve_distinct_questions_and_stay_stable():
    a={'id':'shared','video':'v.mp4','question':'What happens?','options':['run','sit'],'answer':0}
    b={**a,'question':'Where is this?','options':['inside','outside'],'answer':1}
    rows,duplicates=module.canonicalize([a,b,a])
    assert len(rows)==2 and duplicates==1 and len({r['id'] for r in rows})==2
    reordered,_=module.canonicalize([b,a])
    assert {r['id'] for r in rows}=={r['id'] for r in reordered}
    assert all(r['upstream_record_id']=='shared' for r in rows)


def test_gold_conflict_is_rejected():
    row={'id':'shared','video':'v.mp4','question':'What happens?','options':['run','sit'],'answer':0}
    with pytest.raises(ValueError,match='Conflicting gold'):
        module.canonicalize([row,{**row,'answer':1}])
