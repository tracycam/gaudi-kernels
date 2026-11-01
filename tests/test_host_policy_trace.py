import json
import pytest

from tools.validation.executor.host_policy import submit_threads


def test_submission_owner_must_belong_to_current_worker(tmp_path):
    path = tmp_path/'trace.json'
    path.write_text(json.dumps({'traceEvents': [
        dict(name='NativePlanSubmit', ph='B', pid=100, tid=200),
        dict(name='NativePlanSubmit', ph='B', pid=999, tid=888),
        dict(name='NativePlanSubmit', ph='E', pid=100, tid=300),
        dict(name='Other', ph='B', pid=100, tid=400)]}))
    assert submit_threads(path, 100) == {100, 200}
    with pytest.raises(ValueError, match='no native submissions'):
        submit_threads(path, 777)


def test_missing_trace_fails_before_policy_change(tmp_path):
    path = tmp_path/'empty.json'
    path.write_text(json.dumps({'traceEvents': []}))
    with pytest.raises(ValueError, match='no native submissions'):
        submit_threads(path, 100)
