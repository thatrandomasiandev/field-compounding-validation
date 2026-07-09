"""Evaluation core tests."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import numpy as np
import pytest
from field_compounding.evaluation.metrics import mae, normalized_score, rmse
from field_compounding.evaluation.report import format_table_row, load_results, merge_results, validate_results_schema, write_results
from field_compounding.evaluation.runner import aggregate_trials, get_evaluation_seeds, get_violation_values, load_config, resolve_config_path, run_module_benchmark, run_single_trial
from field_compounding.evaluation.statistical_tests import bootstrap_ci, effect_size_cohens_d, paired_bootstrap_test, permutation_test
from field_compounding.theory.violation_severity import get_violation_severity
ROOT=Path(__file__).resolve().parents[1]; SMOKE=ROOT/"tests/fixtures/module_smoke.yaml"

def test_bootstrap_ci_mean_inside():
 v=np.array([1.,2.,3.,4.,5.]); lo,hi=bootstrap_ci(v, n_bootstrap=300, seed=0); assert lo<=v.mean()<=hi

def test_bootstrap_empty(): assert bootstrap_ci(np.array([]))==(0.,0.)

def test_permutation_identical(): a=np.array([1.,1.1,.9]); assert permutation_test(a,a.copy(),200,0)>.05

def test_permutation_separated():
 a=np.array([5.,5.1,4.9,5.05,4.95]); b=np.array([0.,.1,-.1,.05,-.05]); assert permutation_test(a,b,500,0)<.05

def test_rmse_zero(): y=np.array([1.,2.,3.]); assert rmse(y,y)==pytest.approx(0.)

def test_mae_normalized():
 assert mae(np.array([0.,1.]),np.array([.1,1.1]))==pytest.approx(.1); assert normalized_score(.75,1.,.5)==pytest.approx(.5)

def test_aggregate(): assert aggregate_trials([{"pehe":.5},{"pehe":.6}],["pehe"])["pehe"]["n_successful"]==2

def test_config_helpers():
 c=load_config(SMOKE); assert c["module_id"]==7; assert get_evaluation_seeds(c,5)==[0,1,2,3,4]; assert get_violation_values(c)==[0.,1.]; assert get_violation_severity(7,.5)==pytest.approx(.5)

def test_resolve_config_09(): assert resolve_config_path(9,ROOT/"configs").name=="module_09.yaml"

@patch("field_compounding.evaluation.runner.run_trial")
@patch("field_compounding.evaluation.runner.build_model")
@patch("field_compounding.evaluation.runner.instantiate_dgp")
def test_run_single_trial(md,mb,mr):
 c=load_config(SMOKE); m=c["models"][0]; md.return_value.generate.return_value=MagicMock(metadata={}); mb.return_value=object(); mr.return_value={"accuracy":.8}
 assert run_single_trial(7,m["name"],m["class"],{},c["dgp"]["class"],c["dgp"]["params"],0,.1,True)["accuracy"]==.8

def test_report(tmp_path):
 p={"module_id":7,"n_trials":1,"n_successful":1,"models":{},"metadata":{}}; f=tmp_path/"s.json"; write_results(p,f); assert load_results(f)["module_id"]==7; assert merge_results([f])["summary"]["n_modules"]==1; assert validate_results_schema(p)==[]; assert "0.1" in format_table_row("M",{"x":{"mean":.1,"std":.01}})

def test_cohens_d(): assert effect_size_cohens_d(np.array([1.,2.]),np.array([3.,4.]))<0

def test_paired_bootstrap():
 p,d,w=paired_bootstrap_test(np.array([.9,.85]),np.array([.7,.72]),100,0); assert 0<p<=1 and d>0

@patch("field_compounding.evaluation.runner.run_single_trial")
def test_run_module_benchmark(mt,tmp_path):
 mt.return_value={"accuracy":.75}; out=tmp_path/"s.json"; r=run_module_benchmark(SMOKE,out,fast=True,n_trials=2); assert r["n_successful"]==2; assert validate_results_schema(json.loads(out.read_text()))==[]
