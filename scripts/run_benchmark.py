import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from field_compounding.evaluation.runner import resolve_config_path, run_module_benchmark
R=[f"{i:02d}" for i in range(3,15)]
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--module",required=True); p.add_argument("--config"); p.add_argument("--output"); p.add_argument("--n-trials",type=int); p.add_argument("--fast",action="store_true"); p.add_argument("--configs-dir",default=str(ROOT/"configs")); a=p.parse_args()
 m="all" if a.module.lower()=="all" else str(int(a.module)).zfill(2); fast=a.fast or os.environ.get("FIELD_COMPOUNDING_FAST","0")=="1"
 for mod in (R if m=="all" else [m]):
  cfg=Path(a.config) if a.config else resolve_config_path(mod,Path(a.configs_dir)); out=Path(a.output) if a.output and m!="all" else ROOT/"results"/f"section_{mod}.json"
  r=run_module_benchmark(cfg,out,fast=fast,n_trials=a.n_trials); print(r["module_id"],r["n_successful"],r["n_trials"])
