import numpy as np
from field_compounding.data.base import BaseFieldDGP, BenchmarkData
class SmokeFieldDGP(BaseFieldDGP):
 def __init__(self,seed=42,violation_severity=0.0,n_train=32,n_test=16,feature_dim=4,trace_path=None):
  super().__init__(seed,violation_severity,trace_path); self.n_train,self.n_test,self.feature_dim=n_train,n_test,feature_dim
 @property
 def name(self): return 'smoke'
 @property
 def loop_node(self): return 'scene_repr'
 def _generate(self):
  rng=np.random.default_rng(self.seed); return BenchmarkData(train={'x':rng.normal((self.n_train,self.feature_dim)),'y':rng.integers(0,2,self.n_train)},test={'x':rng.normal((self.n_test,self.feature_dim)),'y':rng.integers(0,2,self.n_test)},metadata={'feature_dim':self.feature_dim})
