"""Export the observed errors; no invented floor for zero differences."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
for ax,name,title in zip(axes,['native-vs-a8','native-vs-a8-swa'],['CPU-native vs A8, vendor attention','CPU-native vs A8 + GP/down vector kernels, both FP32 SWA']):
 d=json.loads((a.root/(name+'.json')).read_text())
 for stage,label in [('self_attn.qkv_proj','QKV'),('self_attn.attn','Attention'),('mlp','MLP branch'),('residual','Residual branch')]:
  rows=[r for r in d['per_layer_max_rank'] if r['stage']==stage]
  ax.plot([r['layer'] for r in rows],[r['max_rank_relative_l2'] or np.nan for r in rows],label=label,linewidth=1.2,marker='.',markersize=3)
 for r in d['ranks'][0]['routes']:
  if not r['expert_sets_match']:ax.axvline(r['layer'],color='black',alpha=.16,linewidth=.8)
 ax.set_yscale('log');ax.set_ylim(1e-12,1);ax.set_ylabel('Max-rank relative L2');ax.set_title(title,loc='left',fontsize=11);ax.grid(alpha=.15);ax.legend(ncol=4,fontsize=8)
axes[-1].set_xlabel('Layer index (0-based); gray lines: expert set differs; zero errors omitted')
fig.savefig(a.root/'propagation.png',dpi=180);fig.savefig(a.root/'propagation.svg')
