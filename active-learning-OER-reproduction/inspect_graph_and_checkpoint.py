from pathlib import Path
from collections import Counter
import tensorflow as tf

root=Path(__file__).parent/'OER2022'
files=sorted((root/'initial_train'/'graph').glob('*.pdb'))
print('graph_files',len(files))

def parse(path):
 lines=path.read_text(encoding='utf-8').splitlines()
 site={}
 for line in lines:
  if line.startswith('HETATM'):
   serial=int(line[6:11]); label=line[12:16].strip(); site[serial]='A' if label.startswith('AA') else 'B' if label.startswith('BB') else 'O'
 edges=[]
 for line in lines:
  if line.startswith('CONECT'):
   vals=[int(line[i:i+5]) for i in range(6,len(line),5) if line[i:i+5].strip()]
   edges.extend((vals[0],v) for v in vals[1:])
 return site,edges
for name in ['P01_00.pdb','P07_00.pdb','P20_00.pdb','P30_00.pdb']:
 s,e=parse(root/'initial_train'/'graph'/name); print(name,Counter(tuple(sorted((s[u],s[v]))) for u,v in e),len(e))
ck=root/'initial_train'/'checkpoint'/'OO_00_full'
if (Path(str(ck)+'.index')).exists():
 print('checkpoint_variables')
 for n,shape in tf.train.list_variables(str(ck))[:80]: print(n,shape)
else: print('checkpoint_missing',ck)
