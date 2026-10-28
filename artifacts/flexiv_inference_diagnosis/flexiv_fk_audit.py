import xml.etree.ElementTree as ET
import numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from pathlib import Path
xml=ET.parse('/home/qccc/Flexiv/flexiv_rdk/resources/flexiv_Rizon4_kinematics.urdf').getroot()
chain=[]
for j in xml.findall('joint'):
 o=j.find('origin'); t=np.eye(4); t[:3,:3]=R.from_euler('xyz',list(map(float,o.get('rpy').split()))).as_matrix();t[:3,3]=list(map(float,o.get('xyz').split())); a=j.find('axis');chain.append((t,np.array(list(map(float,a.get('xyz').split()))) if a is not None else None))
def fk(q):
 t=np.eye(4); k=0
 for o,a in chain:
  t=t@o
  if a is not None:
   m=np.eye(4);m[:3,:3]=R.from_rotvec(a*q[k]).as_matrix();t=t@m;k+=1
 return t
z=np.load('/tmp/flexiv_offline_compare.npz');p=z['pred'];g=z['truth'];s=[];eef=[];tools=[];errors=[];pd=[];gd=[]
root=Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008/data')
for ep in sorted(set(z['episode'])):
 t=pq.read_table(root/(ep+'.parquet'),columns=['observation.joint_state','observation.eef_state'])
 for i in z['frame'][z['episode']==ep]:s.append(t['observation.joint_state'][int(i)].as_py());eef.append(t['observation.eef_state'][int(i)].as_py())
for q,e,pp,gg in zip(s,eef,p,g):
 tcp=np.eye(4);tcp[:3,:3]=R.from_rotvec(e[3:]).as_matrix();tcp[:3,3]=e[:3]
 tool=np.linalg.inv(fk(q))@tcp;tools.append(tool)
 a=np.stack([(fk(x[:7])@tool)[:3,3] for x in pp]);b=np.stack([(fk(x[:7])@tool)[:3,3] for x in gg]);errors.append(a-b);pd.append(a-e[:3]);gd.append(b-e[:3])
errors=np.array(errors)*1000;pd=np.array(pd)*1000;gd=np.array(gd)*1000;tools=np.array(tools)
print('inferred_tool_xyz mean/std m',tools[:,:3,3].mean(0),tools[:,:3,3].std(0))
print('tool_orientation_std deg',R.from_matrix(tools[:,:3,:3]).as_rotvec().std(0)*180/np.pi)
for name,mask in [('all',np.ones(len(p),bool)),('start',z['frame']==0)]:
 er=errors[mask]; print(name,'TCP_distance_error_mm mean/median/p95/max',*[float(x) for x in (np.linalg.norm(er,axis=-1).mean(),np.quantile(np.linalg.norm(er,axis=-1),.5),np.quantile(np.linalg.norm(er,axis=-1),.95),np.linalg.norm(er,axis=-1).max())])
 for h in [0,19,31]:print('h',h,'pred_xyz_delta_mm',pd[mask,h].mean(0),'recorded_xyz_delta_mm',gd[mask,h].mean(0),'position_error_mm',np.linalg.norm(er[:,h],axis=-1).mean())
