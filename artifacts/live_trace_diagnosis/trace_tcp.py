import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from rlinf.algorithms.online_residual_rl.inference_trace import iter_trace
out=Path('artifacts/live_trace_diagnosis');z=np.load(out/'trace_arrays.npz');es=list(iter_trace('artifacts/live_inference_trace/session_95a6986a-cda0-4843-abfb-6e508c309b53'));requests=[e for e in es if e['request']['type']=='infer' and not e['request'].get('warmup')];chain=[]
for j in ET.parse('/home/qccc/Flexiv/flexiv_rdk/resources/flexiv_Rizon4_kinematics.urdf').getroot().findall('joint'):
 o=j.find('origin');t=np.eye(4);t[:3,:3]=R.from_euler('xyz',list(map(float,o.get('rpy').split()))).as_matrix();t[:3,3]=list(map(float,o.get('xyz').split()));a=j.find('axis');chain.append((t,None if a is None else np.array(list(map(float,a.get('xyz').split())))))
def fk(q):
 t=np.eye(4);k=0
 for o,a in chain:
  t=t@o
  if a is not None:
   r=np.eye(4);r[:3,:3]=R.from_rotvec(a*q[k]).as_matrix();t=t@r;k+=1
 return t
pose=np.array(requests[0]['request']['action_context']['tcp_pose_wxyz']);tcp=np.eye(4);tcp[:3,3]=pose[:3];tcp[:3,:3]=R.from_quat(pose[[4,5,6,3]]).as_matrix();tool=np.linalg.inv(fk(z['q'][0]))@tcp
p=np.array([(fk(q)@tool)[:3,3] for q in z['target']]);source=requests[0]['response']['diagnostics']['full_actions_physical'][0];boot=np.array([(fk(q[:7])@tool)[:3,3] for q in source]);r=pq.read_table('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008/data/episode_000000.parquet',columns=['observation.eef_state','observation.joint_state','action']).slice(0,64).to_pylist();dq=np.array([x['observation.joint_state'] for x in r]);de=np.array([x['observation.eef_state'][:3] for x in r]);dt=np.array([x['action'][:7] for x in r]);dp=np.array([(fk(q)@tool)[:3,3] for q in dt]);
result={'bootstrap_action0_tcp_delta_mm':((boot[0]-tcp[:3,3])*1000).tolist(),'bootstrap_action19_tcp_delta_mm':((boot[19]-tcp[:3,3])*1000).tolist(),'dataset_action19_delta_mm':((dp[19]-(fk(dq[0])@tool)[:3,3])*1000).tolist(),'max_nominal_target_vs_measured_tcp_error_mm':float(np.linalg.norm(p-z['tcp'],axis=1).max()*1000),'image_state_lag':[]}
for e in requests:
 req=e['request'];tim=req['action_context']['observation_timing'];t=tim['camera_host_time_ns']['main']/1e9
 if t<z['times'][0]:continue
 aligned=np.array([np.interp(t,z['times'],z['q'][:,i]) for i in range(7)]);current=req['observation']['states'][:7];result['image_state_lag'].append({'request':req['frame'],'state_joint_max_diff_deg':float(abs(np.rad2deg(aligned-current)).max())})
replay=np.load(out/'replay_actions.npz')
result['dataset_input_same_noise_action19_delta_mm']=(((fk(replay['dataset_initial_same_noise'][19,:7])@tool)[:3,3]-(fk(dq[0])@tool)[:3,3])*1000).tolist()
(out/'tcp_summary.json').write_text(json.dumps(result,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
for j,name in enumerate('XYZ'):
 axes[j].plot(np.arange(64),(z['tcp'][:64,j]-z['tcp'][0,j])*1000,label='live measured TCP');axes[j].plot(np.arange(64),(p[:64,j]-z['tcp'][0,j])*1000,label='live model target (nominal FK)');axes[j].plot(np.arange(64),(de[:,j]-de[0,j])*1000,label='dataset measured TCP');axes[j].set_ylabel(name+' delta mm');axes[j].grid(alpha=.3)
axes[0].legend();axes[-1].set_xlabel('Frame / tick');fig.tight_layout();fig.savefig(out/'startup_tcp.png',dpi=140)
print(json.dumps({k:v for k,v in result.items() if k!='image_state_lag'},indent=2));print('image-state maximum diff',max(x['state_joint_max_diff_deg'] for x in result['image_state_lag']))
