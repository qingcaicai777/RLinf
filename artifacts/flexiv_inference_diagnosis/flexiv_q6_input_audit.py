import json,io
from pathlib import Path
import numpy as np,torch,yaml,pyarrow.parquet as pq
from PIL import Image
from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model
cfg=yaml.safe_load(Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml').read_text());r=pq.read_table('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008/data/episode_000000.parquet',columns=['observation.gripper_state']).slice(0,1).to_pylist()[0]
home=np.array(cfg['home']['left_joint_positions_rad']+r['observation.gripper_state'],np.float32)
values=[.954642,.963758,.965024,.967444,.970,.980]
cap=Path('/home/qccc/projects/RLinf/artifacts/inference_images/session_1791623407637087412_1092819');ims=[np.array(Image.open(cap/f'tick_000000_{k}.png').convert('RGB')) for k in ['main','wrist']]
states=np.tile(home,(len(values),1));states[:,5]=values
obs={'states':states,'main_images':np.stack([ims[0]]*len(values)),'wrist_images':np.stack([ims[1]]*len(values)),'task_descriptions':[cfg['recording']['task']]*len(values)}
m=load_stage1_model('examples/sft/config/flexiv_joint_stage1_inference.yaml','/home/qccc/checkpoints/myq/connecter_1010/full_weights.pt','cuda:0',norm_stats_path='/home/qccc/checkpoints/myq/connecter_1010/peg_insertion_joint_absolute_1008_v4/norm_stats.json',num_flow_steps=10,use_cuda_graph=False,action_representation='joint_absolute');out=[]
for seed in [42,43,44]:
 noise=torch.randn(1,32,32,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed)).repeat(len(values),1,1)
 with torch.no_grad():p,_=m.predict_action_batch(obs,noise=noise)
 out.append(p.cpu().numpy());print('seed',seed,'q6_target_h15',p[:,15,5].cpu().tolist(),flush=True)
out=np.array(out);s=json.loads(Path('/home/qccc/checkpoints/myq/connecter_1010/peg_insertion_joint_absolute_1008_v4/norm_stats.json').read_text())['norm_stats']['state'];lo=s['q01'][5];hi=s['q99'][5];norm=(states[:,5]-lo)/(hi-lo+1e-6)*2-1;bins=np.digitize(norm,bins=np.linspace(-1,1,257)[:-1])-1
results=[]
for i,v in enumerate(values):
 row={'input_q6':v,'normalized_q6':float(norm[i]),'q6_state_bin':int(bins[i]),'pred_q6_h15_mean':float(out[:,i,15,5].mean()),'pred_q6_delta_h15_deg_mean':float((out[:,i,15,5].mean()-v)*180/np.pi)};results.append(row);print(row,flush=True)
Path('/home/qccc/projects/RLinf/artifacts/flexiv_inference_diagnosis/q6_input_sweep.json').write_text(json.dumps(results,indent=2))
