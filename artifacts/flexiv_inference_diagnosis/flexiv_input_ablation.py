import numpy as np,torch,pyarrow.parquet as pq,yaml,io
from PIL import Image
from pathlib import Path
from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model
root=Path('/home/qccc/projects/RLinf/artifacts/inference_images/session_1791623407637087412_1092819')
t=pq.read_table('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008/data/episode_000000.parquet',columns=['observation.joint_state','observation.gripper_state','observation.image','observation.wrist_image','action'])
r=t.slice(0,1).to_pylist()[0];trainq=np.array(r['observation.joint_state']+r['observation.gripper_state'],np.float32); config=yaml.safe_load(Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml').read_text());home=np.array(config['home']['left_joint_positions_rad']+[.06],np.float32)
ti=[np.array(Image.open(io.BytesIO(r[k])).convert('RGB')) for k in ['observation.image','observation.wrist_image']];li=[np.array(Image.open(root/f'tick_000000_{k}.png').convert('RGB')) for k in ['main','wrist']]
gt=np.array(t['action'].slice(0,32).to_pylist())
obs={'states':np.stack([trainq,home,trainq,home]),'main_images':np.stack([ti[0],ti[0],li[0],li[0]]),'wrist_images':np.stack([ti[1],ti[1],li[1],li[1]]),'task_descriptions':[config['recording']['task']]*4}
print('initial state difference deg',(home[:7]-trainq[:7])*180/np.pi,flush=True)
m=load_stage1_model('examples/sft/config/flexiv_joint_stage1_inference.yaml','/home/qccc/checkpoints/myq/connecter_1010/full_weights.pt','cuda:0',norm_stats_path='/home/qccc/checkpoints/myq/connecter_1010/peg_insertion_joint_absolute_1008_v4/norm_stats.json',num_flow_steps=10,use_cuda_graph=False,action_representation='joint_absolute');outs=[]
for seed in [42,43,44]:
 gen=torch.Generator(device='cuda:0').manual_seed(seed);noise=torch.randn(1,32,32,device='cuda:0',generator=gen).repeat(4,1,1)
 with torch.inference_mode():p,_=m.predict_action_batch(obs,noise=noise)
 outs.append(p.cpu().numpy())
p=np.array(outs);labels=['train_images+train_state','train_images+home_state','online_images+train_state','online_images+home_state']
for i,name in enumerate(labels):
 print(name,'MAE_vs_demo_deg',abs(p[:,i,:,:7]-gt[None,:,:7]).mean()*180/np.pi,'at_h19_command_move_deg',((p[:,i,19,:7]-obs['states'][i,:7])*180/np.pi).mean(0),flush=True)
print('image_only_changed_prediction MAE/max deg',abs(p[:,2,:,:7]-p[:,0,:,:7]).mean()*180/np.pi,abs(p[:,2,:,:7]-p[:,0,:,:7]).max()*180/np.pi)
print('state_only_changed_prediction MAE/max deg',abs(p[:,1,:,:7]-p[:,0,:,:7]).mean()*180/np.pi,abs(p[:,1,:,:7]-p[:,0,:,:7]).max()*180/np.pi)
np.savez('/tmp/flexiv_input_ablation.npz',pred=p,truth=gt,states=obs['states'])
