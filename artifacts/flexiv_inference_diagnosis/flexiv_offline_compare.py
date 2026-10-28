import io, json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model, OpenPIRLInfStage1Adapter

root=Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008')
recipe='examples/sft/config/flexiv_joint_stage1_inference.yaml'
ckpt='/home/qccc/checkpoints/myq/connecter_1010/full_weights.pt'
stats='/home/qccc/checkpoints/myq/connecter_1010/peg_insertion_joint_absolute_1008_v4/norm_stats.json'
prompt=json.loads((root/'meta/tasks.jsonl').read_text().splitlines()[0])['task']
files=sorted((root/'data').glob('episode_*.parquet'))
samples=[]
start=time.time()
for fi,p in enumerate(files):
    tab=pq.read_table(p, columns=['observation.joint_state','observation.gripper_state','observation.image','observation.wrist_image','action'])
    rows=tab.to_pylist()
    n=len(rows)
    if n < 32: continue
    ids=np.unique(np.linspace(0,n-32,min(10,n-31),dtype=int))
    all_actions=np.asarray([r['action'] for r in rows],dtype=np.float32)
    for idx in ids:
        r=rows[int(idx)]
        main=np.asarray(Image.open(io.BytesIO(r['observation.image'])).convert('RGB'))
        wrist=np.asarray(Image.open(io.BytesIO(r['observation.wrist_image'])).convert('RGB'))
        state=np.asarray(r['observation.joint_state']+r['observation.gripper_state'],dtype=np.float32)
        samples.append((p.stem,int(idx),state,main,wrist,all_actions[idx:idx+32].copy()))
    del tab, rows, all_actions
    print(f'loaded {fi+1}/{len(files)}: {p.stem}, n={n}, total samples={len(samples)}',flush=True)
print(f'loaded {len(samples)} samples in {time.time()-start:.1f}s; initializing model',flush=True)
model=load_stage1_model(recipe,ckpt,'cuda:0',norm_stats_path=stats,num_flow_steps=10,use_cuda_graph=False,rtc_guidance_mode='approx',action_representation='joint_absolute')
policy=OpenPIRLInfStage1Adapter(model,20,use_rtc=False,rtc_guidance_mode='approx')
preds=[]
bs=4
for begin in range(0,len(samples),bs):
    batch=samples[begin:begin+bs]
    obs={
      'states':np.stack([x[2] for x in batch]),
      'main_images':np.stack([x[3] for x in batch]),
      'wrist_images':np.stack([x[4] for x in batch]),
      'task_descriptions':[prompt]*len(batch),
    }
    with torch.inference_mode():
       out,_=policy.predict_action_batch(obs)
    preds.extend(out.detach().float().cpu().numpy())
    if (begin//bs)%10==0:
       print(f'inferred {min(begin+bs,len(samples))}/{len(samples)}',flush=True)

torch.cuda.synchronize()
pred=np.asarray(preds,dtype=np.float64)
truth=np.stack([x[5] for x in samples]).astype(np.float64)
err=pred-truth
names=['q1','q2','q3','q4','q5','q6','q7','gripper']
# Per-horizon metrics summarize action-chunk fit; aggregate all samples and dimensions.
print(f'INFERRED samples={len(samples)}, pred_shape={pred.shape}, elapsed_s={time.time()-start:.2f}')
print('ALL_CHUNK overall MAE:',np.mean(np.abs(err),axis=(0,1)))
print('ALL_CHUNK overall RMSE:',np.sqrt(np.mean(err**2,axis=(0,1))))
print('FIRST_ACTION MAE:',np.mean(np.abs(err[:,0,:]),axis=0))
print('FIRST_ACTION RMSE:',np.sqrt(np.mean(err[:,0,:]**2,axis=0)))
print('HORIZON MAE:',[(h,round(float(np.mean(np.abs(err[:,h,:7]))*180/np.pi),3),round(float(np.mean(np.abs(err[:,h,7]))*1000),3)) for h in [0,1,3,7,15,23,31]])
print('PER_DIM:',json.dumps({names[i]:{'mae':float(np.mean(np.abs(err[:,:,i]))),'rmse':float(np.sqrt(np.mean(err[:,:,i]**2))),'p95_abs':float(np.quantile(np.abs(err[:,:,i]),.95))} for i in range(8)},indent=2))
# Metrics in physical units and complete per-episode variation.
q_abs=np.abs(err[:,:,:7])*180/np.pi
g_abs=np.abs(err[:,:,7])*1000
print('JOINTS_ALL_CHUNK_deg: mae=%.4f rmse=%.4f p50=%.4f p95=%.4f p99=%.4f max=%.4f' % (q_abs.mean(),np.sqrt((err[:,:,:7]**2).mean())*180/np.pi,np.quantile(q_abs,.5),np.quantile(q_abs,.95),np.quantile(q_abs,.99),q_abs.max()))
print('GRIPPER_ALL_CHUNK_mm: mae=%.4f rmse=%.4f p50=%.4f p95=%.4f p99=%.4f max=%.4f' % (g_abs.mean(),np.sqrt((err[:,:,7]**2).mean())*1000,np.quantile(g_abs,.5),np.quantile(g_abs,.95),np.quantile(g_abs,.99),g_abs.max()))
# Exclude future padded rows none: all chunks length32.
per_ep={}
for ix,x in enumerate(samples): per_ep.setdefault(x[0],[]).append(ix)
rank=[]
for ep,idx in per_ep.items(): rank.append((float(np.mean(q_abs[idx])),ep,len(idx),float(np.mean(g_abs[idx]))))
print('EPISODE_JOINT_MAE_deg worst->best:')
for row in sorted(rank,reverse=True)[:8]: print(row)
np.savez_compressed('/tmp/flexiv_offline_compare.npz',pred=pred,truth=truth,episode=np.array([x[0] for x in samples]),frame=np.array([x[1] for x in samples]))
policy.close()
