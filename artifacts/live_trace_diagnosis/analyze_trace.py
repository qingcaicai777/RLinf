import json
from pathlib import Path
import numpy as np
from PIL import Image
from rlinf.algorithms.online_residual_rl.inference_trace import iter_trace
p=Path('artifacts/live_inference_trace/session_95a6986a-cda0-4843-abfb-6e508c309b53')
out=Path('artifacts/live_trace_diagnosis');events=list(iter_trace(p))
infer=[e for e in events if e['request']['type']=='infer' and not e['request'].get('warmup')]
receipts=[e for e in events if e['request']['type']=='receipt']
lookup={e['request']['frame']: e for e in infer}
q=np.array([e['request']['info']['diagnostics']['measured_before_control'][:7] for e in receipts]);target=np.array([e['request']['info']['joint_target_rad'] for e in receipts]);tcp=np.array([e['request']['info']['diagnostics']['tcp_pose_wxyz'][:3] for e in receipts]);frames=np.array([e['request']['frame'] for e in receipts]);manual=np.array([e['request']['info']['diagnostics']['manual'] for e in receipts]);times=np.array([e['request']['info']['diagnostics']['sampled_perf_counter_ns'] for e in receipts])/1e9
# Measurement occurs BEFORE the current tick's command.
track=np.rad2deg(abs(q[1:]-target[:-1])).max(1)
alignment=[]
for e in receipts:
 r=e['request'];info=r['info'];a=info['execution_alignment'];source=lookup[a['source_request_frame']];idx=a['prediction_index']
 if idx is not None and info['policy_proposal_valid']:
  full=source['response']['diagnostics']['full_actions_physical'][0]
  alignment.append(np.max(abs(full[idx]-np.array(info['policy_proposal_physical']))))
ages={name:np.array([e['request']['action_context']['observation_timing']['camera_age_ms'][name] for e in infer]) for name in ['main','wrist']}
def stats(a):return {k:float(v) for k,v in zip(['min','median','p95','max'],[np.min(a),np.median(a),np.quantile(a,.95),np.max(a)])}
result={'session':p.name,'requests':len(infer),'control_ticks':len(receipts),'manual_first_tick':int(frames[np.flatnonzero(manual)[0]]) if manual.any() else None,'camera_age_ms':{k:stats(v) for k,v in ages.items()},'model_ms':stats(np.array([e['response']['model_latency_ms'] for e in infer if not e['request'].get('bootstrap')])), 'transport_ms':stats(np.array([(e['received_monotonic']-e['sent_monotonic'])*1000 for e in infer if not e['request'].get('bootstrap')])), 'control_intervals_ms':stats(np.diff(times)*1000),'tracking_previous_target_max_joint_deg':stats(track),'max_model_to_proposal_abs_difference':max(alignment),'invalid_schedule_ticks':[int(e['request']['frame']) for e in receipts if not e['request']['info']['schedule_valid']], 'samples':[]}
for frame in [0,16,20,32,36,48,64,96,128,160,192,256,320,400,500,632]:
 if frame>=len(q):continue
 result['samples'].append({'tick':frame,'measured_q6':float(q[frame,5]),'target_q6':float(target[frame,5]),'tcp_delta_mm':((tcp[frame]-tcp[0])*1000).tolist(),'max_prev_tracking_deg':float(track[frame-1]) if frame else None})
for e in [infer[0],infer[1],infer[2],infer[4],infer[-1]]:
 frame=e['request']['frame']
 for name,key in [('main','main_images'),('wrist','wrist_images')]:Image.fromarray(e['request']['observation'][key]).save(out/f'frame_{frame:06d}_{name}.png')
(out/'trace_summary.json').write_text(json.dumps(result,indent=2));np.savez(out/'trace_arrays.npz',q=q,target=target,tcp=tcp,frames=frames,times=times,manual=manual)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(9,1,figsize=(14,21),sharex=True)
for j in range(7):
 axes[j].plot(frames,np.rad2deg(q[:,j]),label='measured before control');axes[j].plot(frames,np.rad2deg(target[:,j]),label='command',alpha=.8);axes[j].set_ylabel(f'J{j+1} deg');axes[j].grid(alpha=.2)
axes[0].legend()
for j,name in enumerate('XYZ'):axes[7].plot(frames,(tcp[:,j]-tcp[0,j])*1000,label=name)
axes[7].set_ylabel('Measured TCP delta mm');axes[7].legend();axes[7].grid(alpha=.2)
for name,v in ages.items():axes[8].plot([e['request']['frame'] for e in infer],v,label=name)
axes[8].set_ylabel('Camera age ms');axes[8].legend();axes[8].set_xlabel('Robot tick');axes[8].grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'trace_execution.png',dpi=130)
print(json.dumps(result,indent=2))
