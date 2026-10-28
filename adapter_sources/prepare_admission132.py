from common132 import *
from verify132 import CONFIG
from admission132 import approve

root=H/'admission_r11'
root.mkdir(exist_ok=False)
pre={}
for p in [H/'admission132.py',Path(__file__),H/'held_global132_r10_report.json',H/'joint_r10/pre_source_hashes.json',H/'joint_r10/output_hashes.json']:
    pre[str(p)]=sha(p)
verify(load(H/'joint_r10/output_hashes.json'))
verify(load(H/'joint_r10/pre_source_hashes.json'))
held=load(H/'held_global132_r10_report.json')
if set(held['scenes']) != set(SCENES) or held['frozen_thresholds'] != CONFIG:
    raise ValueError('Incomplete original final validation source')
save(root/'pre_execution_hashes.json',pre)
summary={}
for scene in SCENES:
    path=H/'joint_r10'/scene/'native/feature_bindings.json'
    pre[str(path)]=sha(path)
    policy=approve(load(path),held['scenes'][scene]['pairs'],CONFIG)
    folder=root/scene
    folder.mkdir()
    save(folder/'policy.json',policy)
    summary[scene]=dict(approved=len(policy['approved']),rejected=len(policy['rejected']),
                        rejected_records=[dict(i=p['i'],j=p['j'],reason=p['reason'],evidence=p['evidence']) for p in policy['rejected']])
    print('FINAL LOOP ADMISSION',scene,summary[scene],flush=True)
verify(pre)
save(root/'pre_execution_hashes.json',pre)
save(root/'report.json',dict(scenes=summary,independent_validation_claimed=False,frame_coverage_unchanged=True))
save(root/'output_hashes.json',{str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
