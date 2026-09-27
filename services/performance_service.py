"""Serve reproducible performance evidence, never unchecked slide constants."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'output/performance_evidence.json'
REFRESH=ROOT/'output/live_data_catchup_20261022.json'


def performance_evidence():
    if not REPORT.exists():
        return {'status':'unavailable','message':'The current-model evidence report has not been rebuilt yet. No presentation score is substituted.'}
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    mismatches=[]
    for name,digest in report.get('input_sha256',{}).items():
        path=(ROOT/name).resolve()
        path.relative_to(ROOT.resolve())
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            # The old R0 replay remains auditable after the authorised R1 release.
            # Its score is retained as historical evidence, never relabelled R1.
            archive=ROOT/'core/monitor_r0_frozen_20261026.py'
            lock_path=ROOT/'output/active_policy_lock_20261026.json'
            if name=='core/monitor_engine.py' and archive.exists() and lock_path.exists():
                lock=json.loads(lock_path.read_text())
                if (hashlib.sha256(archive.read_bytes()).hexdigest()==digest and path.exists()
                    and lock.get('active_monitor_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()):continue
            mismatches.append(name)
    if not report.get('input_sha256') or mismatches:
        return {'status':'outdated','message':'Model or evaluation inputs changed; rebuild the evidence before presenting these scores.',
                'changed_inputs':mismatches}
    active_path=ROOT/'output/FINAL_SYSTEM_LOCK_20261026.json'
    return {**report,'status':'verified_current_snapshot',
            'active_release':json.loads(active_path.read_text()) if active_path.exists() else None,
            'module2_scope':'Archived R0 policy; active R1 is reported separately under active_release.',
            'data_catchup':json.loads(REFRESH.read_text(encoding='utf-8')) if REFRESH.exists() else None}
