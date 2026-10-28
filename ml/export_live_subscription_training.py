"""Export deduplicated dated training inputs without changing raw captures."""
import argparse,csv,json,sqlite3,hashlib
from pathlib import Path
from services.subscription_training import training_captures,install_view,DB,ROOT


def run(days,*,db_path=DB,output=None):
    output=Path(output or ROOT/'output/live_subscription_training_20261029');output.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db_path) as c:
        before=c.execute('SELECT COUNT(*) FROM subscription_capture_v2').fetchone()[0];install_view(c)
    report={}
    for day in days:
        data=training_captures(day,db_path=db_path);records=data.pop('records')
        path=output/f'{day}.csv'
        if records:
            with path.open('w',encoding='utf8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        data.update(csv=str(path),csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                    removed_duplicate_verified_rows=data['verified_raw_rows']-data['training_rows']-len(data['rejected']))
        report[day]=data
    with sqlite3.connect(db_path) as c:after=c.execute('SELECT COUNT(*) FROM subscription_capture_v2').fetchone()[0]
    if before!=after:raise RuntimeError('Raw capture history changed during export; rerun for a stable count audit')
    result=dict(days=report,raw_rows_before=before,raw_rows_after=after,raw_history_preserved=True)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('days',nargs='+');a=p.parse_args();run(a.days)
