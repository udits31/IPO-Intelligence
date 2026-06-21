"""Install typed production views over the already-mirrored Supabase rows.

No copy or deletion of raw records. Local SQLite is inspected read-only only
by this one-time installer, not used by production API requests.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from db.ledger import DATABASES, _literal

ROOT=Path(__file__).resolve().parents[1]


def inspect_schema(directory=ROOT/'data/telemetry'):
    result={}
    for database in sorted(DATABASES):
        path=directory/database
        if not path.exists():continue
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
            tables=[]
            for table,sql in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                columns=connection.execute(f'PRAGMA table_info("{table}")').fetchall()
                pk=[r[1] for r in sorted(columns,key=lambda r:r[5]) if r[5]]
                if not pk:raise ValueError('Ledger table has no stable identity: '+database+'/'+table)
                unique=[pk]
                for index in connection.execute(f'PRAGMA index_list("{table}")').fetchall():
                    if index[2]:
                        names=[r[2] for r in connection.execute(f'PRAGMA index_info("{index[1]}")')]
                        if names and names not in unique:unique.append(names)
                triggers=[r[0] for r in connection.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?",(table,))]
                tables.append(dict(table=table,columns=[dict(name=r[1],type=r[2],notnull=bool(r[3]),default=r[4],pk=r[5]) for r in columns],
                    pk=pk,unique=unique,immutable=table not in ('data_maintenance_jobs_v1','offer_source_state_v1')))
            result[database]=tables
    return result


def statements(database,tables):
    schema='openbell_'+database.removesuffix('.db')
    result=[]
    for item in tables:
        table=item['table'];projection=[];binary=[];defaults=[]
        for column in item['columns']:
            name=column['name'];kind=column['type'].upper();reference=f'payload->{_literal(name)}'
            text=f'row_key::jsonb->>{_literal(name)}' if column['pk'] else f'payload->>{_literal(name)}'
            if 'BLOB' in kind:
                binary.append(name)
                value=f"decode(({reference})->>'base64','base64')"
            elif 'INT' in kind:value=f'({text})::bigint'
            elif any(k in kind for k in ('REAL','DOUBLE','FLOAT','NUMERIC')):value=f'({text})::double precision'
            else:value=text
            projection.append(f'{value} AS "{name}"')
            if name=='id' and 'INT' in kind and column['pk']:
                sequence=f'{schema}.{table}_id_seq'
                result.append(f'CREATE SEQUENCE IF NOT EXISTS {sequence} AS bigint')
                result.append(f"SELECT setval({_literal(sequence)},GREATEST((SELECT COALESCE(MAX((row_key::jsonb->>'id')::bigint),0)+1 FROM public.capstone_sqlite_mirror WHERE db_name={_literal(database)} AND table_name={_literal(table)}),(SELECT last_value FROM {sequence})),false)")
                defaults.append(f'ALTER VIEW "{table}" ALTER COLUMN "id" SET DEFAULT nextval({_literal(sequence)})')
            elif column['default'] is not None:
                defaults.append(f'ALTER VIEW "{table}" ALTER COLUMN "{name}" SET DEFAULT {column["default"]}')
        result.append(f'CREATE OR REPLACE VIEW "{table}" AS SELECT '+','.join(projection)+f' FROM public.capstone_sqlite_mirror WHERE db_name={_literal(database)} AND table_name={_literal(table)}')
        result.extend(defaults)
        spec=dict(database=database,table=table,pk=item['pk'],unique=item['unique'],binary=binary,immutable=item['immutable'],
            notnull=[c['name'] for c in item['columns'] if c['notnull'] or c['pk']])
        result.append(f'CREATE OR REPLACE TRIGGER openbell_store_row INSTEAD OF INSERT OR UPDATE OR DELETE ON "{table}" FOR EACH ROW EXECUTE FUNCTION public.openbell_write_ledger_view_v1({_literal(json.dumps(spec,separators=(",",":")))})')
    return result


def install(client=None):
    from db.supabase_client import get_supabase
    client=client or get_supabase()
    manifest=inspect_schema()
    if not manifest:manifest=json.loads((ROOT/'db/native_ledger_schema.json').read_text(encoding='utf8'))
    counts={}
    for database,tables in manifest.items():
        for table in tables:
            response=client.rpc('openbell_ledger_transaction_v1',dict(p_database=database,p_statements=statements(database,[table]),p_preview=False)).execute()
            if not response.data or response.data.get('preview'):raise RuntimeError('Native view installation did not commit')
        counts[database]=len(tables)
    target=ROOT/'db/native_ledger_schema.json'
    target.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    if args.apply:print(json.dumps(install()))
    else:
        manifest=inspect_schema()
        (ROOT/'db/native_ledger_schema.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        print(json.dumps({db:len(t) for db,t in manifest.items()}))
