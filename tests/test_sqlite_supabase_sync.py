import base64

from services.sqlite_supabase_sync import decode_row, encode_row


def test_sqlite_row_round_trip_preserves_blob_and_composite_key():
    row = {"ipo_id": "id-1", "date": "2026-10-26", "raw": b"html\x00payload", "value": 1.25}
    key, payload_json, digest = encode_row(row, ["ipo_id", "date"])

    import json
    assert json.loads(key) == {"date": "2026-10-26", "ipo_id": "id-1"}
    payload = json.loads(payload_json)
    assert payload["raw"]["base64"] == base64.b64encode(row["raw"]).decode("ascii")
    assert decode_row(payload) == row
    assert len(digest) == 64


def test_sync_row_hash_does_not_depend_on_column_order():
    a = {"id": 7, "symbol": "XYZ", "status": "ok"}
    b = {"status": "ok", "symbol": "XYZ", "id": 7}
    assert encode_row(a, ["id"])[1:] == encode_row(b, ["id"])[1:]
