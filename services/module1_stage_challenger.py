"""Explicit stage routing for the research challenger; never a live promotion."""
import hashlib
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from ml.train_stage_fundamental_challenger import ARTIFACT,OUT,EXTRA
from ml.model_pipeline import FEATURE_COLS
from services.dated_ipo_fundamentals import load_registry,at_decision


def score_stage_challenger(*,ipo_id,stage,decision_at,source_at,qib,retail,hni,
                           offer_cap_price=None,issue_size_cr=None):
    if stage not in ('penultimate_session','final_book'):
        raise ValueError('No trained model for this bid stage; do not substitute final-book coefficients')
    decision=pd.Timestamp(decision_at);source=pd.Timestamp(source_at)
    if decision.tzinfo is None or source.tzinfo is None or pd.isna(decision) or pd.isna(source) or source>decision:
        raise ValueError('Require dated observations available before the decision')
    report=json.loads(OUT.read_text(encoding='utf-8'))
    if hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()!=report['artifact_sha256']:
        raise ValueError('Stage artifact hash mismatch')
    artifact=joblib.load(ARTIFACT)
    track=artifact['tracks'][stage]
    if decision.date().isoformat()<=track['latest_training_outcome']:
        raise ValueError('A challenger cannot predict before its training outcomes were available')
    values=np.asarray([qib,retail,hni],dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Complete observed bids are required')
    facts=at_decision(load_registry().get(int(ipo_id)),decision_at,offer_cap_price)
    row=dict(zip(FEATURE_COLS,np.log1p(np.maximum(values,.01))))
    row.update({key:facts[key] for key in EXTRA[2:]})
    for key,value in [('log_offer_cap_price',offer_cap_price),('log_issue_size',issue_size_cr)]:
        row[key]=np.log1p(float(value)) if value is not None and np.isfinite(float(value)) and float(value)>0 else None
    probability=float(track['model'].predict_proba(pd.DataFrame([row])[track['features']])[0,1])
    return dict(model_version='module1-stage-dated-fundamentals-challenger-all-20261024',
        research_only=True,validated_for_live=False,training_stage=stage,
        positive_listing_probability=probability,financial_evidence_status=facts['evidence_status'],
        financial_observed=facts['financial_observed'],offer_structure_observed=facts['offer_observed'],
        source_asof=str(source),decision_at=str(decision),training_n=track['training_n'])
