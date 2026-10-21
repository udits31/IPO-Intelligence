from services import model_lineage


def test_lineage_reports_actual_active_artifact_configuration(monkeypatch):
    from services import prospective_module1 as prospective
    class Meta:
        def get_params(self):return {'logistic__C':.5}
    model=dict(experts=['demand','trajectory'],expert_configs={
        'demand':dict(family='logistic',feature_set='gmp_only',C=.5,transform='raw'),
        'trajectory':dict(family='random_forest',feature_set='trajectory',max_depth=5)},
        training_ids=[1,2,3],training_end_year=2025,threshold=.5,meta_model=Meta())
    monkeypatch.setattr(prospective,'bundles',lambda:({'model':{'sha256':'actual-frozen-hash'}},model,None))
    r=model_lineage.active_architecture()
    assert r['artifact_sha256']=='actual-frozen-hash' and r['training_ipos']==3
    assert r['experts'][0]['features']==['gmp_pct']
    assert r['experts'][1]['family']=='random_forest' and r['experts'][1]['max_depth']==5
    assert r['meta_C']==.5 and r['direction_cutoff']==.5
