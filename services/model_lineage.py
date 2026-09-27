"""Expose the actual deployed ensemble construction, not a former proposal."""
def active_architecture():
    from services.prospective_module1 import bundles
    from ml.research_module1_forward_update import SETS
    from services.locked_model_configuration import VERSION
    protocol,model,_=bundles()
    experts=[]
    for key in model['experts']:
        config=model['expert_configs'][key]
        experts.append(dict(id=key,family=config['family'],feature_group=config['feature_set'],
            features=list(SETS.get(config['feature_set'],['gmp_pct'])),
            regularization_C=config.get('C'),max_depth=config.get('max_depth'),
            class_weight=config.get('class_weight'),transform=config.get('transform')))
    parameters=model['meta_model'].get_params()
    penalties=[v for k,v in parameters.items() if k=='C' or k.endswith('__C')]
    return dict(configuration=VERSION,artifact_sha256=protocol['model']['sha256'],
        training_ipos=len(model['training_ids']),training_end_year=model['training_end_year'],
        experts=experts,meta_model='Standardized expert log-odds + logistic regression',
        meta_C=penalties[0] if len(penalties)==1 else None,direction_cutoff=model['threshold'],calibration='none',
        evaluation='Retrospective development comparison; independent prospective results are tracked separately.')
