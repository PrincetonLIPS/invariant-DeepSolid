def add_optim_opts(cfg):
    """
        Add optimisation parameters to cfg
    """
    cfg.batch_size = 500
    cfg.pretrain.iterations = 1000
    cfg.symmetria.gpave.on = False
    cfg.symmetria.gpave.over_phase = False
    cfg.symmetria.gpave.before_det = False
    
    cfg.symmetria.augment.on = True
    cfg.symmetria.augment.subsample.on = True 
    cfg.symmetria.augment.subsample.replace = False
    cfg.symmetria.augment.subsample.num = 2
    
    cfg.symmetria.schedule = lambda t: 'OG'
    
    cfg.log.save_frequency_opt = 'iter'
    cfg.log.save_frequency = 1000

    cfg.optim.iterations = 80000

    return cfg