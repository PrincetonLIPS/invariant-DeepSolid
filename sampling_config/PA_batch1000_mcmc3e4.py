def add_sampling_opts(cfg):
    """
        Add sampling parameters to cfg
    """
    cfg.batch_size = 1000
    
    cfg.symmetria.augment.on = False 
    cfg.symmetria.gpave.on = True
    cfg.symmetria.gpave.subsample.on = False
    cfg.symmetria.measure.on = True # measures symmetry
    cfg.mcmc.steps = 30000

    return cfg