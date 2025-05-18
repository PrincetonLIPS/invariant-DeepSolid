def add_sampling_opts(cfg):
    """
        Add sampling parameters to cfg
    """
    cfg.batch_size = 20
    
    cfg.symmetria.canon.on = True
    cfg.symmetria.canon.rand = False

    cfg.mcmc.steps = 30000

    return cfg
