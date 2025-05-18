def add_optim_opts(cfg):
    """
        Add optimisation parameters to cfg
    """
    cfg.batch_size = 4000
    cfg.pretrain.iterations = 1000
    cfg.symmetria.gpave.on = False
    cfg.symmetria.gpave.over_phase = True
    cfg.symmetria.gpave.before_det = False

    cfg.symmetria.schedule = lambda t: 'OG'
    
    cfg.log.save_frequency_opt = 'iter'
    cfg.log.save_frequency = 1000
  
    cfg.optim.iterations = 80000
    
    return cfg
