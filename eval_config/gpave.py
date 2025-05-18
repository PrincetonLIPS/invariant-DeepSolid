def add_eval_opts(cfg):
    """
        Add eval parameters to cfg
    """
    cfg.symmetria.gpave.on = True
    cfg.symmetria.gpave.subsample.on = False
    return cfg