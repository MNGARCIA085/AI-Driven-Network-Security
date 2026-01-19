


def unflatten_config(flat_cfg):
    cfg = {"model": {}, "training": {}}
    for k, v in flat_cfg.items():
        group, name = k.split(".")
        cfg[group][name] = v
    return cfg
