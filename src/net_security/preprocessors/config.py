from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class DataConfig:
    path: str
    features: Optional[List[str]]
    batch_size: int 
    balance_factor: float
    val_size: float
    random_state: int
    scaler_type: str # 'standard', 'minmax', 'robust', 'none'



def build_data_config(data_cfg: dict, global_cfg: dict) -> DataConfig:
    return DataConfig(
        path = data_cfg['path'],
        features = data_cfg['features'],
        balance_factor = data_cfg['balance_factor'],
        val_size = data_cfg['val_size'],
        scaler_type = data_cfg['scaler_type'],
        batch_size = global_cfg['batch_size'],
        random_state = global_cfg['random_state']
    )



"""
To work with child configs; in this case they are not really different

@dataclass(frozen=True)
class BaseDataConfig:
    path: str
    path_test: str
    path_inference: str
    features: Optional[list[str]]
    val_size: float
    random_state: int


@dataclass(frozen=True)
class NNDataConfig(BaseDataConfig):
    batch_size: int
    balance_factor: float
    scaler_type: str

@dataclass(frozen=True)
class TreeDataConfig(BaseDataConfig):
    class_weight: Optional[str]  # "balanced", None
    categorical_encoding: str    # "onehot", "target"


---
class BaseDataModule:
    def __init__(self, cfg: BaseDataConfig):
        self.cfg = cfg

class NNDataModule(BaseDataModule):
    def __init__(self, cfg: NNDataConfig):
        super().__init__(cfg)
        self.batch_size = cfg.batch_size


class TreeDataModule(BaseDataModule):
    def __init__(self, cfg: TreeDataConfig):
        super().__init__(cfg)

-----
def build_data_module(
    model_kind: str,
    base_cfg: BaseDataConfig,
    data_cfg_raw: dict,
) -> BaseDataModule:

    if model_kind == "nn":
        cfg = NNDataConfig(**base_cfg.__dict__, **data_cfg_raw["nn"])
        return NNDataModule(cfg)

    if model_kind == "tree":
        cfg = TreeDataConfig(**base_cfg.__dict__, **data_cfg_raw["tree"])
        return TreeDataModule(cfg)

    raise ValueError(f"Unknown model_kind: {model_kind}")


OR

REGISTRY = {
    "nn": (NNDataConfig, NNDataModule),
    "tree": (TreeDataConfig, TreeDataModule),
}

def build_data_module(model_kind, base_cfg, data_cfg_raw):
    cfg_cls, module_cls = REGISTRY[model_kind]
    cfg = cfg_cls(**base_cfg.__dict__, **data_cfg_raw[model_kind])
    return module_cls(cfg)


"""