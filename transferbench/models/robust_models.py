from robustbench import load_model
from torch import nn


def get_robustbench_model(name: str, dataset: str, threat_model: str = 'Linf') -> nn.Module:
    """
    @TODO: preprocessing in these models has been already done!
    """
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model

def load_robust_mimir() -> nn.Module:
    """ adversarial training defense """
    return get_robustbench_model(name='Xu2024MIMIR_Swin-L', dataset='imagenet')

def load_robust_meansparse() -> nn.Module:
    """ post-training robustness defense"""
    return get_robustbench_model(name='Amini2024MeanSparse_Swin-L', dataset='imagenet')