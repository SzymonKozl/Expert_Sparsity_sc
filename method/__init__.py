from . import expert_pruning
from . import dynamic_skipping


METHODS = {
    'layerwise_pruning': expert_pruning.get_layerwise_pruning("PrunableMixtralSparseMoeBlockWrapper"),
    'layerwise_pruning_mes': expert_pruning.get_layerwise_pruning("MesWrapper"),
    'progressive_pruning': expert_pruning.progressive_pruning,
    'dynamic_skipping': dynamic_skipping.dynamic_skipping,
}
