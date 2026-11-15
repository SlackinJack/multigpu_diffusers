# Some *experimental* and *vibe-coded* plans
# These are here because Diffusers doesn't support plans for every transformer
# They're only applied if one isn't found in the transformer itself
# better than nothing


CUSTOM_CP_PLANS = {}
CUSTOM_TP_PLANS = {}


def get_custom_cp_plans(pipeline_name):
    return CUSTOM_CP_PLANS.get(pipeline_name)


def get_custom_tp_plans(pipeline_name):
    return CUSTOM_TP_PLANS.get(pipeline_name)


CUSTOM_TP_PLANS["Krea2Pipeline"] = {
    "text_fusion.layerwise_blocks.*.ff.gate": "colwise",
    "text_fusion.layerwise_blocks.*.ff.up": "colwise",
    "text_fusion.layerwise_blocks.*.ff.down": "rowwise",
    "text_fusion.refiner_blocks.*.ff.gate": "colwise",
    "text_fusion.refiner_blocks.*.ff.up": "colwise",
    "text_fusion.refiner_blocks.*.ff.down": "rowwise",
    "transformer_blocks.*.ff.gate": "colwise",
    "transformer_blocks.*.ff.up": "colwise",
    "transformer_blocks.*.ff.down": "rowwise",
}


CUSTOM_TP_PLANS["ZImagePipeline"] = {
    "noise_refiner.*.feed_forward.w1": "colwise",
    "noise_refiner.*.feed_forward.w3": "colwise",
    "noise_refiner.*.feed_forward.w2": "rowwise",
    "context_refiner.*.feed_forward.w1": "colwise",
    "context_refiner.*.feed_forward.w3": "colwise",
    "context_refiner.*.feed_forward.w2": "rowwise",
    "layers.*.feed_forward.w1": "colwise",
    "layers.*.feed_forward.w3": "colwise",
    "layers.*.feed_forward.w2": "rowwise",
}
