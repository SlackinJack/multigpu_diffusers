import modelopt.torch.quantization as mtq

import diffusers
from diffusers import AutoModel as AutoModelD
from diffusers import BitsAndBytesConfig as BitsAndBytesConfigD
from diffusers import NVIDIAModelOptConfig as NVIDIAModelOptConfigD
from diffusers import QuantoConfig as QuantoConfigD
from diffusers import GGUFQuantizationConfig as GGUFQuantizationConfigD
from diffusers.hooks import apply_group_offloading
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_lokr_to_diffusers,
)
from diffusers.utils import load_image
import transformers
from transformers import AutoModel as AutoModelT
from transformers import BitsAndBytesConfig as BitsAndBytesConfigT
from transformers import QuantoConfig as QuantoConfigT

from modules.utils import (
    clean_override_function,
    override_function,
    get_function_from_class,
    check_function_kwargs,
    get_torch_type,
    clean,
    add_alpha_to_latent,
    decode_b64_and_unpickle,
    format_json,
    format_dict,
    json_path_to_dict,
)

import copy
import json
import logging
import safetensors
import time
import torch
import torch.nn.functional as F
import traceback
from lycoris import create_lycoris_from_weights
from pathlib import Path
from safetensors.torch import load_file
from sdnq import SDNQConfig
import cache_dit
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig
from compel import Compel, ReturnedEmbeddingsType

COMPEL_SUPPORTED_MODELS = ["sd1", "sd2", "sdxl"]


GENERIC_HOST_ARGS = {
    "port": int,
}


GENERIC_HOST_ARGS_TOGGLES = []


# these pipelines do not natively support start_step
# (uses a custom workaround in callback_on_step_end):
START_STEP_WORKAROUND_PIPELINES = ["ZImagePipeline"]

# these pipelines do not support using torch deterministic algorithms:
NON_DETERMINISTIC_PIPELINES = ["ZImagePipeline"]

# non-comprehensive
# these pipelines will only output to PIL instead of a latent:
# TODO: to be fair, i could just torch.stack the latents
PIL_OUTPUT_PIPELINES = [
    "StableDiffusionUpscalePipeline",
    "WanPipeline",
    "WanImageToVideoPipeline",
]


TRANSFORMER_MODULE_NAMES = ["transformer", "transformer_2", "unet"]
ENCODER_MODULE_NAMES = [
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "image_encoder",
]
MISC_MODULE_NAMES = [
    "controlnet",
    "motion_adapter",
    "image_processor",
    "feature_extractor",
]
VAE_MODULE_NAMES = ["vae"]
TOKENIZER_MODULE_NAMES = ["tokenizer", "tokenizer_2"]


def get_quant_mapping(target, quantize_to, as_dict=False):
    out = {}
    config = None

    quantize_config = copy.deepcopy(quantize_to)
    backend = quantize_config.pop("backend")
    match backend:
        case "bitsandbytes":
            c = {}
            load_in_4bit, load_in_8bit = (
                quantize_config.get("load_in_4bit"),
                quantize_config.get("load_in_8bit"),
            )
            assert not (load_in_8bit is True and load_in_4bit is True), (
                "Select either 4-bit or 8-bit quantization, but not both"
            )
            if load_in_8bit is True:
                for k, v in quantize_config.items():
                    if "4bit" in k:
                        continue
                    c[k] = v
            elif load_in_4bit is True:
                for k, v in quantize_config.items():
                    if "int8" in k:
                        continue
                    if k in [
                        "bnb_4bit_compute_dtype",
                        "bnb_4bit_quant_storage",
                    ]:
                        v = get_torch_type(v)
                    c[k] = v
            else:
                return out
            if as_dict:
                config = [{"bitsandbytes": c}] * 2
            else:
                config = [BitsAndBytesConfigD(**c), BitsAndBytesConfigT(**c)]

        case "quanto":
            quant_type = quantize_config["quant_type"]
            if as_dict:
                config = [{"quanto": {"weights_dtype": quant_type}}] * 2
            else:
                config = [
                    QuantoConfigD(weights_dtype=quant_type),
                    QuantoConfigT(weights=quant_type),
                ]

        case "sdnq":
            # https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
            kwargs = {}
            for k, v in quantize_config.items():
                kwargs[k] = v
            if as_dict:
                config = [{"sdnq": kwargs}] * 2
            else:
                config = [SDNQConfig(**kwargs)] * 2

        case "modelopt":
            # only for Diffusers, Transformers doesnt exist yet
            # TODO: fix
            kwargs = {}
            for k, v in quantize_config.items():
                kwargs[k] = v
            kwargs["modelopt_config"] = {
                "quant_cfg": {
                    "*weight_quantizer": {"fake_quant": False},
                    "*input_quantizer": {},
                    "*output_quantizer": {"enable": False},
                    "*q_bmm_quantizer": {},
                    "*k_bmm_quantizer": {},
                    "*v_bmm_quantizer": {},
                    "*softmax_quantizer": {},
                },
            }
            for item in mtq.config._default_disabled_quantizer_cfg:
                kwargs["modelopt_config"]["quant_cfg"].update(item)
            if as_dict:
                config = [{"modelopt": kwargs}, None]
            else:
                config = [NVIDIAModelOptConfigD(**kwargs), None]

    if config is not None:
        match target:
            case "transformer":
                if config[0] is not None:
                    for name in TRANSFORMER_MODULE_NAMES:
                        out[name] = config[0]
            case "vae":
                if config[0] is not None:
                    for name in VAE_MODULE_NAMES:
                        out[name] = config[0]
            case "misc":
                if config[0] is not None:
                    for name in MISC_MODULE_NAMES:
                        out[name] = config[0]
            case "encoder":
                if config[1] is not None:
                    for name in ENCODER_MODULE_NAMES:
                        out[name] = config[1]
            case "tokenizer":
                if config[1] is not None:
                    for name in TOKENIZER_MODULE_NAMES:
                        out[name] = config[1]
    return out


def get_quantization_config(quantization_config, as_dict):
    mappings = {}
    quantize_transformer = quantization_config.get("transformer")
    quantize_encoder = quantization_config.get("encoder")
    quantize_vae = quantization_config.get("vae")
    quantize_tokenizer = quantization_config.get("tokenizer")
    quantize_misc = quantization_config.get("misc")
    if quantize_transformer is not None:
        mappings.update(
            get_quant_mapping(
                "transformer", quantize_transformer, as_dict=as_dict
            )
        )
    if quantize_encoder is not None:
        mappings.update(
            get_quant_mapping("encoder", quantize_encoder, as_dict=as_dict)
        )
    if quantize_vae is not None:
        mappings.update(
            get_quant_mapping("vae", quantize_vae, as_dict=as_dict)
        )
    if quantize_tokenizer is not None:
        mappings.update(
            get_quant_mapping("tokenizer", quantize_tokenizer, as_dict=as_dict)
        )
    if quantize_misc is not None:
        mappings.update(
            get_quant_mapping("misc", quantize_misc, as_dict=as_dict)
        )
    if len(list(mappings.keys())) > 0:
        if as_dict:
            return mappings
        else:
            return diffusers.PipelineQuantizationConfig(quant_mapping=mappings)


def get_cache_dit_config(cache_dit_config):
    cd_config = {}
    cd_config["cache_config"] = DBCacheConfig(
        max_warmup_steps=cache_dit_config.get("max_warmup_steps"),
        max_cached_steps=cache_dit_config.get("max_cached_steps"),
        Fn_compute_blocks=cache_dit_config.get("Fn_compute_blocks"),
        Bn_compute_blocks=cache_dit_config.get("Bn_compute_blocks"),
        residual_diff_threshold=cache_dit_config.get(
            "residual_diff_threshold"
        ),
        enable_separate_cfg=cache_dit_config.get("enable_separate_cfg"),
    )
    if "enable_taylorseer" in cache_dit_config:
        cd_config["calibrator_config"] = TaylorSeerCalibratorConfig(
            cache_dit_config.get("taylorseer_order"),
        )
    if "enable_scm" in cache_dit_config:
        cd_config["steps_computation_mask"] = cache_dit.steps_mask(
            mask_policy=cache_dit_config.get("scm_mask_policy"),
            total_steps=cache_dit_config.get("scm_total_steps"),
        )
        if "steps_computation_policy" in cache_dit_config:
            cd_config["steps_computation_policy"] = cache_dit_config.get(
                "steps_computation_policy"
            )
    return cd_config


def get_deepcache_config(deepcache_config):
    dc_config = {}
    dc_config["cache_interval"] = deepcache_config.get("deep_cache_interval")
    dc_config["cache_branch_id"] = deepcache_config.get("deep_cache_id")
    return dc_config


class CommonHost:
    def __init__(self):
        self.local_rank = -1
        self.vae_dtype = None
        self.torch_dtype = torch.float32
        self.initialized = False
        self.progress = 0
        self.pipe = None
        self.pipeline_name = None
        self.backend_name = None
        self.logger = None
        self.log_level = 2
        self.default_scheduler = None
        self.adapter_names = None
        self.loras_are_fused = False
        self.applied = None
        self.is_image_model = True
        self.is_transformer_model_type = False
        self.can_use_deepcache = False
        self.can_use_cachedit = False
        self.offloaded_modules = []
        self.device_map_modules = []
        self.loaded_model_index = None

    def get_initialized_flask(self):
        if self.initialized:
            return "", 200
        else:
            return "", 202

    def get_progress_flask(self):
        return str(self.progress), 200

    def set_logger(self):
        if self.logger is None:
            logger = logging.getLogger(name=str(self.local_rank))
            logger.setLevel(logging.INFO)
            if logger.hasHandlers():
                logger.handlers.clear()
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt=f"[Rank {str(self.local_rank)}]: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
            self.logger = logger
        return

    def log(self, level, text, rank_0_only=True):
        if rank_0_only is True and self.local_rank > 0:
            return
        if self.log_level >= level:
            text = str(text)
            self.logger.info(text)
        return

    def get_applied(self):
        return str(self.applied), 200

    def close_pipeline(self):
        # TODO: fix this method to properly free up resources
        if self.pipe is not None:
            try:
                self.pipe.reset_device_map()
            except Exception:
                pass
            for k, v in self.pipe.components.items():
                setattr(self.pipe, k, None)
            self.pipe = self.pipe.to("cpu")

        # self.local_rank = -1
        self.vae_dtype = None
        self.torch_dtype = torch.float32
        self.initialized = False
        self.progress = 0
        self.pipe = None
        self.pipeline_name = None
        self.backend_name = None
        # self.logger = None
        self.log_level = 2
        self.default_scheduler = None
        self.adapter_names = None
        self.loras_are_fused = False
        self.applied = None
        self.is_image_model = True
        self.is_transformer_model_type = False
        self.can_use_deepcache = False
        self.can_use_cachedit = False
        self.offloaded_modules = []
        self.device_map_modules = []
        self.loaded_model_index = None

        clean()
        time.sleep(5)
        return

    def print_mem_usage(self):
        mem_usage_string = "🖥️ Memory usage:"
        for k, v in self.pipe.components.items():
            try:
                mem = round(v.get_memory_footprint() / 1024 / 1024, 1)
            except Exception:
                mem = "?"
            mem_usage_string += f"\n        {k}: {mem} MB"
        self.log(2, f"{mem_usage_string}")
        return

    def print_params(self, data):
        if self.log_level >= 2:
            formatted = "📋 Received parameters: " + format_json(
                data, indent=8, indent_all=True
            )
            self.log(2, formatted)
        return

    def print_timesteps(self, timesteps):
        if self.log_level >= 1:
            t_string = str(timesteps)
            formatted = "👣 Timesteps:"
            for t in t_string.split("\n"):
                formatted += f"\n        {t}"
            self.log(1, formatted)
        return

    def print_sigmas(self, timesteps):
        if self.log_level >= 1:
            t_string = str(timesteps)
            formatted = "📊 Sigmas:"
            for t in t_string.split("\n"):
                formatted += f"\n        {t}"
            self.log(1, formatted)
        return

    def setup_pipeline(self, data, backend_name=None):
        self.print_params(data)

        # models
        data["checkpoint"] = data.setdefault("checkpoint")
        data["transformer"] = data.setdefault("transformer")
        data["vae"] = data.setdefault("vae")
        data["vae_fp16"] = data.setdefault("vae_fp16")
        data["control_net"] = data.setdefault("control_net")
        data["text_encoder"] = data.setdefault("text_encoder")
        data["text_encoder_2"] = data.setdefault("text_encoder_2")
        data["text_encoder_3"] = data.setdefault("text_encoder_3")
        data["lora"] = data.setdefault("lora")

        # memory & optimizations
        data["enable_vae_slicing"] = data.setdefault("enable_vae_slicing")
        data["enable_vae_tiling"] = data.setdefault("enable_vae_tiling")
        data["enable_attention_slicing"] = data.setdefault(
            "enable_attention_slicing"
        )
        data["xformers_efficient"] = data.setdefault("xformers_efficient")
        data["group_offload"] = data.setdefault("group_offload")
        data["sd_fuse_qkv_projections"] = data.setdefault(
            "sd_fuse_qkv_projections"
        )

        # compile
        data["compile"] = data.setdefault("compile")
        data["torch"] = data.setdefault("torch")

        # quantization
        data["quantize"] = data.setdefault("quantize")

        # attention
        data["attn_backend"] = data.setdefault("attn_backend")

        # debug, misc
        data["logging_level"] = data.setdefault("logging_level")
        if data["logging_level"] is not None:
            if data["logging_level"] in ["basic", 0, "0"]:
                self.log_level = 0
            elif data["logging_level"] in ["info", 1, "1"]:
                self.log_level = 1
            else:
                self.log_level = 2

        # checks
        is_applied_config = True
        for k, v in data.items():
            if (
                self.applied is None
                or k not in self.applied
                or v != self.applied[k]
            ):
                is_applied_config = False
                break
        if is_applied_config:
            self.progress = 100
            return "", 200
        else:
            try:
                self.log(0, "⏳ Initializing pipeline", rank_0_only=False)

                # reset current
                self.close_pipeline()

                # setup current
                self.loaded_model_index = json_path_to_dict(
                    f"{data['checkpoint']['checkpoint']}/model_index.json"
                )
                self.is_transformer_model_type = (
                    "transformer" in self.loaded_model_index
                )
                self.can_use_cachedit = self.is_transformer_model_type is True
                self.can_use_deepcache = (
                    self.is_transformer_model_type is False
                )
                self.pipeline_name = self.loaded_model_index["_class_name"]
                if data["checkpoint"].get("class_override") is not None:
                    pipeline_class_name = data["checkpoint"]["class_override"]
                else:
                    pipeline_class_name = self.pipeline_name
                self.log(0, f"ℹ️ Loading pipeline {pipeline_class_name}")
                pipeline_class = getattr(diffusers, pipeline_class_name)
                self.is_image_model = (
                    "video" not in pipeline_class_name.lower()
                    and "wan" not in pipeline_class_name.lower()
                )  # TODO: improve this
                self.backend_name = backend_name

                # torch tweaks
                if data["torch"] is not None:
                    cache_size_limit = data["torch"].get("torch_cache_limit")
                    accumulated_cache_size_limit = data["torch"].get(
                        "torch_accumlated_cache_limit"
                    )
                    capture_scalar_outputs = data["torch"].get(
                        "torch_capture_scalar"
                    )
                    if cache_size_limit is not None:
                        torch._dynamo.config.cache_size_limit = (
                            cache_size_limit
                        )
                    if accumulated_cache_size_limit is not None:
                        torch._dynamo.config.accumulated_cache_size_limit = (
                            accumulated_cache_size_limit
                        )
                    if capture_scalar_outputs is not None:
                        torch._dynamo.config.capture_scalar_outputs = (
                            capture_scalar_outputs
                        )
                """
                # TODO: these should be up to the user
                torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True, enable_cudnn=True)
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.force_fuse_int_mm_with_mul = True
                torch._inductor.config.use_mixed_mm = True
                if self.pipeline_name not in NON_DETERMINISTIC_PIPELINES:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
                """

                # update globals
                self.vae_dtype = (
                    torch.float16 if data["vae_fp16"] is True else None
                )
                self.torch_dtype = get_torch_type(data["torch_dtype"])

                kwargs = {}
                kwargs["torch_dtype"] = self.torch_dtype
                kwargs["use_safetensors"] = True
                kwargs["local_files_only"] = True
                kwargs["low_cpu_mem_usage"] = True
                kwargs["add_watermarker"] = False
                if data["checkpoint"].get("device_map") is not None:
                    if data["checkpoint"]["device_map"] in ["tp_plan_auto"]:
                        self.log(
                            0,
                            "⚠️ tp_plan_auto not supported for this model - ignoring",
                        )
                    else:
                        kwargs["device_map"] = data["checkpoint"]["device_map"]
                    self.device_map_modules.append("pipeline")

                # set transformer
                transformers_kwargs = {}
                parallelize_transformer = False
                if (
                    self.backend_name in ["parallel"]
                    and data.get("tp_config") is not None
                ):
                    transformers_kwargs["parallel_config"] = data["tp_config"]
                    parallelize_transformer = True
                if data["transformer"] is not None:
                    if self.is_transformer_model_type:
                        kwargs["transformer"] = self.load_model(
                            data["transformer"], model_args=transformers_kwargs
                        )
                    else:
                        kwargs["unet"] = self.load_model(
                            data["transformer"], model_args=transformers_kwargs
                        )
                elif parallelize_transformer is True:
                    if self.is_transformer_model_type:
                        kwargs["transformer"] = self.load_model(
                            data["checkpoint"],
                            subfolder="transformer",
                            model_args=transformers_kwargs,
                        )
                    else:
                        kwargs["unet"] = self.load_model(
                            data["checkpoint"],
                            subfolder="unet",
                            model_args=transformers_kwargs,
                        )
                clean()
                self.progress = 10

                # set vae
                vae_kwargs = {}
                if data["vae"] is not None:
                    kwargs["vae"] = self.load_model(
                        data["vae"], is_vae=True, model_args=vae_kwargs
                    )
                clean()
                self.progress = 20

                # set text encoder(s)
                encoder_kwargs = {}
                for k in ENCODER_MODULE_NAMES:
                    if data.get(k) is not None:
                        kwargs[k] = self.load_encoder(
                            data[k], model_args=encoder_kwargs
                        )
                    else:
                        te = self.load_encoder(
                            data["checkpoint"],
                            subfolder=k,
                            model_args=encoder_kwargs,
                        )
                        if te is not None:
                            kwargs[k] = te
                clean()
                self.progress = 30

                # set control net
                if data["control_net"] is not None:
                    kwargs["controlnet"] = self.load_model(data["control_net"])
                clean()
                self.progress = 40

                # quantize
                if data["quantize"] is not None:
                    kwargs["quantization_config"] = get_quantization_config(
                        data["quantize"], False
                    )

                # initialize pipeline
                self.pipe = pipeline_class.from_pretrained(
                    data["checkpoint"]["checkpoint"], **kwargs
                )
                del kwargs
                self.default_scheduler = copy.deepcopy(self.pipe.scheduler)
                if (
                    data["attn_backend"] is not None
                    and data["attn_backend"].get("backend") is not None
                ):
                    target = (
                        self.pipe.transformer
                        if self.is_transformer_model_type is True
                        else self.pipe.unet
                    )
                    target.set_attention_backend(
                        data["attn_backend"]["backend"]
                    )
                    self.log(
                        0,
                        f"ℹ️ Set attention backend to {data['attn_backend']['backend']}",
                    )
                clean()
                self.log(0, "✅ Pipeline initialized", rank_0_only=False)
                # for debugging
                if self.local_rank == 0:
                    # self.log(0, "\n\n\n" + str(self.pipe.transformer) + "\n\n\n")
                    # raise ValueError
                    pass
                self.progress = 50

                # set memory saving
                if data["enable_vae_slicing"] is True:
                    self.pipe.vae.enable_slicing()
                if data["enable_vae_tiling"] is True:
                    self.pipe.vae.enable_tiling()
                if data["enable_attention_slicing"] is True:
                    self.pipe.enable_attention_slicing()
                if data["xformers_efficient"] is True:
                    self.pipe.enable_xformers_memory_efficient_attention()  # NOTE: can cause tensor size mismatches
                self.progress = 60

                # group offloading
                target = (
                    self.pipe.transformer
                    if self.is_transformer_model_type
                    else self.pipe.unet
                )
                target.hf_device_map = None
                self.pipe.reset_device_map()
                if data["group_offload"] is None:
                    pass
                else:
                    transformer_offload_config = data["group_offload"].get(
                        "transformer"
                    )
                    encoder_offload_config = data["group_offload"].get(
                        "encoder"
                    )
                    vae_offload_config = data["group_offload"].get("vae")
                    misc_offload_config = data["group_offload"].get("misc")
                    onload_device = "cuda"
                    kwargs = {
                        "onload_device": torch.device(onload_device),
                        "non_blocking": True,
                    }

                    if transformer_offload_config is not None:
                        kwargs["offload_device"] = transformer_offload_config[
                            "offload_device"
                        ]
                        kwargs["offload_type"] = transformer_offload_config[
                            "offload_type"
                        ]
                        if kwargs["offload_type"] == "block_level":
                            kwargs["num_blocks_per_group"] = (
                                transformer_offload_config.get(
                                    "num_blocks_per_group"
                                )
                            )
                        if (
                            transformer_offload_config.get("use_stream")
                            is True
                        ):
                            kwargs["use_stream"] = True

                        for name in TRANSFORMER_MODULE_NAMES:
                            if hasattr(self.pipe, name):
                                module = getattr(self.pipe, name)
                                if module is not None:
                                    module.enable_group_offload(**kwargs)
                                    self.offloaded_modules.append(name)

                    if encoder_offload_config is not None:
                        kwargs["offload_device"] = encoder_offload_config[
                            "offload_device"
                        ]
                        kwargs["offload_type"] = encoder_offload_config[
                            "offload_type"
                        ]
                        if kwargs["offload_type"] == "block_level":
                            kwargs["num_blocks_per_group"] = (
                                encoder_offload_config.get(
                                    "num_blocks_per_group"
                                )
                            )
                        if encoder_offload_config.get("use_stream") is True:
                            kwargs["use_stream"] = True
                        for name in ENCODER_MODULE_NAMES:
                            if hasattr(self.pipe, name):
                                module = getattr(self.pipe, name)
                                if module is not None:
                                    apply_group_offloading(module, **kwargs)
                                    self.offloaded_modules.append(name)

                    if vae_offload_config is not None:
                        kwargs["offload_device"] = vae_offload_config[
                            "offload_device"
                        ]
                        kwargs["offload_type"] = vae_offload_config[
                            "offload_type"
                        ]
                        if kwargs["offload_type"] == "block_level":
                            kwargs["num_blocks_per_group"] = (
                                vae_offload_config.get("num_blocks_per_group")
                            )
                        if vae_offload_config.get("use_stream") is True:
                            kwargs["use_stream"] = True
                        for name in VAE_MODULE_NAMES:
                            if hasattr(self.pipe, name):
                                module = getattr(self.pipe, name)
                                if module is not None:
                                    module.enable_group_offload(**kwargs)
                                    self.offloaded_modules.append(name)

                    if misc_offload_config is not None:
                        kwargs["offload_device"] = misc_offload_config[
                            "offload_device"
                        ]
                        kwargs["offload_type"] = misc_offload_config[
                            "offload_type"
                        ]
                        if kwargs["offload_type"] == "block_level":
                            kwargs["num_blocks_per_group"] = (
                                misc_offload_config.get("num_blocks_per_group")
                            )
                        if misc_offload_config.get("use_stream") is True:
                            kwargs["use_stream"] = True
                        for name in MISC_MODULE_NAMES:
                            if hasattr(self.pipe, name):
                                module = getattr(self.pipe, name)
                                if module is not None:
                                    try:
                                        module.enable_group_offload(**kwargs)
                                        self.offloaded_modules.append(name)
                                    except Exception:
                                        pass
                clean()
                self.progress = 70

                # set lora
                if data["lora"] is not None:
                    target = (
                        self.pipe.transformer
                        if self.is_transformer_model_type
                        else self.pipe.unet
                    )
                    names = []
                    lokr_names = []
                    keys_to_delete = []
                    for k, v in data["lora"].items():
                        self.log(1, f"⏳ Loading adapter: {k}")
                        if k.endswith(".safetensors") or k.endswith(".sft"):
                            weights = safetensors.torch.load_file(
                                k, device=f"cuda:{self.local_rank}"
                            )
                        else:
                            weights = torch.load(
                                k,
                                map_location=torch.device(
                                    f"cuda:{self.local_rank}"
                                ),
                            )
                        # detect if this adapter is a lora or a lokr:
                        is_lora_adapter = True
                        for k2, v2 in weights.items():
                            if "lokr" in k2:
                                is_lora_adapter = False
                                break
                        w = k.split("/")[-1]
                        a = w if "." not in w else w.split(".")[0]
                        if is_lora_adapter is True:
                            names.append(a)
                            self.pipe.load_lora_weights(
                                weights,
                                weight_name=w,
                                adapter_name=a,
                                local_files_only=True,
                                low_cpu_mem_usage=True,
                            )
                            self.log(0, f"ℹ️ Added LoRA (scale={v}): {k}")
                        else:
                            # NOTE: this wont load a comfyui lokr - it still has to be for diffusers
                            weights = _convert_non_diffusers_lokr_to_diffusers(
                                weights
                            )
                            lycoris_net, _ = create_lycoris_from_weights(
                                v, k, target, weights_sd=weights
                            )
                            lycoris_net.merge_to()
                            lokr_names.append(a)
                            keys_to_delete.append(k)
                            self.log(0, f"ℹ️ Added LoKr (scale={v}): {k}")

                    for k in keys_to_delete:
                        del data["lora"][k]

                    if (
                        len(names) > 0
                        and len(self.pipe.get_active_adapters()) > 0
                    ):
                        target.set_adapters(names, list(data["lora"].values()))
                        loaded_adapters = target.active_adapters()
                        loaded_adapters_string = ""
                        for la in loaded_adapters:
                            loaded_adapters_string += "\n        " + la
                        self.log(
                            0, f"📟 Total loaded LoRAs: {len(loaded_adapters)}"
                        )
                        self.log(
                            1, f"📚 LoRa adapters: {loaded_adapters_string}"
                        )
                        if len(names) > 0:
                            self.adapter_names = names
                        else:
                            self.adapter_names = None
                    if len(lokr_names) > 0:
                        loaded_adapters_string = ""
                        for la in lokr_names:
                            loaded_adapters_string += "\n        " + la
                        self.log(
                            0, f"📟 Total loaded LoKrs: {len(lokr_names)}"
                        )
                        self.log(
                            1, f"📚 LoKr adapters: {loaded_adapters_string}"
                        )
                clean()
                self.progress = 80

                # compiles & optimizations
                if (
                    self.is_transformer_model_type is False
                    and data["sd_fuse_qkv_projections"]
                ):
                    self.pipe.fuse_qkv_projections()
                    self.log(0, "ℹ️ Fusing qkv projections")

                if data["compile"] is not None:
                    compile_transformer = data["compile"].get(
                        "compile_transformer"
                    )
                    compile_vae = data["compile"].get("compile_vae")
                    compile_encoder = data["compile"].get("compile_encoder")
                    compile_backend = data["compile"].get("compile_backend")
                    compile_mode = data["compile"].get("compile_mode")
                    compile_options = data["compile"].get("compile_options")
                    dynamic = data["compile"].get("dynamic")
                    fullgraph = data["compile"].get("fullgraph")

                    if (
                        compile_options is not None
                        and len(compile_options) > 5
                    ):
                        try:
                            compile_options = json.loads(compile_options)
                        except Exception:
                            self.log(
                                0,
                                "⚠️ Invalid JSON for compile_options - ignoring compile_options",
                            )
                            compile_options = None
                    else:
                        compile_options = None

                    if (
                        compile_mode is not None
                        and compile_options is not None
                    ):
                        self.log(
                            0,
                            "⚠️ compile_mode and compile_options are both defined - will ignore compile_mode",
                        )
                        compile_mode = None

                    compiler_config = {}
                    compiler_config["fullgraph"] = fullgraph is True
                    compiler_config["dynamic"] = dynamic is True
                    if compile_backend is not None:
                        compiler_config["backend"] = compile_backend
                    if compile_mode is not None:
                        compiler_config["mode"] = compile_mode
                    if compile_options is not None:
                        compiler_config["options"] = compile_options

                    if compile_transformer:
                        if self.is_transformer_model_type is True:
                            self.compile_helper("transformer", compiler_config)
                        else:
                            self.compile_helper("unet", compiler_config)
                    if compile_vae:
                        self.compile_helper("vae", compiler_config)
                    if compile_encoder:
                        self.compile_helper("encoder", compiler_config)
                clean()
                self.progress = 90

                # complete
                self.log(
                    0, "✅ Model initialization completed", rank_0_only=False
                )
                self.print_mem_usage()
                self.applied = data
                self.progress = 100
                self.initialized = True
                return "", 200
            except Exception:
                self.log(0, traceback.format_exc(), rank_0_only=False)
                return "", 500

    def load_model(
        self, model_dict, is_vae=False, subfolder=None, model_args=None
    ):
        is_checkpoint = True
        config_path = None

        model_path = model_dict.get("checkpoint")
        if model_path is None:
            is_checkpoint = False
            model_path = model_dict.get("model")
            config_path = model_dict.get("config")
            assert config_path is not None, (
                "You must provide a config_path when loading from a single file"
            )

        self.log(
            0,
            f"""ℹ️ Loading model: {model_path}
        with config: {str(config_path)}""",
        )
        kwargs = {}
        kwargs["torch_dtype"] = (
            self.vae_dtype if is_vae is True else self.torch_dtype
        )
        kwargs["use_safetensors"] = True
        kwargs["local_files_only"] = True
        kwargs["low_cpu_mem_usage"] = True
        if model_args is not None and len(model_args) > 0:
            kwargs.update(model_args)
        if subfolder is not None:
            kwargs["subfolder"] = subfolder
        if model_dict.get("device_map") is not None:
            if model_dict["device_map"] in ["tp_plan_auto"]:
                self.log(
                    0,
                    "⚠️ tp_plan_auto not supported for this model - ignoring",
                )
            else:
                kwargs["device_map"] = model_dict["device_map"]
            self.device_map_modules.append("vae" if is_vae else "transformer")

        if model_path.endswith(".bin") or model_path.endswith(".ckpt"):
            kwargs["use_safetensors"] = False  # NOTE: safetensors off
        elif model_path.endswith(".gguf"):
            kwargs["quantization_config"] = GGUFQuantizationConfigD(
                compute_dtype=self.torch_dtype
            )
            kwargs["use_safetensors"] = False  # NOTE: safetensors off

        if not is_checkpoint:
            kwargs["config"] = config_path
        else:
            if subfolder is None:
                config_path = f"{model_path}/config.json"
            else:
                config_path = f"{model_path}/{subfolder}/config.json"

        formatted = format_dict(kwargs, indent=8, indent_all=True)
        config_json = json_path_to_dict(config_path)
        # ignore class_override when subfolder is specified
        if model_dict.get("class_override") is not None and subfolder is None:
            model_class = getattr(diffusers, model_dict["class_override"])
        else:
            try:
                arches = config_json.get("architectures")
                if arches is not None:
                    arch = arches[0]
                else:
                    arch = config_json.get("_class_name")
                model_class = getattr(diffusers, arch)
                self.log(
                    0,
                    f"ℹ️ Using {arch} to load this model, with args: {formatted}",
                )
            except Exception:
                model_class = AutoModelD
                self.log(
                    0,
                    f"ℹ️ Using AutoModel (Diffusers) to load this model, with args: {formatted}",
                )

        if is_checkpoint:
            return model_class.from_pretrained(model_path, **kwargs)
        else:
            return model_class.from_single_file(model_path, **kwargs)

    def load_encoder(self, model_dict, subfolder=None, model_args=None):
        is_checkpoint = True
        config_path = None

        model_path = model_dict.get("checkpoint")
        if model_path is None:
            is_checkpoint = False
            model_path = model_dict.get("model")
            config_path = model_dict.get("config")
            assert config_path is not None, (
                "You must provide a config_path when loading from a single file"
            )

        self.log(
            0,
            f"""ℹ️ Loading model: {model_path}
        with config: {str(config_path)}""",
        )

        kwargs = {}
        kwargs["dtype"] = self.torch_dtype
        kwargs["use_safetensors"] = True
        kwargs["local_files_only"] = True
        kwargs["low_cpu_mem_usage"] = True
        if model_args is not None and len(model_args) > 0:
            kwargs.update(model_args)
        if subfolder is not None:
            kwargs["subfolder"] = subfolder
        if model_dict.get("device_map") is not None:
            if model_dict["device_map"] in ["tp_plan_auto"]:
                kwargs["tp_plan"] = "auto"
            else:
                kwargs["device_map"] = model_dict["device_map"]
            self.device_map_modules.append("encoder")

        if (
            model_path.endswith(".bin")
            or model_path.endswith(".ckpt")
            or model_path.endswith(".gguf")
        ):
            kwargs["use_safetensors"] = False  # NOTE: safetensors off

        if not is_checkpoint:
            kwargs["config"] = config_path
            if model_path.endswith(".gguf"):
                kwargs["gguf_file"] = model_path
                model_path = model_path.split("/")[-1].split(".")[0]
            elif model_path.endswith(".safetensors") or model_path.endswith(
                ".sft"
            ):
                kwargs["state_dict"] = load_file(model_path)
                model_path = None
        else:
            if subfolder is None:
                config_path = f"{model_path}/config.json"
            else:
                config_path = f"{model_path}/{subfolder}/config.json"
                if not Path(config_path).exists():
                    return None

        formatted = format_dict(kwargs, indent=8, indent_all=True)
        config_json = json_path_to_dict(config_path)
        # ignore class_override when subfolder is specified
        if model_dict.get("class_override") is not None and subfolder is None:
            model_class = getattr(transformers, model_dict["class_override"])
        else:
            try:
                arches = config_json.get("architectures")
                if arches is not None:
                    arch = arches[0]
                else:
                    arch = config_json.get("_class_name")
                model_class = getattr(transformers, arch)
                self.log(
                    0,
                    f"ℹ️ Using {arch} to load this model, with args: {formatted}",
                )
            except Exception:
                model_class = AutoModelT
                self.log(
                    0,
                    f"ℹ️ Using AutoModel (Transformers) to load this model, with config {formatted}",
                )
            return model_class.from_pretrained(model_path, **kwargs)

    def fuse_pipeline_loras(self):
        if self.adapter_names is not None and self.loras_are_fused is False:
            target = (
                self.pipe.transformer
                if self.is_transformer_model_type
                else self.pipe.unet
            )
            target.fuse_lora(adapter_names=self.adapter_names, lora_scale=1.0)
            self.pipe.unload_lora_weights()
            self.loras_are_fused = True
        return

    def compile_helper(self, target, compile_config):
        if target in ["transformer", "unet"]:
            for k in TRANSFORMER_MODULE_NAMES:
                if (
                    hasattr(self.pipe, k)
                    and getattr(self.pipe, k, None) is not None
                ):
                    self.fuse_pipeline_loras()
                    setattr(
                        self.pipe,
                        k,
                        torch.compile(getattr(self.pipe, k), **compile_config),
                    )
        elif target == "vae":
            for k in VAE_MODULE_NAMES:
                if (
                    hasattr(self.pipe, k)
                    and getattr(self.pipe, k, None) is not None
                ):
                    setattr(
                        self.pipe,
                        k,
                        torch.compile(getattr(self.pipe, k), **compile_config),
                    )
        elif target == "encoder":
            for k in ENCODER_MODULE_NAMES:
                if (
                    hasattr(self.pipe, k)
                    and getattr(self.pipe, k, None) is not None
                ):
                    setattr(
                        self.pipe,
                        k,
                        torch.compile(getattr(self.pipe, k), **compile_config),
                    )
        else:
            self.log(0, "❌ Unknown compile target - not compiling")
            return
        self.log(0, f"⚙️ {target} will be compiled")
        return

    def prepare_inputs(self, data):
        """
        NOTE:
        data is kwargs for the pipeline.

        valid custom arguments (not directly passed to pipeline):
            - start_step    (Int for when to start inference)
            - end_step      (Int for when to stop inference)
            - scheduler     (Custom dict for setting the scheduler)
            - compel        (True or False)
        """
        self.print_params(data)

        # checks
        assert (
            data.get("image") is not None
            or data.get("prompt") is not None
            or data.get("prompt_embeds") is not None
        ), "No input provided"
        assert (
            data.get("prompt") is None or data.get("prompt_embeds") is None
        ), "Provide only one positive input"
        assert (
            data.get("negative_prompt") is None
            or data.get("negative_prompt_embeds") is None
        ), "Provide only one negative input"
        if self.applied.get("control_net") is not None:
            assert data.get("control_image") is not None, (
                "No ControlNet image provided for a ControlNet-loaded pipeline"
            )

        # decodes
        if data.get("image") is not None:
            data["image"] = decode_b64_and_unpickle(data["image"])
        if data.get("control_image") is not None:
            data["control_image"] = decode_b64_and_unpickle(
                data["control_image"]
            )
        if data.get("latent") is not None:
            data["latent"] = decode_b64_and_unpickle(data["latent"])
        if data.get("prompt_embeds") is not None:
            data["prompt_embeds"] = decode_b64_and_unpickle(
                data["prompt_embeds"]
            )
        if data.get("pooled_prompt_embeds") is not None:
            data["pooled_prompt_embeds"] = decode_b64_and_unpickle(
                data["pooled_prompt_embeds"]
            )
        if data.get("negative_prompt_embeds") is not None:
            data["negative_prompt_embeds"] = decode_b64_and_unpickle(
                data["negative_prompt_embeds"]
            )
        if data.get("negative_pooled_prompt_embeds") is not None:
            data["negative_pooled_prompt_embeds"] = decode_b64_and_unpickle(
                data["negative_pooled_prompt_embeds"]
            )

        # delete empty/invalid/unused
        if data.get("prompt") is not None and len(data["prompt"]) == 0:
            del data["prompt"]
        if (
            data.get("negative_prompt") is not None
            and len(data["negative_prompt"]) == 0
        ):
            del data["negative_prompt"]
        if data.get("start_step") is not None and data["start_step"] <= 0:
            del data["start_step"]
        if (
            data.get("end_step") is not None
            and data["end_step"] >= data["num_inference_steps"]
        ):
            del data["end_step"]

        # load images
        if data.get("image") is not None:
            data["image"] = load_image(data["image"])
        if (
            data.get("control_image") is not None
            and self.applied.get("control_net") is not None
        ):
            data["control_image"] = load_image(data["control_image"])

        # check all data to see if there are still any pickled & encoded values
        for k, v in data.items():
            try:
                result = decode_b64_and_unpickle(v)
                if result is not None:
                    data[k] = result
            except Exception:
                pass
        return data

    def set_scheduler(self, scheduler_config):
        try:
            klass = getattr(
                diffusers.schedulers, scheduler_config["scheduler"]
            )
            assert klass is not None, (
                f"Diffusers schedulers does not have class {scheduler_config['scheduler']}"
            )
        except Exception:
            raise NotImplementedError
        params = {}
        for k, v in scheduler_config.items():
            if k != "scheduler":
                params[k] = v
        self.pipe.scheduler = klass.from_config(params)
        return

    def set_scheduler_timesteps(self, start=0):
        clean_override_function(self.pipe.__class__, "retrieve_timesteps")
        if start is not None and start > 0:
            old_retrieve_timesteps = get_function_from_class(
                self.pipe.__class__, "retrieve_timesteps"
            )

            def new_retrieve_timesteps(*args, **kwargs):
                def result():
                    return old_retrieve_timesteps(*args, **kwargs)

                timesteps, num_inference_steps = result()

                if len(timesteps) > 0 and start > 0:
                    timesteps = self.pipe.scheduler.timesteps[
                        start * self.pipe.scheduler.order :
                    ]
                    if hasattr(self.pipe.scheduler, "set_begin_index"):
                        self.pipe.scheduler.set_begin_index(
                            start * self.pipe.scheduler.order
                        )
                return timesteps, num_inference_steps - start

            override_function(
                self.pipe.__class__,
                "retrieve_timesteps",
                new_retrieve_timesteps,
            )
        return

    def setup_inference(self, data, callbacks={}):
        self.progress = 0

        # set scheduler
        custom_timesteps, custom_sigmas = None, None
        if data.get("scheduler") is not None:
            self.pipe.scheduler = None
            data["scheduler"] = json.loads(data["scheduler"])
            custom_timesteps = data["scheduler"].pop("timesteps", None)
            custom_sigmas = data["scheduler"].pop("sigmas", None)
            self.set_scheduler(data["scheduler"])
            self.log(
                0,
                f"ℹ️ Set scheduler to {str(type(self.pipe.scheduler).__name__)}",
            )
            if custom_timesteps is not None and custom_sigmas is not None:
                self.log(
                    0,
                    "⚠️ Both timesteps and sigmas were provided! Only one is allowed - using timesteps.",
                )
                custom_sigmas = None
        elif self.pipe.scheduler != self.default_scheduler:
            self.pipe.scheduler = None
            self.pipe.scheduler = copy.deepcopy(self.default_scheduler)
            self.log(
                0,
                f"ℹ️ Reverted scheduler to {str(type(self.pipe.scheduler).__name__)}",
            )

        if data.get("start_step") is not None and data["start_step"] > 0:
            if self.pipeline_name not in START_STEP_WORKAROUND_PIPELINES:
                self.set_scheduler_timesteps(start=data["start_step"])
        else:
            self.set_scheduler_timesteps()

        # patch to remove unsupported kwargs from flowmatch schedulers
        if (
            data.get("scheduler") is not None
            and "FlowMatch" in self.loaded_model_index["scheduler"][1]
        ):
            clean_override_function(self.pipe.__class__, "retrieve_timesteps")
            old_retrieve_timesteps = get_function_from_class(
                self.pipe.__class__, "retrieve_timesteps"
            )

            def new_retrieve_timesteps(*args, **kwargs):
                if not check_function_kwargs(
                    self.pipe.scheduler.set_timesteps, "mu"
                ):
                    del kwargs["mu"]
                if not check_function_kwargs(
                    self.pipe.scheduler.set_timesteps, "sigmas"
                ):
                    del kwargs["sigmas"]

                def result():
                    return old_retrieve_timesteps(*args, **kwargs)

                timesteps, num_inference_steps = result()
                return timesteps, num_inference_steps

            override_function(
                self.pipe.__class__,
                "retrieve_timesteps",
                new_retrieve_timesteps,
            )
            self.log(1, "ℹ️ Patched flowmatch scheduler")

        # NOTE: no more scheduler after this line
        if data.get("scheduler") is not None:
            del data["scheduler"]

        # process input latent
        if data.get("latent") is not None:
            data["latent"] = self.process_input_latent(data["latent"])

        # set up progress bar
        def set_step_progress(pipe, index, timestep, callback_kwargs):
            if torch.any(callback_kwargs["latents"].isnan()):
                self.log(
                    0,
                    "⁉️ NaN detected in latents - stopping generation",
                    rank_0_only=False,
                )
                self.pipe._interrupt = True
                self.progress = 100
                return callback_kwargs
            the_index = index / self.pipe.scheduler.order

            # host callbacks
            for k, v in callbacks.items():
                if the_index == int(k):
                    try:
                        v(data, index, timestep, callback_kwargs)
                    except Exception:
                        pass

            # start_step workaround
            if (
                data.get("latent") is not None
                and self.pipeline_name in START_STEP_WORKAROUND_PIPELINES
            ):
                if the_index == data.get("start_step", 0):
                    self.log(
                        0,
                        f"ℹ️ Injecting latent at step {str(the_index)}",
                        rank_0_only=False,
                    )
                    callback_kwargs["latents"] = data["latent"]
                elif the_index < data.get("start_step", 0):
                    callback_kwargs["latents"] = torch.zeros_like(
                        data["latent"]
                    )

            # end calculations
            end = data["num_inference_steps"]
            if custom_timesteps is not None or custom_sigmas is not None:
                end = len(
                    custom_timesteps
                    if custom_timesteps is not None
                    else custom_sigmas
                )
            if data.get("end_step") is not None and data["end_step"] < end:
                end = data["end_step"]

            # end_step
            # if data.get("end_step", None) is not None and the_index + 1 > data["end_step"]:
            if the_index + 1 > end:
                self.log(
                    0,
                    "ℹ️ End step reached - stopping generation",
                    rank_0_only=False,
                )
                self.pipe._interrupt = True

            # self.progress = int((the_index + 1 + data.get("start_step", 0)) / min(data["num_inference_steps"], data["end_step"] if data["end_step"] is not None else 2 ** 32 - 1) * 100)
            self.progress = int(
                (the_index + 1 + data.get("start_step", 0)) / end * 100
            )
            if self.progress == 100 or self.pipe._interrupt is True:
                self.print_timesteps(self.pipe.scheduler.timesteps)
                if (
                    str(type(self.pipe.scheduler).__name__)
                    == "EulerDiscreteScheduler"
                ):
                    self.print_sigmas(self.pipe.scheduler.sigmas)
            return callback_kwargs

        # compel
        if (
            data.get("compel") is True
            and data.get("prompt_embeds") is None
            and data.get("negative_prompt_embeds") is None
        ):
            if self.pipeline_name in ["StableDiffusionXLPipeline"]:
                embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
            else:
                embeddings_type = (
                    ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                )
            compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[
                    self.pipe.text_encoder,
                    self.pipe.text_encoder_2,
                ],
                returned_embeddings_type=embeddings_type,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
            )
            data["prompt_embeds"], data["pooled_prompt_embeds"] = compel(
                [data["prompt"]]
            )
            del data["prompt"]
            if (
                data.get("negative_prompt") is not None
                and len(data["negative_prompt"]) > 0
            ):
                (
                    data["negative_prompt_embeds"],
                    data["negative_pooled_prompt_embeds"],
                ) = compel([data["negative_prompt"]])
                del data["negative_prompt"]
        else:
            if data.get("prompt_embeds") is not None:
                data["pooled_prompt_embeds"] = data["prompt_embeds"][0][1][
                    "pooled_output"
                ]
                data["prompt_embeds"] = data["prompt_embeds"][0][0]
            if data.get("negative_prompt_embeds") is not None:
                data["negative_pooled_prompt_embeds"] = data[
                    "negative_prompt_embeds"
                ][0][1]["pooled_output"]
                data["negative_prompt_embeds"] = data[
                    "negative_prompt_embeds"
                ][0][0]

        if (
            data.get("prompt_embeds") is not None
            and data.get("negative_prompt_embeds") is not None
        ):
            if self.pipeline_name not in ["ZImagePipeline"]:
                dim_1 = max(
                    data["prompt_embeds"].size(1),
                    data["negative_prompt_embeds"].size(1),
                )
                pad_pos = max(0, dim_1 - data["prompt_embeds"].size(1))
                pad_neg = max(
                    0, dim_1 - data["negative_prompt_embeds"].size(1)
                )
                if pad_pos > 0:
                    data["prompt_embeds"] = F.pad(
                        data["prompt_embeds"], (0, 0, 0, pad_pos)
                    )
                if pad_neg > 0:
                    data["negative_prompt_embeds"] = F.pad(
                        data["negative_prompt_embeds"], (0, 0, 0, pad_neg)
                    )

        # delete non-standard kwargs
        non_standard_kwargs = ["start_step", "end_step"]

        # set pipe arguments
        kwargs = {}
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(
            data.pop("seed")
        )
        kwargs["callback_on_step_end"] = set_step_progress
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        if custom_timesteps is not None:
            kwargs["timesteps"] = custom_timesteps
        if custom_sigmas is not None:
            kwargs["sigmas"] = custom_sigmas
        if self.pipeline_name in PIL_OUTPUT_PIPELINES:
            kwargs["output_type"] = "pil"
        else:
            kwargs["output_type"] = "latent"
        for k, v in data.items():
            if k in non_standard_kwargs:
                continue
            else:
                kwargs[k] = v
        return kwargs

    def process_input_latent(self, latents):
        # NOTE: this matters because it sometimes becomes NaN if you just move it to the target device
        latents = latents.to(
            dtype=self.torch_dtype, device=torch.device("cpu")
        )
        if self.is_transformer_model_type:
            latents = latents.to(self.pipe.transformer.device)
        else:
            latents = latents.to(self.pipe.unet.device)
        latents = add_alpha_to_latent(latents)
        return latents

    def process_latent_for_output(self, latents, is_latent_output):
        latents = add_alpha_to_latent(latents)

        if is_latent_output:
            return latents

        if self.vae_dtype is not None:
            default_vae_dtype = self.pipe.vae.dtype
            self.pipe.vae = self.pipe.vae.to(self.vae_dtype)
        latents = latents.to(
            dtype=self.pipe.vae.dtype, device=self.pipe.vae.device
        )
        if self.pipeline_name in ["Krea2Pipeline"]:
            # copied from diffusers
            latents_mean = (
                torch.tensor(self.pipe.vae.config.latents_mean)
                .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(
                self.pipe.vae.config.latents_std
            ).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            latents = self.pipe.vae.decode(latents, return_dict=False)[0][
                :, :, 0
            ]
        elif self.pipeline_name in ["QwenImage21Pipeline"]:
            # copied from diffusers
            latents_mean = (
                torch.tensor(self.pipe.vae.config.latents_mean)
                .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.pipe.vae.config.latents_std)
                .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std + latents_mean
            latents = self.pipe.vae.decode(latents, return_dict=False)[0][
                :, :, 0
            ]
        else:
            if self.pipeline_name not in [
                "Flux2Pipeline",
                "Flux2KleinPipeline",
                "QwenImage21Pipeline",
            ]:
                latents = latents / self.pipe.vae.config.scaling_factor
            latents = self.pipe.vae.decode(latents, return_dict=False)[0]
        latents = self.pipe.image_processor.postprocess(
            latents, output_type="pil"
        )
        if self.vae_dtype is not None:
            self.pipe.vae = self.pipe.vae.to(default_vae_dtype)
        return latents

    def convert_latent_to_output_latent(self, latents):
        with torch.no_grad():
            latents = self.process_latent_for_output(latents, True)
        return latents

    def convert_latent_to_image(self, latents):
        with torch.no_grad():
            latents = self.process_latent_for_output(latents, False)
        return latents

    def get_output_images(self, output_images, data):
        if self.pipeline_name in ["FluxPipeline", "QwenImage21Pipeline"]:
            output_images = self.pipe._unpack_latents(
                output_images,
                data["height"],
                data["width"],
                self.pipe.vae_scale_factor,
            )
        elif self.pipeline_name in ["Krea2Pipeline"]:
            output_images = self.pipe._unpack_latents(
                output_images, data["height"], data["width"]
            )
        return output_images
