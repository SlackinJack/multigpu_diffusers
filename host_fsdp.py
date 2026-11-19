import argparse
import cache_dit
import diffusers
import logging
import os
import signal
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from DeepCache import DeepCacheSDHelper
from datetime import timedelta
from flask import Flask, request, jsonify, cli
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from collections import defaultdict
import torch.nn as nn


import modules.host_common as Common
from modules.custom_plans import get_custom_tp_plans
from modules.host_common import (
    GENERIC_HOST_ARGS,
    GENERIC_HOST_ARGS_TOGGLES,
    CommonHost,
    get_deepcache_config,
    get_cache_dit_config,
)
from modules.utils import clean, pickle_and_encode_b64


logging.getLogger("werkzeug").setLevel(logging.CRITICAL)
logging.getLogger("flask").setLevel(logging.CRITICAL)
logging.getLogger("werkzeug").disabled = True
logging.getLogger("flask").disabled = True
cli.show_server_banner = lambda *_: None
app = Flask(__name__)
base = None
fsdp_config = None


def __initialize_distributed_environment():
    global base
    base = CommonHost()
    mp.set_start_method("spawn", force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank, timeout=timedelta(days=1))
    base.local_rank = local_rank
    base.set_logger()
    base.initialized = True
    return


def __run_host():
    parser = argparse.ArgumentParser()
    for k, v in GENERIC_HOST_ARGS.items():
        parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:
        parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()

    torch._logging.set_logs(all=logging.CRITICAL)
    if base.local_rank == 0:
        base.log(1, "ℹ️ Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port, debug=False, use_reloader=False)
    else:
        while True:
            base.log(0, "⏳ Waiting for tasks", rank_0_only=False)
            params = [{"stop": True}]
            dist.broadcast_object_list(params, src=0)
            if params[0].get("stop") is not None:
                base.log(0, "🛑 Received exit signal - shutting down", rank_0_only=False)
                __close_host()
                return
            elif params[0].get("sleep") is not None and params[0].get("time") is not None:
                t = params[0].get("time")
                base.log(0, f"💤 Received sleep signal - pausing for {t} seconds", rank_0_only=False)
                time.sleep(int(t))
                base.log(0, "⏰ Sleep finished", rank_0_only=False)
            else:
                base.log(0, "📋 Received task", rank_0_only=False)
                __handle_request_parallel(*params)
    return


@app.route("/<path>", methods=["GET", "POST"])
def handle_path(path):
    match path:
        # status
        case "initialize":
            return base.get_initialized_flask()
        case "applied":
            return base.get_applied()
        case "progress":
            return base.get_progress_flask()

        # generation
        case "apply":
            return __apply_pipeline(request.json)
        case "generate":
            return __generate_image(request.json)
        case "offload":
            return "Operation not supported by this host", 500
        case "sleep":
            return __handle_sleep(request.json)
        case "close":
            base.log(0, "🛑 Received exit signal - shutting down", rank_0_only=False)
            dist.broadcast_object_list([{"stop": True}], src=0)
            __close_host()
            # raise HostShutdown
            return "", 200
        case _:
            return "", 404


def __close_host():
    global fsdp_config, dist
    fsdp_config = None
    base.close_pipeline()
    dist.barrier()
    dist.destroy_process_group()
    dist = None
    os.kill(os.getpid(), signal.SIGTERM)
    return


def __handle_request_parallel(data):
    if data.get("checkpoint") is not None:
        __apply_pipeline_parallel(data)
    elif data.get("seed") is not None:
        __generate_image_parallel(data)
    else:
        assert False, "Unknown data type"


def __apply_pipeline(data):
    dist.broadcast_object_list([data], src=0)
    response = __apply_pipeline_parallel(data)
    return response


def __generate_image(data):
    dist.broadcast_object_list([data], src=0)
    response = __generate_image_parallel(data)
    return jsonify(response)


def __handle_sleep(data):
    dist.broadcast_object_list([data], src=0)
    return "", 200


def __apply_pipeline_parallel(data):
    global fsdp_config, enable_cache_dit_runonce
    # Params:
    # deep_cache
    # deep_cache_interval
    # deep_cache_id
    # cache_dit
    config = data.get("backend")
    assert config is not None, "Configuration must be provided"

    with torch.no_grad():
        result = base.setup_pipeline(data, backend_name="fsdp")

    if result[1] == 200:
        base.log(1, "ℹ️ Initializing FSDP")
        mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
        dist.barrier()
        target = base.pipe.transformer if base.is_transformer_model_type else base.pipe.unet

        # TODO: this should be up to the user
        def shard_model(model):
            target_names = ["transformer", "unet"]
            for module in list(model.modules()):
                class_name = type(module).__name__.lower()
                if module is not model and any(name.lower() in class_name for name in target_names):
                    fully_shard(module, mesh=mesh, reshard_after_forward=True)
            fully_shard(model, mesh=mesh, reshard_after_forward=True)

        shard_model(target)
        dist.barrier()
        fsdp_config = config
        enable_cache_dit_runonce = False
    return result


def __generate_image_parallel(data):
    global enable_cache_dit_runonce

    if base.local_rank < dist.get_world_size():
        data = base.prepare_inputs(data)

        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()

            # inference kwargs
            kwargs = base.setup_inference(data)

            # caching
            enable_deepcache = (
                base.can_use_deepcache is True
                and fsdp_config.get("deepcache_config") is not None
            )
            enable_cache_dit = (
                base.can_use_cachedit is True
                and fsdp_config.get("cache_dit_config") is not None
            )
            deepcache_config = None
            cache_dit_config = None
            if enable_deepcache:
                deepcache_config = fsdp_config.get("deepcache_config")
            if enable_cache_dit:
                cache_dit_config = fsdp_config.get("cache_dit_config")

            # inference
            if enable_deepcache:
                helper = DeepCacheSDHelper(pipe=base.pipe)
                dc_config = get_deepcache_config(deepcache_config)
                helper.set_params(**dc_config)
                helper.enable()
                base.log(0, "ℹ️ DeepCache enabled", rank_0_only=False)
            if enable_cache_dit:
                cd_config = get_cache_dit_config(cache_dit_config)
                if not enable_cache_dit_runonce:
                    cache_dit.enable_cache(
                        base.pipe,
                        num_inference_steps=kwargs["num_inference_steps"],
                        verbose=False,
                        **cd_config,
                    )
                    enable_cache_dit_runonce = True
                    base.log(0, "ℹ️ CacheDiT enabled", rank_0_only=False)
                else:
                    base.log(0, "ℹ️ Using CacheDiT", rank_0_only=False)
                cache_dit.refresh_context(
                    base.pipe.transformer,
                    num_inference_steps=kwargs["num_inference_steps"],
                    verbose=False,
                    **cd_config,
                )
            start_time = time.perf_counter()
            output = base.pipe(**kwargs)
            end_time = time.perf_counter()
            base.log(0, f"⏱️ Processing time: {end_time - start_time:0.3f}")
            if enable_deepcache:
                helper.disable()
                base.log(0, "ℹ️ DeepCache disabled", rank_0_only=False)

        # clean up
        clean()

        # output
        if base.local_rank == 0:
            if output is not None:
                if base.is_image_model:
                    if base.pipeline_name in ["StableDiffusionUpscalePipeline"]:
                        output = output.images[0]
                    else:
                        base.pipe.vae = base.pipe.vae.to("cuda:0")
                        output_images = output.images
                        output_images = base.get_output_images(output_images, data)
                        images = base.convert_latent_to_image(output_images)
                        latents = base.convert_latent_to_output_latent(output_images)
                        images = pickle_and_encode_b64(images[0])
                        latents = pickle_and_encode_b64(latents)
                        base.pipe.vae = base.pipe.vae.to("cpu")
                        return {
                            "message": "OK",
                            "output": images,
                            "latent": latents,
                            "is_image": True,
                        }
                else:
                    output = output.frames[0]
                output = pickle_and_encode_b64(output)
                return {
                    "message": "OK",
                    "output": output,
                    "is_image": False,
                }
            else:
                return {
                    "message": "No image from pipeline",
                    "output": None,
                    "is_image": False,
                }


if __name__ == "__main__":
    __initialize_distributed_environment()
    __run_host()
