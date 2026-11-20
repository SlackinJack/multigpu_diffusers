import argparse
import cache_dit
import logging
import os
import signal
import time
import torch
from DeepCache import DeepCacheSDHelper
from flask import Flask, request, cli


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
backend_config = None
enable_cache_dit_runonce = False


def __initialize_environment():
    global base
    base = CommonHost()
    base.local_rank = 0
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
    base.log(1, "ℹ️ Starting Flask host", rank_0_only=False)
    app.run(host="localhost", port=args.port, debug=False, use_reloader=False)
    return


@app.route("/<path>", methods=["GET", "POST"])
def handle_path(path):
    global base
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
            return __apply_pipeline_parallel(request.json)
        case "generate":
            return __generate_image_parallel(request.json)
        case "offload":
            return __offload_modules_parallel()
        case "sleep":
            return "Operation not supported by this host", 500
        case "close":
            base.log(0, "🛑 Received exit signal - shutting down", rank_0_only=False)
            base.close_pipeline()
            base = None
            os.kill(os.getpid(), signal.SIGTERM)
            # raise HostShutdown
            return "", 200
        case _:
            return "", 404


def __offload_modules_parallel():
    return __move_pipe("cpu")


def __move_pipe_module(module, dev):
    try:
        mod = vars(base.pipe)[module]
        c_device = torch.device(mod.device)
        t_device = torch.device(dev)
        if c_device != t_device:
            vars(base.pipe)[module] = mod.to(t_device)
            if "cpu" in dev:
                vars(base.pipe)[module] = mod.cpu()
            elif "cuda" in dev:
                torch.cuda.set_device(t_device)
                torch.cuda.synchronize()
            return 0
        else:
            return -1
    except Exception:
        return 1


def __move_pipe(device):

    if len(base.offloaded_modules) > 0:
        return "ℹ️ Group offloading active - not offloading", 500

    if base.pipeline_name in ["QwenImage21Pipeline"]:
        return "ℹ️ Pipeline does not need to be moved", 200

    flag_vae_sliced = False
    if base.applied.get("enable_vae_slicing") is not None and base.applied.get("enable_vae_slicing") is True:
        flag_vae_sliced = True
        base.pipe.disable_vae_slicing()

    flag_vae_tiled = False
    if base.applied.get("enable_vae_tiling") is not None and base.applied.get("enable_vae_tiling") is True:
        flag_vae_tiled = True
        base.pipe.disable_vae_tiling()

    flag_attention_sliced = False
    if base.applied.get("enable_attention_slicing") is not None and base.applied.get("enable_attention_slicing") is True:
        flag_attention_sliced = True
        base.pipe.disable_attention_slicing()

    moved = []
    not_moved = []
    alr_moved = []
    for k, v in base.pipe.components.items():
        if v is not None:
            has_moved = __move_pipe_module(k, device)
            match has_moved:
                case 0:
                    moved.append(k)
                case 1:
                    not_moved.append(k)
                case -1:
                    alr_moved.append(k)
    base.pipe = base.pipe.to(torch.device(device))

    if flag_vae_sliced:
        base.pipe.enable_vae_slicing()
    if flag_vae_tiled:
        base.pipe.enable_vae_tiling()
    if flag_attention_sliced:
        base.pipe.enable_attention_slicing()

    clean()
    if ":" in device:
        device = device.split(":")[0]
    msg = f"""➡️ Moved pipeline components:
        Moved to {device}: {str(moved)}
        Not moved to {device}: {str(not_moved)}
        Already on {device}: {str(alr_moved)}"""
    base.log(2, msg)
    return msg, 200


def __apply_pipeline_parallel(data):
    global backend_config, enable_cache_dit_runonce
    # Params:
    # deep_cache
    # deep_cache_interval
    # deep_cache_id
    # cache_dit
    backend_config = data.get("backend")
    assert backend_config is not None, "Backend configuration must be provided"
    with torch.no_grad():
        result = base.setup_pipeline(data, backend_name="single")
        if result[1] == 200:
            if len(base.device_map_modules) == 0:
                base.pipe = base.pipe.to("cuda")
            enable_cache_dit_runonce = False
        return result


def __generate_image_parallel(data):
    global enable_cache_dit_runonce
    data = base.prepare_inputs(data)

    with torch.inference_mode():
        if len(base.offloaded_modules) == 0 and len(base.device_map_modules) == 0:
            __move_pipe("cuda")
        torch.cuda.reset_peak_memory_stats()

        # inference kwargs
        kwargs = base.setup_inference(data)

        # caching
        enable_deepcache = base.can_use_deepcache is True and backend_config.get("deepcache_config") is not None
        enable_cache_dit = base.can_use_cachedit is True and backend_config.get("cache_dit_config") is not None
        deepcache_config = None
        cache_dit_config = None
        if enable_deepcache:
            deepcache_config = backend_config.get("deepcache_config")
        if enable_cache_dit:
            cache_dit_config = backend_config.get("cache_dit_config")

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
                cache_dit.enable_cache(base.pipe, num_inference_steps=kwargs["num_inference_steps"], verbose=False, **cd_config)
                enable_cache_dit_runonce = True
                base.log(0, "ℹ️ CacheDiT enabled", rank_0_only=False)
            else:
                base.log(0, "ℹ️ Using CacheDiT", rank_0_only=False)
            cache_dit.refresh_context(base.pipe.transformer, num_inference_steps=kwargs["num_inference_steps"], verbose=False, **cd_config)

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
        if output is not None:
            if base.is_image_model:
                if base.pipeline_name in ["StableDiffusionUpscalePipeline"]:
                    output = output.images[0]
                else:
                    output_images = output.images
                    output_images = base.get_output_images(output_images, data)
                    images = base.convert_latent_to_image(output_images)
                    latents = base.convert_latent_to_output_latent(output_images)
                    images = pickle_and_encode_b64(images[0])
                    latents = pickle_and_encode_b64(latents)
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
    __initialize_environment()
    __run_host()
