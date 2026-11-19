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
parallel_config = None


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
    global parallel_config, dist
    parallel_config = None
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
    global parallel_config, enable_cache_dit_runonce
    # Params:
    # deep_cache
    # deep_cache_interval
    # deep_cache_id
    # cache_dit
    config = data.get("backend")
    assert config is not None, "Configuration must be provided"

    with torch.no_grad():
        # tensor parallel
        tensor_parallel = config.get("tensor_parallel") not in [None, False]
        tp_degree = config.get("tp_degree")

        # context parallel
        context_parallel = config.get("context_parallel") not in [None, False]
        ulysses_degree = config.get("ulysses_degree")
        ring_degree = config.get("ring_degree")
        ulysses_anything = config.get("ulysses_anything") not in [None, False]
        ring_anything = config.get("ring_anything") not in [None, False]

        requested_world_size = tp_degree * ulysses_degree * ring_degree
        assert requested_world_size == dist.get_world_size(), (
            "(tp * ulysses * ring) needs to be equal to world size"
        )
        mesh = init_device_mesh(
            "cuda",
            (tp_degree, ring_degree, ulysses_degree),
            mesh_dim_names=("tp", "ring", "ulysses"),
        )
        tp_mesh, cp_mesh = mesh["tp"], mesh["ring", "ulysses"]
        base.log(
            0,
            f"ℹ️ Device mesh created with size {dist.get_world_size()} (T:{tp_degree}|U:{ulysses_degree}|R:{ring_degree})",
        )

        def move_pipe_components():
            components_to_move = [k for k, v in base.pipe.components.items()]
            components_to_move = [
                k
                for k in components_to_move
                if k not in base.offloaded_modules
            ]
            component_names_to_check = (
                Common.ENCODER_MODULE_NAMES
                + Common.VAE_MODULE_NAMES
                + Common.MISC_MODULE_NAMES
            )
            for k in component_names_to_check:
                if (
                    data.get(k) is not None
                    and data[k].get("device_map") is not None
                ):
                    components_to_move = [
                        k2 for k2 in components_to_move if k2 != k
                    ]

            for k in components_to_move:
                try:
                    component = getattr(base.pipe, k)
                    if component is not None:
                        try:
                            setattr(base.pipe, k, component.to("cuda"))
                            base.log(0, f"➡️ Moved component {k} to cuda")
                        except Exception:
                            pass
                except Exception:
                    pass
            return

        if tensor_parallel is True and context_parallel is True:
            base.log(0, "ℹ️ Using both context and tensor parallelism")

            # setup pipe
            result = base.setup_pipeline(data, backend_name="parallel")
            target = (
                base.pipe.transformer
                if base.is_transformer_model_type
                else base.pipe.unet
            )

            # apply tpconfig to transformer
            tp_config = diffusers.TensorParallelConfig(
                tp_degree=tp_degree, mesh=tp_mesh
            )
            if (
                base.adapter_names is not None
                and base.loras_are_fused is False
            ):
                base.log(
                    0,
                    "⏳ Reloading transformer as LoRAs are loaded on a tensor parallel configuration",
                )
                base.fuse_pipeline_loras()
            target.enable_parallelism(config=tp_config)

            # dynamically move pipe components
            move_pipe_components()

            # setup cpconfig
            cp_config = diffusers.ContextParallelConfig(
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                ulysses_anything=ulysses_anything,
                ring_anything=ring_anything,
                mesh=cp_mesh,
            )
            cp_config.setup(
                base.local_rank,
                dist.get_world_size(),
                torch.device(f"cuda:{base.local_rank}"),
                mesh=cp_mesh,
            )
            cp_plan = getattr(target, "_cp_plan", None)
            assert cp_plan is not None, (
                "No context parallel plan found in transformer"
            )

            # apply cpconfig to transformer
            diffusers.hooks.apply_context_parallel(target, cp_config, cp_plan)
        elif tensor_parallel is True:
            base.log(0, "ℹ️ Using tensor parallelism")

            # setup pipe
            result = base.setup_pipeline(data, backend_name="parallel")
            target = (
                base.pipe.transformer
                if base.is_transformer_model_type
                else base.pipe.unet
            )

            # apply tpconfig to transformer
            tp_config = diffusers.TensorParallelConfig(
                tp_degree=tp_degree, mesh=tp_mesh
            )

            if (
                base.adapter_names is not None
                and base.loras_are_fused is False
            ):
                base.log(
                    0,
                    "⏳ Reloading transformer as LoRAs are loaded on a tensor parallel configuration",
                )
                base.fuse_pipeline_loras()
            if getattr(target, "_tp_plan", None) is None:
                custom_tp_plan = get_custom_tp_plans(base.pipeline_name)
                assert custom_tp_plan is not None, (
                    "No tp_plan found in the transformer, and no custom plans are available"
                )
                setattr(target, "_tp_plan", custom_tp_plan)
            target.enable_parallelism(config=tp_config)

            # dynamically move pipe components
            move_pipe_components()
        elif context_parallel is True:
            base.log(0, "ℹ️ Using context parallelism")

            # setup pipe
            result = base.setup_pipeline(data, backend_name="parallel")
            target = (
                base.pipe.transformer
                if base.is_transformer_model_type
                else base.pipe.unet
            )

            # dynamically move pipe components
            move_pipe_components()

            # apply cpconfig to transformer
            cp_config = diffusers.ContextParallelConfig(
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                ulysses_anything=ulysses_anything,
                ring_anything=ring_anything,
                mesh=cp_mesh,
            )
            cp_plan = getattr(target, "_cp_plan", None)
            assert cp_plan is not None, (
                "No context parallel plan found in transformer"
            )
            target.enable_parallelism(config=cp_config, cp_plan=cp_plan)

        dist.barrier()
        if result[1] == 200:
            parallel_config = config
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
                and parallel_config.get("deepcache_config") is not None
            )
            enable_cache_dit = (
                base.can_use_cachedit is True
                and parallel_config.get("cache_dit_config") is not None
            )
            deepcache_config = None
            cache_dit_config = None
            if enable_deepcache:
                deepcache_config = parallel_config.get("deepcache_config")
            if enable_cache_dit:
                cache_dit_config = parallel_config.get("cache_dit_config")

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
    __initialize_distributed_environment()
    __run_host()
