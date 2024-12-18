# Taken and modified from https://github.com/huggingface/trl
# Copyright 2024 The AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file is copied from https://github.com/OpenRLHF/OpenRLHF"""
import socket
from transformers import (
    AutoModelForCausalLM,
)
import ray
import torch
import torch.distributed
from open_instruct.vllm_utils2 import create_vllm_engines, init_process_group


if __name__ == "__main__":
    vllm_tensor_parallel_size = 2
    vllm_num_engines = 1
    vllm_sync_backend = "nccl"
    model_name_or_path = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    model_name_or_path2 = "allenai/Llama-3.1-Tulu-3-8B"
    # llm = LLMRayActor.remote("meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2)
    # output = ray.get(llm.generate.remote("San Franciso is a"))
    # print(f"output: {output}")
    
    vllm_engines = create_vllm_engines(
        vllm_num_engines,
        vllm_tensor_parallel_size,
        model_name_or_path,
        None,
        1,
        False,
        4096,
    )

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    vllm_num_engines, vllm_tensor_parallel_size = (
        vllm_num_engines,
        vllm_tensor_parallel_size,
    )
    world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
    backend = vllm_sync_backend
    # https://github.com/OpenRLHF/OpenRLHF/issues/313
    # if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
    #     backend = "gloo"
    #     print(
    #         "Warning: using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
    #     )
    refs = [
        engine.init_process_group.remote(
            master_address,
            master_port,
            i * vllm_tensor_parallel_size + 1,
            world_size,
            "openrlhf",
            backend=backend,
        )
        for i, engine in enumerate(vllm_engines)
    ]
    model_update_group = init_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name="openrlhf",
    )
    ray.get(refs)
    torch.set_default_device("cuda:7")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path2, torch_dtype=torch.bfloat16)
    model = model.to("cuda:7")
    def broadcast_to_vllm():
        # avoid OOM
        torch.cuda.empty_cache()
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []
        for name, param in model.named_parameters():
            count += 1
            shape = param.shape
            refs = [
                engine.update_weight.remote(
                    name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                )
                for engine in vllm_engines
            ]
            refss.extend(refs)
            torch.distributed.broadcast(param.data, 0, group=model_update_group)
        ray.get(refss)

    broadcast_to_vllm()
    print("broadcasted model to vllm")
