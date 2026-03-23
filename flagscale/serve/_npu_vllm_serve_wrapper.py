# Wrapper for `vllm serve` on Ascend NPU.
# Forces NPUPlatform registration before vllm's own import-time platform
# detection, which is needed because vllm_ascend.register() returns a class
# path string but the installed vllm version does not consume the return value.

import torch_npu  # noqa: F401 - must import before vllm to init CANN
from vllm_ascend.platform import NPUPlatform

import vllm.platforms as _vllm_platforms

if not isinstance(_vllm_platforms.current_platform, NPUPlatform):
    _vllm_platforms._current_platform = NPUPlatform()

import sys

sys.argv[0] = "vllm"

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
