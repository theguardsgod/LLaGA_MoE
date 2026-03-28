# MoE-LLaGA training with FlashAttn monkey patch for memory efficiency.

import sys
sys.path.append(".")
sys.path.append("./utils")
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from train_moe import _train

if __name__ == "__main__":
    _train()
