# verify_stack.py — run once after setup to confirm everything imports cleanly

import grid2op
from lightsim2grid import LightSimBackend
import torch
import torch_geometric
import networkx as nx
from langchain_community.llms import Ollama
from pydantic import BaseModel
import wandb

print(f"grid2op:          {grid2op.__version__}")
print(f"torch:            {torch.__version__}")
print(f"torch_geometric:  {torch_geometric.__version__}")
print(f"networkx:         {nx.__version__}")
if torch.cuda.is_available():
    print(f"CUDA version:      {torch.version.cuda}")
    print(f"CUDA available:   {torch.cuda.is_available()}") # False on local, True on Research PC
elif torch.xpu.is_available():
    print(f"XPU available:    {torch.xpu.is_available()}") # True on local, False on Research PC

# Quick Grid2Op smoke test
env = grid2op.make("l2rpn_neurips_2020_track1_small", backend=LightSimBackend())
obs = env.reset()
print(f"\nGrid2Op OK — obs.rho shape: {obs.rho.shape}")  # → (59,)

# Quick Ollama smoke test
llm = Ollama(model="qwen2.5:7b")
resp = llm.invoke("Reply with only the word: OK")
print(f"Ollama OK — response: {resp.strip()}")

print("\n✓ Stack verified. Ready to build.")