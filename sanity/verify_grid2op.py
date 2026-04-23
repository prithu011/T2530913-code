import grid2op
from lightsim2grid import LightSimBackend

# NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY]
env = grid2op.make("l2rpn_neurips_2020_track1_small", backend=LightSimBackend())
print(f"Buses: {env.n_sub}, Lines: {env.n_line}")  # → Buses: 36, Lines: 59