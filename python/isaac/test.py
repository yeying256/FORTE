from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
# from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.prims import Articulation

# 创建 world
world = World()
world.scene.add_default_ground_plane()

# 获取 assets 路径
# assets_root = get_assets_root_path()

# 添加 Franka 
asset_path = "/media/wangxiao/iit_work_disk/IIT_ws/Moca_polytope/Moca_Isaac/Moca_fix_wheels.usd"

add_reference_to_stage(
    usd_path=asset_path,
    prim_path="/World/MOCA"
)



moca = Articulation(prim_paths_expr="/World/MOCA")  # create an articulation object






# 初始化
world.reset()

moca.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
moca.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])

# 仿真循环
while simulation_app.is_running():
    world.step(render=True)




simulation_app.close()