# from isaacsim import SimulationApp
import sys, os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

from isaac.isaac import Moca

def main():
    # 机器人
    Moco_ = Moca("/media/wangxiao/iit_work_disk/IIT_ws/Moca_polytope/python/config/config.yaml")
    Moco_.starting()
    Moco_.update()
    Moco_.end()

    pass


if __name__ == "__main__":
    main()