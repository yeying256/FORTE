import os
import yaml
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CURRENT_DIR / "config/config.yaml"
def load_config(config_path=None):

    """
    加载 YAML 配置文件
    """
    if config_path is None:
        config_path = CONFIG_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config