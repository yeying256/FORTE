# moca_vlm

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
VLM 质量估计 ROS 包，提供质量请求/结果 topic，可调用外部 API 或读取离线 JSON。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg moca_vlm
source devel/setup.bash
```

### 系统 Python + 本地 vendor 依赖
实体机器人上不需要 conda，可以把 VLM 在线 API 依赖安装到工程目录：

```bash
cd <your_catkin_workspace>
PYTHON_BIN=python3 src/moca_vlm/scripts/install_system_python_vendor.sh
```

安装后依赖会放到 `src/moca_vlm/vendor/python`，`mass_estimator_api_helper.py`
会在系统 `python3` 里优先加载这个目录。默认配置已经指向：

```yaml
api_python_executable: python3
vendor_python_path: "$(find moca_vlm)/vendor/python"
```

默认安装 OpenAI/Ollama 兼容接口所需依赖。如果要使用 Gemini，并且机器人
Python 版本支持 `google-genai`，可以额外执行：

```bash
PYTHON_BIN=python3 REQUIREMENTS_FILE=src/moca_vlm/requirements-vlm-gemini.txt \
  src/moca_vlm/scripts/install_system_python_vendor.sh
```

如果机器人无法联网，可以先在同系统/同架构机器上准备 wheelhouse，再拷贝到机器人：

```bash
python3 -m pip download -r src/moca_vlm/requirements-vlm.txt -d /tmp/moca_vlm_wheelhouse
WHEELHOUSE=/tmp/moca_vlm_wheelhouse src/moca_vlm/scripts/install_system_python_vendor.sh
```

### Launch 文件

- `launch/mass_estimator.launch`: 启动 VLM 质量估计节点，可选择 API 模式或离线 JSON 模式。
  运行示例：`roslaunch moca_vlm mass_estimator.launch`
  常用参数：`config_file:=$(find moca_vlm)/config/mass_estimator_params.yaml`
  主要节点：`moca_vlm/mass_estimator_node.py/name=moca_vlm_mass_estimator`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch moca_vlm <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
