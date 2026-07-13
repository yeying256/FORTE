# vscode不能补全的问题

## why
Because the vscode can't find the python code of isaac path, even you set the explantion of python path (${isaac_path}/kit/python/bin/python3).

## how
Follow the instruction "https://docs.isaacsim.omniverse.nvidia.com/4.5.0/development_tools/vscode.html" to construct the vscode setting.json.

Copy the path from setting.json to the convert_setting.py. And run it.
You can run the convert_setting.py to convert the revrent search path of setting.json to the absolut path vscode setting.json. 

