#!/usr/bin/env bash
# On the pod (from the unpacked package in /root/hrc): CPU torch is enough (ManiCast is tiny); rendering = MuJoCo EGL on the
# pod GPU. The laptop's ur_bullet312 packages without lerobot (lerobot source comes in the package at ./lerobot).
set -euo pipefail
cd /root/hrc
apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libopengl0 >/dev/null
python3 -m pip install -q uv
[ -x /root/venv/bin/python ] || python3 -m uv venv -q -p 3.12 /root/venv
grep -v '^lerobot' workstation/env/ur_bullet312.txt | grep -viE '^(rospkg|catkin|ros)' > /tmp/req.txt
python3 -m uv pip install -q -p /root/venv/bin/python --torch-backend cpu -r /tmp/req.txt
python3 -m uv pip install -q -p /root/venv/bin/python mujoco==3.3.7 pyopengl==3.1.10
# the lerobot source from the package (git 64b2317, same as the laptop)
echo "/root/hrc/lerobot_src" > "$(/root/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/lerobot_src.pth"
mkdir -p /root/hrc/lerobot_src && rm -rf /root/hrc/lerobot_src/lerobot && mv /root/hrc/lerobot /root/hrc/lerobot_src/
cd /root/hrc && MUJOCO_GL=egl INTENT_PYTHON=/root/venv/bin/python ./run.sh -c "import mujoco, lerobot, torch, av; from lerobot.datasets.lerobot_dataset import LeRobotDataset; r=mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>'),64,64); print('mujoco', mujoco.__version__, 'torch', torch.__version__, 'egl ok')"
echo ENV_READY
