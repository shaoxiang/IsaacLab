## tutorials
python scripts/tutorials/00_sim/spawn_prims.py
python scripts/tutorials/00_sim/create_empty.py
python scripts/tutorials/00_sim/launch_app.py

python scripts/tutorials/03_envs/create_cube_base_env.py --num_envs 8
python scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint assets/Policies/Isaac-Velocity-Rough-H1-v0/policy.pt

python scripts/tutorials/01_assets/run_rigid_object.py
python scripts/tutorials/01_assets/run_articulation.py

python scripts/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras
python scripts/tutorials/04_sensors/run_ray_caster.py
python scripts/tutorials/04_sensors/run_ray_caster_camera.py

## demos
python scripts/demos/arms.py
python scripts/demos/multi_asset.py
python scripts/tutorials/00_sim/spawn_prims.py

python scripts/tutorials/01_assets/run_deformable_object.py
python scripts/tutorials/01_assets/run_rigid_object.py
python scripts/tutorials/01_assets/run_tennis_ball.py
python scripts/demos/multi_asset.py
python scripts/demos/multi_asset_more.py

## benchmark
python scripts/benchmarks/benchmark_rlgames.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras
python scripts/benchmarks/benchmark_rlgames.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras --num_envs 128

## Tests
python source/isaaclab/test/assets/check_ridgeback_franka.py

## Manage-Based Ant
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --num_envs 32

## Manage-Based Humanoid
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Humanoid-v0 --num_envs 32

## Manage-Based navigation
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Navigation-Flat-Anymal-C-v0 --num_envs 32
Isaac-Navigation-Flat-Anymal-C-Play-v0

## Manage-Based locomotion velocity
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Spot-v0 --num_envs 32

## Manage-Based Flat-H1
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-H1-v0 --headless --video

## Manage-Based Rough-H1

Observation Manager: <ObservationManager> contains 1 groups.
+----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (256,)) |
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_lin_vel                   |     (3,)    |
|     1     | base_ang_vel                   |     (3,)    |
|     2     | projected_gravity              |     (3,)    |
|     3     | velocity_commands              |     (3,)    |
|     4     | joint_pos                      |    (19,)    |
|     5     | joint_vel                      |    (19,)    |
|     6     | actions                        |    (19,)    |
|     7     | height_scan                    |    (187,)   |
+-----------+--------------------------------+-------------+

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0 --headless --video
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-H1-Play-v0

## quadcopter env set
python scripts/my_test/quadcopter_camera.py

python scripts/demos/quadcopter.py

python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Franka-Cabinet-Direct-v0

### Cartpole

python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-Direct-v0
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras --video --num_envs 512
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-Depth-Camera-Direct-v0 --headless --enable_cameras --num_envs 32

python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-RGB-ResNet18-v0 --headless --enable_cameras --video
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-RGB-TheiaTiny-v0 --headless --enable_cameras --video

python scripts/reinforcement_learning/rl_games/play.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --checkpoint logs/rl_games/cartpole_camera_direct/2024-06-06_11-01-07/nn/cartpole_camera_direct.pth

python -m tensorboard.main --logdir logs/rl_games/cartpole_camera_direct

### Jetbot

python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Jetbot-Direct-v0

### 无人机
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Quadcopter-Direct-v0
python scripts/reinforcement_learning/rl_games/play.py --task=Isaac-Quadcopter-Direct-v0
python -m tensorboard.main --logdir logs/rl_games/quadcopter_direct/2024-06-05_19-17-12
python -m tensorboard.main --logdir logs/rsl_rl/quadcopter_direct/2024-06-20_15-53-02

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-Direct-v0 --headless

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Quadcopter-Direct-v0 --checkpoint=model_11600.pt --num_envs=2
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Quadcopter-Direct-play-v0 --headless --checkpoint model_11600.pt

python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Quadcopter-Direct-v0 --headless

### UAV
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-UAV-Direct-v0 --num_envs 8
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-UAV-PTZ-Direct-v0 --num_envs 8
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-UAV-Control-Direct-v0 --num_envs 8192 --headless
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-UAV-Control-Direct-v0 --num_envs 8192 --headless --resume True --load_run 2024-12-12_13-27-40 --checkpoint model_5000.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-UAV-Control-Direct-v0 --num_envs 8 --load_run policy --checkpoint model_9000.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 2 --headless --enable_cameras --checkpoint model_9000.pt

python scripts/reinforcement_learning/rsl_rl/play_jit.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 2 --headless --enable_cameras --checkpoint policy.pt

python scripts/reinforcement_learning/rsl_rl/play_onnx.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 2 --headless --enable_cameras --checkpoint policy.onnx

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-UAV-Fly-v0 --num_envs 4 --enable_cameras --headless --livestream 1

### PTZ-UAV
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 256 --enable_cameras --headless --video

python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 scripts/reinforcement_learning/skrl/train.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 2048 --enable_cameras --headless --distributed

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 256 --enable_cameras --headless --resume True
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 8 --enable_cameras

### Kaya
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Kaya-Direct-v0 --num_envs 4096 --headless
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Kaya-Tennis-v0 --num_envs 4096 --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Kaya-Tennis-Play-v0 --num_envs 2
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Kaya-VA-v0 --num_envs 256 --enable_cameras --headless
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Kaya-VA-v0 --num_envs 128 --enable_cameras --video --video_length 1000 --video_interval 50000 --headless

python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Kaya-Tennis-v1 --num_envs 2048 --algorithm TD3 --headless
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Kaya-Tennis-v1 --num_envs 1024 --video --video_length 1000 --video_interval 200 --algorithm TD3 --headless
python -m tensorboard.main --logdir logs/skrl/kaya_tennis_td3/2025-02-12_17-06-06_td3_torch

### Agilex Robotics
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-ScoutMini-Direct-v0 --num_envs 4096 --headless

python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-ScoutMini-AV-v0 --num_envs 4096 --headless

#### Multi GPU Train
python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Kaya-VA-v0 --num_envs 4096 --enable_cameras --headless --distributed

python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Kaya-VA-v0 --num_envs 4096 --enable_cameras --video --video_length 1000 --video_interval 50000 --headless --distributed

python -m tensorboard.main --logdir logs/skrl/kaya_va_direct/2025-01-22_16-27-36_ppo_torch
python -m tensorboard.main --logdir logs/rsl_rl/kaya_tennis/2025-01-22_22-03-52

### Duck
#### Direct Duck
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Duck-Direct-v0 --num_envs 4096 --headless
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Duck-Direct-v0 --num_envs 4096 --headless --resume True
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Duck-Direct-v0 --num_envs 4 --headless --livestream 1

#### Managed Duck
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Duck-v0 --num_envs 4096 --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Duck-Play-v0
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Duck-Play-v0 --headless --livestream 1

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-BDX-v0 --num_envs 4096 --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-BDX-Play-v0
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-BDX-Play-v0 --headless --livestream 1
python -m tensorboard.main --logdir logs/rsl_rl/bdx_flat/2025-01-17_20-30-01
python source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_viewer.py --file humanoid_walk.npz --render-scene True

### teddy_bear
python scripts/environments/state_machine/lift_teddy_bear.py --num_envs 4

### Manage-Based 无人机
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-v0 --num_envs 32

#### 平滑控制版本
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-Smooth-v0 --headless

#### 带IMU
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-IMU-v0 --headless

#### 带摄像头
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Quadcopter-RGB-Camera-Direct-v0 --headless --enable_cameras --video
python -m tensorboard.main --logdir logs/rl_games/quadcopter_direct_camera/2024-06-09_07-56-25

python scripts/reinforcement_learning/rl_games/play.py --task=Isaac-Quadcopter-RGB-Camera-Direct-v0 --checkpoint logs\rl_games\quadcopter_direct_camera\2024-06-12_09-46-08\nn\quadcopter_direct_camera.pth

python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Quadcopter-Direct-v0

#### 视觉避障 1.0
python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Quadcopter-RGB-Camera-Direct-v0 --headless --enable_cameras
python -m tensorboard.main --logdir logs/sb3/Isaac-Quadcopter-Direct-v0/2024-06-24_11-45-43

#### 视觉避障 2.0
python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Quadcopter-Vision-OA-v0 --enable_cameras
python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Quadcopter-Vision-OA-v0 --headless --enable_cameras --video

python -m tensorboard.main --logdir logs/sb3/Isaac-Quadcopter-Direct-v0/2024-06-24_11-45-43
python -m tensorboard.main --logdir logs/sb3/Isaac-Quadcopter-RGB-Camera-Direct-v0/2024-06-24_16-35-21

#### 视觉避障 3.0
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-Vision-Depth-v0 --headless --enable_cameras

#### 人形机器人

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Humanoid-Direct-v0 

### SLAM
python scripts/standalone/lidar_imu_pub/main.py --checkpoint assets/Policies/Isaac-Velocity-Rough-H1-v0/policy.pt

### AMP
#### Humanoid-AMP
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Humanoid-AMP-Dance-Direct-v0 --headless --algorithm AMP
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Humanoid-AMP-Run-Direct-v0 --headless --algorithm AMP
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Humanoid-AMP-Walk-Direct-v0 --headless --algorithm AMP

python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Humanoid-AMP-Dance-Direct-v0 --headless --algorithm AMP --video --video_length 200 --video_interval 10000

python scripts/reinforcement_learning/skrl/play.py --task=Isaac-Humanoid-AMP-Dance-Direct-v0 --num_envs 32 --algorithm AMP --real-time
python scripts/reinforcement_learning/skrl/play.py --task=Isaac-Humanoid-AMP-Run-Direct-v0 --num_envs 32 --algorithm AMP --real-time
python scripts/reinforcement_learning/skrl/play.py --task=Isaac-Humanoid-AMP-Walk-Direct-v0 --num_envs 32 --algorithm AMP --real-time

python -m tensorboard.main --logdir logs/skrl/humanoid_amp_walk/2025-02-13_08-01-20_amp_torch

### Test
#### Test contact_sensor
python source/isaaclab/test/sensors/test_contact_sensor.py

#### Test Imu sensor
python source/isaaclab/test/sensors/test_imu.py
python source/isaaclab/test/sensors/check_imu_sensor.py
python scripts/tutorials/06_ros/imu_sensor_to_ros.py

#### Test Camera

python scripts/tutorials/04_sensors/add_more_sensors_on_robot.py --enable_cameras

### ROS
python scripts/tutorials/06_ros/tiled_camera.py --enable_cameras
python scripts/tutorials/06_ros/tiled_camera_threading.py --enable_cameras
python scripts/tutorials/06_ros/camera.py --enable_cameras

#### lifelong_slam
python .\source\standalone\lidar_slam\play.py --task=Isaac-Quadcopter-Direct-Lidar-v0 --load_run 2024-10-22_13-08-21 --checkpoint model_1000.pt

### 大模型机器人

python -m tensorboard.main --logdir logs/eureka/Isaac-Cartpole-Direct-v0/2024-12-05_11-22-51

## 修改意见
_isaac_sim\exts\omni.isaac.sensor\omni\isaac\sensor\scripts\menu.py
data = json.load(open(d + "/" + file))

with open(d + "/" + file, 'r', encoding='utf-8') as f:
    data = json.load(f)

isaac sim 4.2 版本后：
data = json.load(open(os.path.join(d, file)))

改为：
with open(os.path.join(d, file), encoding='utf-8') as f:
    data = json.load(f)

如遇到：
File "/root/anaconda3/envs/sx_isaaclab/lib/python3.10/site-packages/rsl_rl/utils/utils.py", line 83, in store_code_state
    f.write(content)
UnicodeEncodeError: 'ascii' codec can't encode characters in position 4947-4949: ordinal not in range(128)

将这个代码
with open(diff_file_name, "x") as f:
    content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
    f.write(content)
修改为：
with open(diff_file_name, "x", encoding='utf-8') as f:
    content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
    f.write(content)

### 编队单点导航
python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Quadcopter-Form-v0 --headless
python scripts/reinforcement_learning/sb3/train_form_td3.py --task=Isaac-Quadcopter-Form-v0 --headless
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Quadcopter-Form-v0 --headless
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-Form-v0 --headless
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Quadcopter-Form-v0 --headless --max_iterations 1000000

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Quadcopter-Form-Play-v0 --checkpoint model_10000.pt

python -m tensorboard.main --logdir logs/skrl/quadcopter_form_direct/2024-07-25_09-13-04
python -m tensorboard.main --logdir logs/rsl_rl/quadcopter_form_direct/2024-07-31_20-04-06

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-Form-v0 --headless --resume True --load_run 2024-07-31_20-04-06 --checkpoint model_8600.pt

### 编队飞路径
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-Form-Path-v0 --headless

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Quadcopter-Form-Path-v0 --headless --resume True --load_run 2024-07-31_20-04-06 --checkpoint model_8600.pt

### 更新Isaac Lab
.\isaaclab.bat --install

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple einops

单独升级 rsl_rl
pip install git+https://github.com/leggedrobotics/rsl_rl.git

单独升级 skrl
./isaaclab.bat -p -m pip show skrl
./isaaclab.bat -p -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade skrl
./isaaclab.sh -p -m pip install git+https://github.com/Toni-SM/skrl.git@develop
./isaaclab.sh -p -m pip install git+https://gitee.com/shaoxiang/skrl.git@develop
.\isaaclab.bat -p -m pip install git+https://github.com/Toni-SM/skrl.git@develop
.\isaaclab.bat -p -m pip install git+https://gitee.com/shaoxiang/skrl.git@develop

单独升级 stable_baselines3
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple stable-baselines3
pip install git+https://github.com/DLR-RM/stable-baselines3

单独升级 robomimic
sudo apt install cmake build-essential
./isaaclab.sh -p -m pip install git+https://gitee.com/shaoxiang/robomimic

### asserts error
#### 将目录替换成你自己的下载的即可
.\isaac-sim.bat --/persistent/isaac/asset_root/default="D:\omniverse\Downloads\Assets\Isaac\4.2"

#### 如果pytorch出了问题
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
或者：
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 -f https://download.pytorch.org/whl/torch

### YOLO
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics --no-deps
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime-gpu --no-deps

## Tips
1、添加可变形物体，核心要注意 replicate_physics 为 False
2、contact sensor 使用注意，
如果使用 filter_prim_paths_expr 只保留与某些特定的物体之间碰撞，那么读取碰撞数值时应该使用force_matrix_w。 
net_forces_w 报告总的净法向力，过滤后的力位于单独的属性 force_matrix_w 
3、特权信息
observations = {"policy": obs, "critic": states}

## 人形机器人动捕数据播放
git clone https://hf-mirror.com/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset

# 将目录替换成你自己的下载的即可
.\isaac-sim.bat --/persistent/isaac/asset_root/default="D:\omniverse\Downloads\Assets\Isaac\4.2"
./isaac-sim.sh --/persistent/isaac/asset_root/default="/home/ai/omniverse/Downloads/Isaac/4.2"

./isaac-sim.sh --/persistent/isaac/asset_root/default="/home/dell/omniverse/Downloads/Assets/Isaac/4.2"

### isaac lab 安装相关

* 软链接到 _isaac_sim
```
ln -s /home/dell/.local/share/ov/pkg/isaac-sim-4.2.0 _isaac_sim
ln -s /home/ai/omniverse/pkg/isaac-sim-4.5.0 _isaac_sim
```

```
cmd需要管理员权限
mklink /D _isaac_sim D:\OMNIVERSE\pkg\isaac-sim-4.5.0
```

* 创建 conda env
```
.\isaaclab.bat --conda my_labv2
conda activate my_labv2
```

* 安装 isaac lab
```
.\isaaclab.bat --install
```

#### tips
* conda 换源

```bash
channels:
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r

channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

* conda remove env
```
conda env remove --name <environment_name>
```

* ROS 配置
```

```