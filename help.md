## tutorials
python source/standalone/tutorials/00_sim/spawn_prims.py
python source/standalone/tutorials/00_sim/create_empty.py
python source/standalone/tutorials/00_sim/launch_app.py

python source/standalone/tutorials/03_envs/create_cube_base_env.py --num_envs 8

python source/standalone/tutorials/01_assets/run_rigid_object.py
python source/standalone/tutorials/01_assets/run_articulation.py

python source/standalone/tutorials/04_sensors/add_sensors_on_robot.py

python source/standalone/tutorials/04_sensors/add_sensors_on_robot.py
python source/standalone/tutorials/04_sensors/run_ray_caster.py
python source/standalone/tutorials/04_sensors/run_ray_caster_camera.py

## demos
python source/standalone/demos/arms.py
python source/standalone/demos/multi_asset.py
python source/standalone/tutorials/00_sim/spawn_prims.py

python source/standalone/tutorials/01_assets/run_deformable_object.py
python source/standalone/tutorials/01_assets/run_rigid_object.py
python source/standalone/tutorials/01_assets/run_tennis_ball.py
python source/standalone/demos/multi_asset.py
python source/standalone/demos/multi_asset_more.py

## Tests
python source/extensions/omni.isaac.lab/test/assets/check_ridgeback_franka.py

## Manage-Based Ant
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Ant-v0 --num_envs 32

## Manage-Based Humanoid
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Humanoid-v0 --num_envs 32

## Manage-Based navigation
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Navigation-Flat-Anymal-C-v0 --num_envs 32
Isaac-Navigation-Flat-Anymal-C-Play-v0

## Manage-Based locomotion velocity
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Velocity-Flat-Spot-v0 --num_envs 32

## Manage-Based Flat-H1
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Velocity-Flat-H1-v0 --headless --video

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

python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0 --headless --video
python source/standalone/workflows/rsl_rl/play.py --task=Isaac-Velocity-Rough-H1-Play-v0

## quadcopter env set
python source/standalone/my_test/quadcopter_camera.py

python source/standalone/demos/quadcopter.py

python source/standalone/workflows/rl_games/train.py --task=Isaac-Franka-Cabinet-Direct-v0

### Cartpole

python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-Direct-v0
python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras --video --num_envs 512

python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-RGB-ResNet18-v0 --headless --enable_cameras --video
python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-RGB-TheiaTiny-v0 --headless --enable_cameras --video

python source/standalone/workflows/rl_games/play.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --checkpoint logs/rl_games/cartpole_camera_direct/2024-06-06_11-01-07/nn/cartpole_camera_direct.pth

python -m tensorboard.main --logdir logs/rl_games/cartpole_camera_direct

### Jetbot

python source/standalone/workflows/rl_games/train.py --task=Isaac-Jetbot-Direct-v0

### 无人机
python source/standalone/workflows/rl_games/train.py --task=Isaac-Quadcopter-Direct-v0
python source/standalone/workflows/rl_games/train.py --task=Isaac-Quadcopter-Direct-v0 --checkpoint=

python source/standalone/workflows/rl_games/play.py --task=Isaac-Quadcopter-Direct-v0
python -m tensorboard.main --logdir logs/rl_games/quadcopter_direct/2024-06-05_19-17-12
python -m tensorboard.main --logdir logs/rsl_rl/quadcopter_direct/2024-06-20_15-53-02

python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-Direct-v0 --headless

python source/standalone/workflows/rsl_rl/play.py --task=Isaac-Quadcopter-Direct-v0 --checkpoint=model_11600.pt --num_envs=2
python source/standalone/workflows/rsl_rl/play.py --task=Isaac-Quadcopter-Direct-play-v0 --headless --checkpoint model_11600.pt

python source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-Direct-v0 --headless

### UAV
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-UAV-Direct-v0 --num_envs 8
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-UAV-PTZ-Direct-v0 --num_envs 8
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-UAV-Control-Direct-v0 --num_envs 8
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-UAV-Control-Direct-v0 --num_envs 8192 --headless --resume True --load_run 2024-12-12_13-27-40 --checkpoint model_5000.pt

python source/standalone/workflows/rsl_rl/play.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 2 --headless --enable_cameras --checkpoint model_9000.pt

python source/standalone/workflows/rsl_rl/play_jit.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 2 --headless --enable_cameras --checkpoint policy.pt

python source/standalone/workflows/rsl_rl/play_onnx.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 2 --headless --enable_cameras --checkpoint policy.onnx

### PTZ-UAV
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 256 --enable_cameras --headless --video
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 256 --enable_cameras --headless --resume True
python source/standalone/workflows/rsl_rl/play.py --task=Isaac-PTZ-Control-Direct-v0 --num_envs 8 --enable_cameras

### Kaya
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Kaya-Direct-v0 --num_envs 256 --headless
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Kaya-Tennis-v0 --num_envs 64

### teddy_bear
python source/standalone/environments/state_machine/lift_teddy_bear.py --num_envs 4

### Manage-Based 无人机
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-v0 --num_envs 32

#### 平滑控制版本
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-Smooth-v0 --headless

#### 带IMU
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-IMU-v0 --headless

#### 带摄像头
python source/standalone/workflows/rl_games/train.py --task=Isaac-Quadcopter-RGB-Camera-Direct-v0 --headless --enable_cameras --video
python -m tensorboard.main --logdir logs/rl_games/quadcopter_direct_camera/2024-06-09_07-56-25

python source/standalone/workflows/rl_games/play.py --task=Isaac-Quadcopter-RGB-Camera-Direct-v0 --checkpoint logs\rl_games\quadcopter_direct_camera\2024-06-12_09-46-08\nn\quadcopter_direct_camera.pth

python source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-Direct-v0

#### 视觉避障 1.0
python source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-RGB-Camera-Direct-v0 --headless --enable_cameras
python -m tensorboard.main --logdir logs/sb3/Isaac-Quadcopter-Direct-v0/2024-06-24_11-45-43

#### 视觉避障 2.0
python source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-Vision-OA-v0 --enable_cameras
python source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-Vision-OA-v0 --headless --enable_cameras --video

python -m tensorboard.main --logdir logs/sb3/Isaac-Quadcopter-Direct-v0/2024-06-24_11-45-43
python -m tensorboard.main --logdir logs/sb3/Isaac-Quadcopter-RGB-Camera-Direct-v0/2024-06-24_16-35-21

#### 视觉避障 3.0
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-Vision-Depth-v0 --headless --enable_cameras

#### 人形机器人

python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Humanoid-Direct-v0 

## Test
### Test contact_sensor
python source\extensions\omni.isaac.lab\test\sensors\test_contact_sensor.py

### Test Imu sensor
python source\extensions\omni.isaac.lab\test\sensors\test_imu.py
python source\extensions\omni.isaac.lab\test\sensors\check_imu_sensor.py
python source/standalone/tutorials/06_ros/imu_sensor_to_ros.py

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

### 编队单点导航
python source/standalone/workflows/sb3/train.py --task=Isaac-Quadcopter-Form-v0 --headless
python source/standalone/workflows/sb3/train_form_td3.py --task=Isaac-Quadcopter-Form-v0 --headless
python source/standalone/workflows/rl_games/train.py --task=Isaac-Quadcopter-Form-v0 --headless
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-Form-v0 --headless
python source/standalone/workflows/skrl/train.py --task=Isaac-Quadcopter-Form-v0 --headless --max_iterations 1000000

python source/standalone/workflows/rsl_rl/play.py --task=Isaac-Quadcopter-Form-Play-v0 --checkpoint model_10000.pt

python -m tensorboard.main --logdir logs/skrl/quadcopter_form_direct/2024-07-25_09-13-04
python -m tensorboard.main --logdir logs/rsl_rl/quadcopter_form_direct/2024-07-31_20-04-06

python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-Form-v0 --headless --resume True --load_run 2024-07-31_20-04-06 --checkpoint model_8600.pt

### 编队飞路径
python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-Form-Path-v0 --headless

python source/standalone/workflows/rsl_rl/train.py --task=Isaac-Quadcopter-Form-Path-v0 --headless --resume True --load_run 2024-07-31_20-04-06 --checkpoint model_8600.pt

### helps
./isaaclab.bat -p -m pip show skrl
./isaaclab.bat -p -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade skrl

### 更新Isaac Lab
.\isaaclab.bat --install

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple einops

单独升级 rsl_rl
pip install git+https://github.com/leggedrobotics/rsl_rl.git

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
