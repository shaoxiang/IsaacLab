# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different types of markers.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/markers.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different types of markers.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.math import quat_from_angle_axis
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
import numpy as np

for i in range(100):
    # print(f"/World/Cube{i}", self._terrain.env_origins[index])
    prim_utils.create_prim(f"/World/Balls/tb_{i}", "Xform", translation= np.random.uniform(low=-10.0, high=10.0, size=(3)))
    
tennis_ball = RigidObjectCfg(
    prim_path="/World/Balls/tb_.*/tb",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"D:/workspace/tennis/tennis_3d/tb1.usdc",
        # activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0567),
        # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
)

def main():
    """Main function."""
    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view([0.0, 18.0, 12.0], [0.0, 3.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Origin5/Table", cfg, translation=(0.55, 0.0, 1.05))

    cube_object = RigidObject(cfg=tennis_ball)

    # Play the simulator
    sim.reset()

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
