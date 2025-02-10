# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the deformable object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_deformable_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Deformable Object
    # Tennis Object
    cfg_tennis = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Tennis_Ball/tennis_ball_low.usd",
            scale=(0.0385, 0.0385, 0.0385),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
        debug_vis=True,
    )
    tennis_object = DeformableObject(cfg=cfg_tennis)
    
    # cfg_tennis.func("/World/Objects/Tennis", cfg_tennis, translation=(0.0, 0.0, 2.0))

    # return the scene information
    scene_entities = {"tennis_object": tennis_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    tennis_object = entities["tennis_object"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target = tennis_object.data.nodal_kinematic_target.clone()

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset the nodal state of the object
            nodal_state = tennis_object.data.default_nodal_state_w.clone()
            # apply random pose to the object
            pos_w = torch.rand(tennis_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(tennis_object.num_instances, device=sim.device)
            nodal_state[..., :3] = tennis_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            nodal_state[..., 3:] = torch.zeros_like(nodal_state[..., 3:]).uniform_(-100.0, 100.0)
            # write nodal state to simulation
            tennis_object.write_nodal_state_to_sim(nodal_state)

            # Write the nodal state to the kinematic target and free all vertices
            # nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            # nodal_kinematic_target[..., 3] = 1.0
            # tennis_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # reset buffers
            tennis_object.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

        # update the kinematic target for cubes at index 0 and 3
        # we slightly move the cube in the z-direction by picking the vertex at index 0
        # nodal_kinematic_target[[0, 3], 0, 2] += 0.001
        # set vertex at index 0 to be kinematically constrained
        # 0: constrained, 1: free
        # nodal_kinematic_target[[0, 3], 0, 3] = 0.0
        # write kinematic target to simulation
        # cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # collision_element_quat_w = tennis_object.data.collision_element_quat_w
        # collision_element_deform_gradient_w = tennis_object.data.collision_element_deform_gradient_w
        # collision_element_stress_w = tennis_object.data.collision_element_stress_w
        # print("collision_element_quat_w:", collision_element_quat_w)
        # print("collision_element_deform_gradient_w:", collision_element_deform_gradient_w)
        # print("collision_element_stress_w:", collision_element_stress_w)

        root_pos_w = tennis_object.data.root_pos_w
        print("root_pos_w:", root_pos_w)
        
        # write internal data to simulation
        tennis_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        tennis_object.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {tennis_object.data.root_pos_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
