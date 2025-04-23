import time

import mujoco
import mujoco.viewer

from model import Model

from robot import Robot

import numpy as np
import torch

import os

MODEL_PATH = 'model'

def getModel():
  model = Model(6, 2)
  if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
  return model

def runTrainingLoop(xml, model, hyperparams, episodes=500, max_time=5):
  try:
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)

    target = m.geom('end').pos
    robot = Robot(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
      for _ in range(episodes):
        # Access the camera object
        cam = viewer.cam

        cam.azimuth = 223
        cam.elevation = -40
        cam.distance = 600
        cam.lookat[:] = [0, 70, 3.2]  # what the camera is looking at

        start = time.time()

        records = []

        while viewer.is_running() and time.time() - start < max_time:
          step_start = time.time()

          # mj_step can be replaced with code that also evaluates
          # a policy and applies a control signal before stepping the physics.
          mujoco.mj_step(m, d)

          deltaPosition = [(target[0] - robot.getPosition()[0]) / 144, (target[1] - robot.getPosition()[1]) / 144]
          # deltaPosition = [0, 0]
          state = robot.getState(deltaPosition)
          
          out = model.forward(state)
          out = torch.clamp(out, -100, 100)
          records.append((state, out, model.getLoss(state, out)))

          dx, dy = out
          robot.setVelocity(dx, dy, 0)
          # print(robot.getDistances())

          mujoco.mj_forward(m, d)  # Update simulation with new state

          # Pick up changes to the physics state, apply perturbations, update options from GUI.
          viewer.sync()

          if robot.hasCollision():
            # time.sleep(5)
            break

          # Rudimentary time keeping, will drift relative to wall clock.
          time_until_next_step = m.opt.timestep - (time.time() - step_start)
          if time_until_next_step > 0:
            time.sleep(time_until_next_step)

          if torch.norm(state[:2]) < 0.5 / 144.0: break
        
        mujoco.mj_resetData(m, d)
      
        model.train(records)

  except KeyboardInterrupt:
    torch.save(model.state_dict(), 'model')
  torch.save(model.state_dict(), 'model')

if __name__ == "__main__":
  runTrainingLoop("model.xml", getModel(), {})