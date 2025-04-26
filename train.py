import time

import mujoco
import mujoco.viewer

from model import Model

from robot import Robot

import numpy as np
import torch

import os

MODEL_PATH = 'model'

def getModel(hyperparameters):
  model = Model(
    hyperparameters["inputSize"],
    hyperparameters["outputSize"],
    hyperparameters["learningRate"],
  )

  if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
  return model

def randomizePosition(robot, m):
  x, y = (np.random.rand() - 0.5) * 288, (np.random.rand() - 0.5) * 288
  m.geom('start').pos = (x, y, 0)
  m.geom('end').pos = ((np.random.rand() - 0.5) * 288, (np.random.rand() - 0.5) * 288, 0)

  robot.setPosition(x, y)

def runTrainingLoop(xml, hyperparameters, episodes=10000, save_frequency=100, max_time=5):
  try:
    model = getModel(hyperparameters)

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)

    if not hyperparameters["visualize"]:
      # hide rays
      m.geom('forward').rgba[3] = 0.0
      m.geom('right').rgba[3] = 0.0
      m.geom('backward').rgba[3] = 0.0
      m.geom('left').rgba[3] = 0.0


    if not hyperparameters["visualizeXY"]:
      # hide XY plane indicators
      m.geom('X1').rgba[3] = 0.0
      m.geom('X2').rgba[3] = 0.0

      m.geom('Y1').rgba[3] = 0.0
      m.geom('Y2').rgba[3] = 0.0
      m.geom('Y3').rgba[3] = 0.0


    robot = Robot(m, d, hyperparameters["visualize"])

    with mujoco.viewer.launch_passive(m, d) as viewer:
      for i in range(episodes):
        if i % save_frequency == 0:
          torch.save(model.state_dict(), 'model')

        # Access the camera object
        cam = viewer.cam

        cam.azimuth = 0
        cam.elevation = -90
        cam.distance = 600
        cam.lookat[:] = [0, 0, 0]  # what the camera is looking at

        randomizePosition(robot, m)
        target = m.geom('end').pos

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

          if torch.norm(state[:2]) < 0.5: break
        
        mujoco.mj_resetData(m, d)
      
        model.train(records)

  except KeyboardInterrupt:
    torch.save(model.state_dict(), 'model')
  torch.save(model.state_dict(), 'model')

if __name__ == "__main__":
  hyperparameters = {
    "inputSize": 8,
    "outputSize": 2,
    "learningRate": 1.0,
    "visualize": False,
    "visualizeXY": False
  }

  runTrainingLoop("model.xml", hyperparameters)