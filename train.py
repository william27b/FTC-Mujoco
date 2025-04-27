import time

import mujoco
import mujoco.viewer

import contextlib
import signal

from model import Model

from robot import Robot

import numpy as np
import torch

import os

class TimeoutException(Exception): pass

@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

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

  endX, endY = 0, 0
  while abs(endX) < 47.5 and abs(endY) < 47.5:
    endX, endY = (np.random.rand() - 0.5) * 288, (np.random.rand() - 0.5) * 288

  m.geom('end').pos = (endX, endY, 0)

  robot.setPosition(x, y)

def runTrainingLoop(xml, hyperparameters, episodes=10000, save_frequency=100, max_time=5):
  def launchViewer(m, d, render):
    if render:
      return mujoco.viewer.launch_passive(m, d)
    return contextlib.nullcontext()
  
  try:
    model = getModel(hyperparameters)

    m = mujoco.MjModel.from_xml_path(xml)
    max_steps = max_time / m.opt.timestep
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

    with launchViewer(m, d, hyperparameters["render"]) as viewer:
      if viewer:
        # Access the camera object
        cam = viewer.cam

        cam.azimuth = 0
        cam.elevation = -90
        cam.distance = 600
        cam.lookat[:] = [0, 0, 0]  # what the camera is looking at

      ep_start = time.time()
      for i in range(episodes):
        print(i)
        print("episode_time", time.time() - ep_start)

        ep_start = time.time()
        if i % save_frequency == 0:
          torch.save(model.state_dict(), 'model')

        randomizePosition(robot, m)
        target = m.geom('end').pos

        start = time.time()

        records = []

        ep_steps = 0
        while (not viewer or viewer.is_running()) and ep_steps < max_steps and time.time() - start < max_time:
          ep_steps += 1

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
          if hyperparameters["render"]:
            viewer.sync()

          if robot.hasCollision():
            # time.sleep(5)
            break

          # Rudimentary time keeping, will drift relative to wall clock.
          time_until_next_step = m.opt.timestep - (time.time() - step_start)
          if viewer and time_until_next_step > 0:
            time.sleep(time_until_next_step)

          if torch.norm(state[:2]) < 0.05: break
        
        mujoco.mj_resetData(m, d)

        print(len(records))
        model.train(records)

  except KeyboardInterrupt:
    torch.save(model.state_dict(), 'model')
  torch.save(model.state_dict(), 'model')

if __name__ == "__main__":
  hyperparameters = {
    "inputSize": 8,
    "outputSize": 2,
    "learningRate": 0.01,
    "visualize": False,
    "visualizeXY": True,
    "render": True
  }

  runTrainingLoop("model.xml", hyperparameters, episodes=10_000)

  # try:
  #     with time_limit(60):
  #         runTrainingLoop("model.xml", hyperparameters)
  # except TimeoutException as e:
  #     print("Timed out!")