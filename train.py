import time

import mujoco
import mujoco.viewer

from model import Model

from robot import Robot

import numpy as np
import torch

# def model(current, target, velocity):
  # return ((target[0]-current[0]) * 1.0, (target[1]-current[1]) * 1.0)

def runTrainingLoop(xml, model, hyperparams, episodes=1_000, max_time=5):
  m = mujoco.MjModel.from_xml_path(xml)
  d = mujoco.MjData(m)

  target = m.geom('end').pos
  robot = Robot(m, d)

  with mujoco.viewer.launch_passive(m, d) as viewer:
    for _ in range(episodes):
      start = time.time()

      records = []

      while viewer.is_running() and time.time() - start < max_time:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        deltaPosition = [target[0] - robot.getPosition()[0], target[1] - robot.getPosition()[1]]
        deltaPosition = torch.Tensor(np.array(deltaPosition))
        
        out = model.forward(deltaPosition)
        # print(out)
        out = torch.clamp(out, -20, 20)
        # print(out)
        # print()
        records.append((deltaPosition, out, model.getLoss(deltaPosition, out)))

        dx, dy = out

        # dx, dy = model(robot.getPosition()[:2], target, robot.getVelocity()[:2])
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
      
      mujoco.mj_resetData(m, d)
    
      model.train(records)

if __name__ == "__main__":
  runTrainingLoop("model.xml", Model(2, 2), {})