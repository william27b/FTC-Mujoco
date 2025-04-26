import numpy as np
import mujoco
import torch

from scipy.spatial.transform import Rotation as R # type: ignore

class Robot():
    def __init__(self, model, data, visualize):
        self.model = model
        self.data = data

        self.robot_qpos_addr = self.model.joint('robotfree').qposadr[0]

        self.visualize = visualize
        if self.visualize:
            self.physicalRays = [
                self.model.geom('forward'),
                self.model.geom('right'),
                self.model.geom('backward'),
                self.model.geom('left'),
            ]

    def getRay(self, origin, direction):
        # Optional: geomgroup can be None (or a 6x1 uint8 array mask)
        geomgroup = None

        flg_static = 1
        bodyexclude = -1  # No body excluded (use valid body ID to exclude one)

        geomid = np.zeros((1, 1), dtype=np.int32)  # Writable array for result

        # Perform ray casting
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            direction,
            geomgroup,
            flg_static,
            bodyexclude,
            geomid
        )

        if geomid[0][0] == -1:
            return -1, -1

        return distance - 14, self.model.body(self.model.geom_bodyid[geomid[0][0]]).name

    def getDistances(self):
        origin = self.getPosition() + np.array([0, 0, 3.5])
        forwardDirection =   np.array([[0.0], [1.0], [0.0]])
        rightDirection =     np.array([[1.0], [0.0], [0.0]])
        backwardsDirection = np.array([[0.0], [-1.0], [0.0]])
        leftDirection =      np.array([[-1.0], [0.0], [0.0]])

        return [
            self.getRay(origin, forwardDirection),
            self.getRay(origin, rightDirection),
            self.getRay(origin, backwardsDirection),
            self.getRay(origin, leftDirection)
        ]
    
    def alterPhysicalRays(self, distances):
        forwardDistance, forwardRay = max(distances[0][0], 0), self.physicalRays[0]
        forwardRay.size[1] = forwardDistance / 2
        forwardRay.pos[1] = 14 + forwardDistance / 2

        rightDistance, rightRay = max(distances[1][0], 0), self.physicalRays[1]
        rightRay.size[1] = rightDistance / 2
        rightRay.pos[0] = 14 + rightDistance / 2

        backwardDistance, backwardRay = max(distances[2][0], 0), self.physicalRays[2]
        backwardRay.size[1] = backwardDistance / 2
        backwardRay.pos[1] = -14 - backwardDistance / 2

        leftDistance, leftRay = max(distances[3][0], 0), self.physicalRays[3]
        leftRay.size[1] = leftDistance / 2
        leftRay.pos[0] = -14 - leftDistance / 2
    
    def getState(self, deltaPosition):
        state = deltaPosition

        distances = self.getDistances()
        if self.visualize:
            self.alterPhysicalRays(distances)

        for distance in distances:
            if distance[1] != -1:
                state.append(min(1 / max(distance[0], 1e-6), 100))
                # state.append(0.0)
                continue

            state.append(0.0)

        # state.extend(self.getVelocity())
        state.extend([0, 0])
        state = torch.Tensor(state)
        return state

    def getPosition(self):
        return self.data.qpos[self.robot_qpos_addr: self.robot_qpos_addr+3]

    def getVelocity(self):
        return self.data.qvel[self.robot_qpos_addr: self.robot_qpos_addr+2]

    def setPosition(self, x, y):
        self.data.qpos[self.robot_qpos_addr    ] = x  # x-axis position
        self.data.qpos[self.robot_qpos_addr + 1] = y  # y-axis position

    def setVelocity(self, dx, dy, dz):
        self.data.qvel[self.robot_qpos_addr    ] = dx  # x-axis velocity
        self.data.qvel[self.robot_qpos_addr + 1] = dy  # y-axis velocity
        self.data.qvel[self.robot_qpos_addr + 2] = dz  # z-axis velocity

        # Set initial pose: rotate 90 degrees around Y-axis
        rot = R.from_euler('z', 0, degrees=True)
        quat = rot.as_quat()  # [x, y, z, w] format

        # MuJoCo expects [w, x, y, z]
        quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])

        # Set orientation in d.qpos
        self.data.qpos[self.robot_qpos_addr+3:self.robot_qpos_addr+7] = quat_mj

    def hasCollision(self):
        for contact in self.data.contact:
          if contact.geom1 != 0 and contact.geom2 != 0: # Check if the contact is valid
            return True
          
        return False