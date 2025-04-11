import numpy as np
import mujoco

class Robot():
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.robot_qpos_addr = self.model.joint('robotfree').qposadr[0]

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
        origin = self.getPosition() + np.array([0, 0, 35])
        forwardDirection = np.array([[0.0], [-1.0], [0.0]])

        return [
            self.getRay(origin, forwardDirection)
        ]
    
    def getPosition(self):
        return self.data.qpos[self.robot_qpos_addr: self.robot_qpos_addr+3]

    def getVelocity(self):
        return self.data.qvel[self.robot_qpos_addr: self.robot_qpos_addr+3]

    def setPosition(self, x, y):
        self.data.qpos[self.robot_qpos_addr    ] = x  # x-axis velocity
        self.data.qpos[self.robot_qpos_addr + 1] = y  # y-axis velocity

    def setVelocity(self, dx, dy, dz):
        self.data.qvel[self.robot_qpos_addr    ] = dx  # x-axis velocity
        self.data.qvel[self.robot_qpos_addr + 1] = dy  # y-axis velocity
        self.data.qvel[self.robot_qpos_addr + 2] = dz  # z-axis velocity

    def hasCollision(self):
        for contact in self.data.contact:
          if contact.geom1 != 0 and contact.geom2 != 0: # Check if the contact is valid
            return True
          
        return False