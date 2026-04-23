from typing import TYPE_CHECKING, cast
import numpy as np
if TYPE_CHECKING:
    from controller import Robot, Compass

COMPASS_NOISE = 0.02

class CompassSensor:
    def __init__(self, robot: "Robot", name: str = "compass"):
        self.compass = cast("Compass", robot.getDevice(name))
        timestep = int(robot.getBasicTimeStep())
        self.compass.enable(timestep)

    def getValues(self):
        return self.compass.getValues() + np.random.normal(0, COMPASS_NOISE, 3)