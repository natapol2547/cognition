from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from controller import Robot, Compass

class CompassSensor:
    def __init__(self, robot: "Robot", name: str = "compass"):
        self.compass = cast("Compass", robot.getDevice(name))
        timestep = int(robot.getBasicTimeStep())
        self.compass.enable(timestep)

    def getValues(self):
        return self.compass.getValues()