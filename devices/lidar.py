from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from controller import Robot, Lidar

class LidarSensor:
    def __init__(self, robot: "Robot", name: str = "lidar"):
        self.lidar = cast("Lidar", robot.getDevice(name))
        timestep = int(robot.getBasicTimeStep())
        self.lidar.enable(timestep)

    def getRangeImage(self):
        return self.lidar.getRangeImage()
    
    def getFov(self):
        return self.lidar.getFov()
    
    def getMaxRange(self):
        return self.lidar.getMaxRange()