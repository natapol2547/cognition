from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from controller import Robot, Camera

class CameraSensor:
    def __init__(self, robot: "Robot", name: str = "camera"):
        self.camera = cast("Camera", robot.getDevice(name))
        timestep = int(robot.getBasicTimeStep())
        self.camera.enable(timestep)

    def getImage(self):
        return self.camera.getImage()