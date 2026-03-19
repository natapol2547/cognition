from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from controller import Robot, PositionSensor

class EncoderSensor:
    def __init__(self, robot: "Robot", name: str):
        self.encoder = cast("PositionSensor", robot.getDevice(name))
        timestep = int(robot.getBasicTimeStep())
        self.encoder.enable(timestep)

    def getValue(self):
        return self.encoder.getValue()