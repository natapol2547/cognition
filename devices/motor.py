from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from controller import Robot, Motor

class MotorActuator:
    def __init__(self, robot: "Robot", name: str):
        self.motor = cast("Motor", robot.getDevice(name))
        self.motor.setPosition(float("inf"))
        self.stop()

    def setVelocity(self, velocity: float):
        self.motor.setVelocity(velocity)
    
    def stop(self):
        self.setVelocity(0.0)