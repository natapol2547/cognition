from __future__ import annotations
from types import ModuleType
from typing import TYPE_CHECKING, cast

import os
import sys
import importlib
from dotenv import load_dotenv

if TYPE_CHECKING:
    from controller import Motor, Robot, Supervisor


def get_controller_module() -> ModuleType:
    load_dotenv()
    webots_home = os.getenv("WEBOTS_HOME")
    if not webots_home:
        raise ValueError("WEBOTS_HOME not found in .env file!")

    controller_path = os.path.join(webots_home, "lib", "controller", "python")
    if controller_path not in sys.path:
        sys.path.append(controller_path)
    return importlib.import_module("controller")


def get_webots_robot() -> "Robot":
    controller = get_controller_module()
    return cast("Robot", controller.Robot())

def get_supervisor() -> "Supervisor":
    controller = get_controller_module()
    return cast("Supervisor", controller.Supervisor())