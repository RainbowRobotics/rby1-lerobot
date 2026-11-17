"""XR teleoperation of the Rainbow Robotics RB-Y1, independent of any robot-learning framework.

Depends on NumPy and SciPy only. A host (the LeRobot plugin
``lerobot_teleoperator_rby1``, a simulator, a script) feeds one
:class:`~rby1_xr_teleop.xr_frame.XRFrame` and the robot's measured state per
tick and gets targets back.
"""

__version__ = "0.3.1"
