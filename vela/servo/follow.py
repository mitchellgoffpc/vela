#!/usr/bin/env python
import time
from vela.servo.sts import STSHandler, Registers

# TODO: Discover these automatically
LEADER_PORT = '/dev/tty.usbmodem58CD1772041'
FOLLOWER_PORT = '/dev/tty.usbmodem58CD1774791'
MAX_SPEED = 2400
MAX_ACCEL = 50

with STSHandler(LEADER_PORT) as leader, STSHandler(FOLLOWER_PORT) as follower:
    print("Press enter to calibrate.")
    input()

    print("Calibrating...")
    for name, arm in {'leader': leader, 'follower': follower}.items():
        for sts_id in range(1, 7):
            # Set offset to 0
            with arm.unlock_eprom(sts_id):
                arm.write_value(sts_id, Registers.POSITION_CORRECTION, 2, 0)

            # Calculate offset
            pos = arm.read_pos(sts_id)
            offset = pos - 2048
            if offset > 2048:
                offset -= 4096
            if offset < -2048:
                offset += 4096
            if offset < 0:
                offset = 2048 + abs(offset)

            # Apply calculated offset
            with arm.unlock_eprom(sts_id):
                arm.write_value(sts_id, Registers.POSITION_CORRECTION, 2, offset)

            pos = arm.read_pos(sts_id)
            print(f"{name}, STS {sts_id}: position = {pos}")

    print("Calibration complete.")

    # Main loop
    try:
        while True:
            for sts_id in range(1, 7):
                leader_position = leader.read_pos(sts_id)
                follower.write_target(sts_id, leader_position, MAX_SPEED, MAX_ACCEL)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    # Disable torque before exiting
    for sts_id in range(1, 7):
        leader.write_value(sts_id, Registers.TORQUE_SWITCH, 1, 0)
        follower.write_value(sts_id, Registers.TORQUE_SWITCH, 1, 0)
