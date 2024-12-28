import struct
from typing import Iterator
from contextlib import contextmanager
from vela.servo.packets import PacketHandler

class BaudRates:
    RATE_1M     = 0
    RATE_500K   = 1
    RATE_250K   = 2
    RATE_128K   = 3
    RATE_115200 = 4
    RATE_76800  = 5
    RATE_57600  = 6
    RATE_38400  = 7

class Registers:
    FIRMWARE_MAJOR           = 0x00
    FIRMWARE_MINOR           = 0x01
    SERVO_MAJOR              = 0x03
    SERVO_MINOR              = 0x04
    ID                       = 0x05
    BAUDRATE                 = 0x06
    RESPONSE_DELAY           = 0x07
    RESPONSE_STATUS_LEVEL    = 0x08
    MINIMUM_ANGLE            = 0x09
    MAXIMUM_ANGLE            = 0x0B
    MAXIMUM_TEMPERATURE      = 0x0D
    MAXIMUM_VOLTAGE          = 0x0E
    MINIMUM_VOLTAGE          = 0x0F
    MAXIMUM_TORQUE           = 0x10
    UNLOADING_CONDITION      = 0x13
    LED_ALARM_CONDITION      = 0x14
    POS_PROPORTIONAL_GAIN    = 0x15
    POS_DERIVATIVE_GAIN      = 0x16
    POS_INTEGRAL_GAIN        = 0x17
    MINIMUM_STARTUP_FORCE    = 0x18
    CK_INSENSITIVE_AREA      = 0x1A
    CCK_INSENSITIVE_AREA     = 0x1B
    CURRENT_PROTECTION_TH    = 0x1C
    ANGULAR_RESOLUTION       = 0x1E
    POSITION_CORRECTION      = 0x1F
    OPERATION_MODE           = 0x21
    TORQUE_PROTECTION_TH     = 0x22
    TORQUE_PROTECTION_TIME   = 0x23
    OVERLOAD_TORQUE          = 0x24
    SPEED_PROPORTIONAL_GAIN  = 0x25
    OVERCURRENT_TIME         = 0x26
    SPEED_INTEGRAL_GAIN      = 0x27
    TORQUE_SWITCH            = 0x28
    TARGET_ACCELERATION      = 0x29
    TARGET_POSITION          = 0x2A
    RUNNING_TIME             = 0x2C
    RUNNING_SPEED            = 0x2E
    TORQUE_LIMIT             = 0x30
    WRITE_LOCK               = 0x37
    CURRENT_POSITION         = 0x38
    CURRENT_SPEED            = 0x3A
    CURRENT_DRIVE_VOLTAGE    = 0x3C
    CURRENT_VOLTAGE          = 0x3E
    CURRENT_TEMPERATURE      = 0x3F
    ASYNCHRONOUS_WRITE_ST    = 0x40
    STATUS                   = 0x41
    MOVING_STATUS            = 0x42
    CURRENT_CURRENT          = 0x45


class STSHandler(PacketHandler):
    def read_pos(self, device_id: int) -> int:
        return self.read_value(device_id, Registers.CURRENT_POSITION, 2)

    def read_speed(self, device_id: int) -> int:
        return self.read_value(device_id, Registers.CURRENT_SPEED, 2)

    def read_moving(self, device_id: int) -> bool:
        return bool(self.read_value(device_id, Registers.MOVING_STATUS, 1))

    def write_target(self, device_id: int, position: int, speed: int, acc: int) -> None:
        data = [*struct.pack('<B', acc), *struct.pack('<H', position), 0, 0, *struct.pack('<H', speed)]
        return self.write_bytes(device_id, Registers.TARGET_ACCELERATION, data)

    def reg_write_target(self, device_id: int, position: int, speed: int, acc: int):
        data = [*struct.pack('<B', acc), *struct.pack('<H', position), 0, 0, *struct.pack('<H', speed)]
        return self.reg_write_bytes(device_id, Registers.TARGET_ACCELERATION, data)

    @contextmanager
    def unlock_eprom(self, device_id: int) -> Iterator[None]:
        self.write_value(device_id, Registers.WRITE_LOCK, 1, 0)
        yield
        self.write_value(device_id, Registers.WRITE_LOCK, 1, 1)
