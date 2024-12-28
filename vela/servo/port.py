import serial
from typing import Optional

SUPPORTED_BAUDRATES = [4800, 9600, 14400, 19200, 38400, 57600, 115200, 128000, 250000, 500000, 1000000]
DEFAULT_BAUDRATE = 1000000

class SerialPort:
    def __init__(self, port_name: str, baudrate: int = DEFAULT_BAUDRATE) -> None:
        if baudrate not in SUPPORTED_BAUDRATES:
            raise ValueError(f"Unsupported baud rate: {baudrate}. Supported rates are: {SUPPORTED_BAUDRATES}")

        self.baudrate: int = baudrate
        self.port_name: str = port_name
        self.port: Optional[Serial] = None

    def __enter__(self) -> 'SerialPort':
        self.port = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            timeout=0)
        self.port.reset_input_buffer()
        return self

    def __exit__(self, *_) -> None:
        if self.port:
            self.port.close()

    def flush(self) -> None:
        if self.port:
            self.port.flush()

    def in_waiting(self) -> int:
        return self.port.in_waiting if self.port else 0

    def read(self, length: int) -> bytes:
        return self.port.read(length) if self.port else b''

    def write(self, packet: bytes) -> int:
        return self.port.write(packet) if self.port else 0
