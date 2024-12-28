import struct
from typing import Self
from vela.servo.port import SerialPort, DEFAULT_BAUDRATE

TXPACKET_MAX_LEN = 250
RXPACKET_MAX_LEN = 250
BROADCAST_ID = 0xFE  # 254
MAX_ID = 0xFC  # 252
STS_END = 0  # STServo bit end(STS/SMS=0, SCS=1)
PACK_FORMAT = {1: '<B', 2: '<H', 4: '<I'}

class PacketInfo:
    HEADER0 = 0
    HEADER1 = 1
    ID = 2
    LENGTH = 3
    INSTRUCTION = 4
    ERROR = 4
    PARAMS = 5

class ErrorFlags:
    VOLTAGE = 1
    ANGLE = 2
    OVERHEAT = 4
    OVERELE = 8
    OVERLOAD = 32

class Instructions:
    PING = 1
    READ = 2
    WRITE = 3
    REG_WRITE = 4
    ACTION = 5
    SYNC_READ = 130  # 0x82
    SYNC_WRITE = 131  # 0x83

def get_rx_error_msg(error):
    if error & ErrorFlags.VOLTAGE: return "Input voltage error"
    elif error & ErrorFlags.ANGLE: return "Angle send error"
    elif error & ErrorFlags.OVERHEAT: return "Overheat error"
    elif error & ErrorFlags.OVERELE: return "OverEle error"
    elif error & ErrorFlags.OVERLOAD: return "Overload error"
    else: return "Unknown error"


class PacketHandler:
    def __init__(self, port_name: str, baudrate: int = DEFAULT_BAUDRATE):
        self.port = SerialPort(port_name, baudrate)
        self.busy = False

    def __enter__(self) -> Self:
        self.port.__enter__()
        return self

    def __exit__(self, *_) -> None:
        self.port.__exit__()

    # Read / Write

    def ping(self, device_id: int) -> int:
        if device_id >= BROADCAST_ID:
            raise ValueError("Broadcast ID cannot be used for ping")
        self.send_recv_packet(device_id, [Instructions.PING])
        return self.read_value(device_id, 3, 2)

    def action(self, device_id: int = BROADCAST_ID) -> None:
        self.send_recv_packet(device_id, [Instructions.ACTION])

    def read_value(self, device_id: int, address: int, n_bytes: int) -> int:
        if n_bytes not in PACK_FORMAT:
            raise ValueError("`read_value` method can only read 1, 2, or 4 bytes at a time")
        data = self.read_bytes(device_id, address, n_bytes)
        return struct.unpack(PACK_FORMAT[n_bytes], bytes(data))[0]

    def write_value(self, device_id: int, address: int, n_bytes: int, data: int) -> None:
        if n_bytes not in PACK_FORMAT:
            raise ValueError("`write_value` method can only write 1, 2, or 4 bytes at a time")
        packed_data = struct.pack(PACK_FORMAT[n_bytes], data)
        self.write_bytes(device_id, address, list(packed_data))

    def read_bytes(self, device_id: int, address: int, length: int) -> list[int]:
        if device_id >= BROADCAST_ID:
            raise ValueError("Broadcast ID cannot be used for read instruction")
        return self.send_recv_packet(device_id, [Instructions.READ, address, length])

    def write_bytes(self, device_id: int, address: int, data: list[int]) -> None:
        if device_id >= BROADCAST_ID:
            raise ValueError("Broadcast ID cannot be used for write instruction")
        if len(data) + 5 > TXPACKET_MAX_LEN:  # 5: HEADER0 HEADER1 ID LENGTH <DATA> CHKSUM
            raise ValueError("Data length exceeds maximum allowed length")
        self.send_recv_packet(device_id, [Instructions.WRITE, address, *data])

    def reg_write_bytes(self, device_id: int, address: int, data: list[int]) -> None:
        if len(data) + 5 > TXPACKET_MAX_LEN:  # 5: HEADER0 HEADER1 ID LENGTH <DATA> CHKSUM
            raise ValueError("Data length exceeds maximum allowed length")
        self.send_recv_packet(device_id, [Instructions.REG_WRITE, address, *data])

    def reg_write_value(self, device_id: int, address: int, n_bytes: int, data: int) -> None:
        if n_bytes not in PACK_FORMAT:
            raise ValueError("`reg_write_value` method can only write 1, 2, or 4 bytes at a time")
        packed_data = struct.pack(PACK_FORMAT[n_bytes], data)
        self.reg_write_bytes(device_id, address, list(packed_data))

    # Send / Receive

    def send_recv_packet(self, device_id: int, data: list[int]) -> list[int]:
        self.send_packet(device_id, data)
        while True:
            rx_device_id, response = self.recv_packet()
            if rx_device_id == device_id:
                return response

    def send_packet(self, device_id: int, data: list[int]) -> None:
        if self.busy:
            raise RuntimeError("Port is currently busy")
        if len(data) + 5 > TXPACKET_MAX_LEN:  # 5: HEADER0 HEADER1 ID LENGTH <DATA> CHKSUM
            raise ValueError("Data length exceeds maximum allowed length")

        # create packet
        packet = [0xFF, 0xFF, device_id, len(data) + 1, *data]
        checksum = sum(packet[idx] for idx in range(2, len(packet)))
        packet.append(~checksum & 0xFF)

        # transmit
        self.port.flush()
        written_packet_length = self.port.write(bytes(packet))
        if written_packet_length != len(packet):
            raise RuntimeError("Failed to transmit the entire instruction packet")
        if device_id != BROADCAST_ID:
            self.busy = True

    def recv_packet(self) -> tuple[int, list[int]]:
        packet: list[int] = []
        wait_length = 6  # minimum length (HEADER0 HEADER1 ID LENGTH ERROR CHKSUM)

        while True:
            packet.extend(self.port.read(wait_length - len(packet)))
            if len(packet) >= wait_length:
                # remove any bytes before the header
                idx = next((i for i in range(len(packet) - 1) if packet[i] == 0xFF and packet[i + 1] == 0xFF), len(packet) - 2)
                if idx != 0:
                    packet = packet[idx:]
                    continue

                # if the packet is corrupt, remove the first byte in the packet
                if (packet[PacketInfo.ID] > 0xFD) or (packet[PacketInfo.LENGTH] > RXPACKET_MAX_LEN) or (packet[PacketInfo.ERROR] > 0x7F):
                    packet = packet[1:]
                    continue

                # re-calculate the exact length of the rx packet
                if wait_length != (packet[PacketInfo.LENGTH] + PacketInfo.LENGTH + 1):
                    wait_length = packet[PacketInfo.LENGTH] + PacketInfo.LENGTH + 1
                    continue

                # verify checksum
                checksum = sum(packet[i] for i in range(2, wait_length - 1))
                checksum = ~checksum & 0xFF
                if packet[wait_length - 1] != checksum:
                    raise RuntimeError("Checksum verification failed")

                # check for errors
                if packet[PacketInfo.ERROR] != 0:
                    raise RuntimeError(get_rx_error_msg(packet[PacketInfo.ERROR]))

                self.busy = False
                return packet[PacketInfo.ID], packet[PacketInfo.PARAMS:-1]
