import asyncio
from enum import Enum
from typing import Optional, Tuple

# --- pymodbus import that works across common versions ---
try:
    # pymodbus >= 3
    from pymodbus.client import ModbusSerialClient
except Exception:
    try:
        # some builds
        from pymodbus.client.serial import ModbusSerialClient
    except Exception:
        # pymodbus 2.x
        from pymodbus.client.sync import ModbusSerialClient


class Robotiq2F85USBGripper:
    """
    Robotiq 2F-85 / 2F-140 gripper control via USB (USB-RS485 adapter) using Modbus RTU.

    This is a ROS-free replacement for the TCP/URCap style gripper interface.
    """

    # Modbus register bases from the Robotiq manual:
    # - Robot Output / Gripper Input first register: 0x03E8 (1000)
    # - Robot Input / Gripper Output first register: 0x07D0 (2000)
    OUT_REG0 = 0x03E8  # 1000
    IN_REG0  = 0x07D0  # 2000

    class GripperStatus(Enum):
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(Enum):
        # gOBJ meanings from manual:
        MOVING = 0
        DETECTED_OPENING = 1
        DETECTED_CLOSING = 2
        AT_REQUESTED = 3

    def __init__(
        self,
        port: str,
        slave_id: int = 9,
        baudrate: int = 115200,
        timeout_s: float = 0.05,
    ) -> None:
        self.port = port
        self.slave_id = int(slave_id)

        self.client = ModbusSerialClient(
            method="rtu",
            port=self.port,
            baudrate=int(baudrate),
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=float(timeout_s),
        )

        # serialize read/write to avoid interleaving on the serial line
        self.command_lock = asyncio.Lock()

        self._connected = False

    # ----------------------------
    # Low-level helpers
    # ----------------------------
    @staticmethod
    def _regs_to_bytes(regs) -> Tuple[int, int, int, int, int, int]:
        """
        Convert 3x16-bit registers into 6 bytes.
        We treat:
          high byte = byte0, low byte = byte1
        because that's how the Robotiq docs present "Byte 0, Byte 1, ..." per register.
        """
        b = []
        for r in regs:
            b0 = (r >> 8) & 0xFF
            b1 = r & 0xFF
            b.extend([b0, b1])
        return tuple(b[:6])

    @staticmethod
    def _pack_regs(action_req: int, pos_req: int, speed: int, force: int,
                  opt0: int = 0, opt1: int = 0) -> list:
        """
        Build the 3 output registers (1000..1002) from the 6 bytes:
          Byte0 action request
          Byte1 options
          Byte2 options/reserved
          Byte3 position request (rPR)
          Byte4 speed (rSP)
          Byte5 force (rFR)
        """
        action_req = int(action_req) & 0xFF
        pos_req = int(pos_req) & 0xFF
        speed = int(speed) & 0xFF
        force = int(force) & 0xFF
        opt0 = int(opt0) & 0xFF
        opt1 = int(opt1) & 0xFF

        reg0 = (action_req << 8) | opt0
        reg1 = (opt1 << 8) | pos_req
        reg2 = (speed << 8) | force
        return [reg0, reg1, reg2]

    def _read_holding_sync(self, address: int, count: int):
        # pymodbus 2.x uses unit= ; pymodbus 3 uses slave=
        try:
            rr = self.client.read_holding_registers(address, count, slave=self.slave_id)
        except TypeError:
            rr = self.client.read_holding_registers(address, count, unit=self.slave_id)

        if rr is None or getattr(rr, "isError", lambda: True)():
            raise RuntimeError(f"Modbus read failed (addr={address}, count={count}): {rr}")
        return rr.registers

    def _write_registers_sync(self, address: int, values: list):
        try:
            wr = self.client.write_registers(address, values, slave=self.slave_id)
        except TypeError:
            wr = self.client.write_registers(address, values, unit=self.slave_id)

        if wr is None or getattr(wr, "isError", lambda: True)():
            raise RuntimeError(f"Modbus write failed (addr={address}, values={values}): {wr}")
        return True

    async def _read_status_bytes(self) -> Tuple[int, int, int, int, int, int]:
        async with self.command_lock:
            regs = await asyncio.to_thread(self._read_holding_sync, self.IN_REG0, 3)
        return self._regs_to_bytes(regs)

    async def _write_command(self, action_req: int, pos_req: int, speed: int, force: int):
        regs = self._pack_regs(action_req, pos_req, speed, force)
        async with self.command_lock:
            await asyncio.to_thread(self._write_registers_sync, self.OUT_REG0, regs)

    # ----------------------------
    # Public API (matches your old style)
    # ----------------------------
    async def connect(self) -> None:
        ok = await asyncio.to_thread(self.client.connect)
        if not ok:
            raise RuntimeError(f"Failed to open Modbus RTU serial port: {self.port}")
        self._connected = True

    async def disconnect(self) -> None:
        if self._connected:
            await asyncio.to_thread(self.client.close)
        self._connected = False

    async def activate(self) -> None:
        """
        Activation sequence:
          - clear activation (rACT=0)
          - set activation (rACT=1)
          - wait until gSTA==ACTIVE
        """
        # rACT=0, rGTO=0 => action_req = 0x00
        await self._write_command(action_req=0x00, pos_req=0x00, speed=0x00, force=0x00)
        await asyncio.sleep(0.05)

        # rACT=1, rGTO=0 => action_req = 0x01
        await self._write_command(action_req=0x01, pos_req=0x00, speed=0x00, force=0x00)

        # wait for gSTA==3 (ACTIVE)
        for _ in range(200):
            if await self.is_active():
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("Gripper activation timed out")

    async def is_active(self) -> bool:
        b0, _, _, _, _, _ = await self._read_status_bytes()
        gSTA = (b0 >> 4) & 0x03
        gACT = b0 & 0x01
        return (gACT == 1) and (gSTA == self.GripperStatus.ACTIVE.value)

    async def get_fault_status(self) -> int:
        _, _, b2, _, _, _ = await self._read_status_bytes()
        return int(b2)

    async def get_object_status(self) -> "Robotiq2F85USBGripper.ObjectStatus":
        b0, _, _, _, _, _ = await self._read_status_bytes()
        gOBJ = (b0 >> 6) & 0x03
        return Robotiq2F85USBGripper.ObjectStatus(int(gOBJ))

    async def get_current_pressure(self) -> int:
        """
        Compatibility shim:
        The 2F-85 doesn't have "pressure" like a vacuum gripper.
        We return the *position byte* gPO (0..255), where:
          0   ~ open
          255 ~ closed
        """
        _, _, _, _, b4, _ = await self._read_status_bytes()
        return int(b4)

    # Convenience (explicit naming)
    async def get_position_byte(self) -> int:
        return await self.get_current_pressure()

    async def automatic_grip(self) -> None:
        """
        Close fully:
          rACT=1, rGTO=1 => action_req = 0x09
          rPR=0xFF (full close), rSP=0xFF, rFR=0xFF
        """
        # Ensure active
        if not await self.is_active():
            await self.activate()

        await self._write_command(action_req=0x09, pos_req=0xFF, speed=0xFF, force=0xFF)

    async def automatic_release(self) -> None:
        """
        Open fully:
          rACT=1, rGTO=1 => 0x09
          rPR=0x00 (open), rSP=0xFF, rFR=0xFF
        """
        if not await self.is_active():
            await self.activate()

        await self._write_command(action_req=0x09, pos_req=0x00, speed=0xFF, force=0xFF)
