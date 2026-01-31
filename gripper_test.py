import time
from pyrobotiqgripper import RobotiqGripper as RG

PORT = "/dev/ttyUSB0"
SLAVE = 9

# g = RG(portname=PORT, slaveAddress=SLAVE)
g = RG(portname=PORT)

g.resetActivate()

g.readAll()
g.write_registers(1000, [0b0000100100000000, 255, (200<<8)+150])
time.sleep(1.0)
g.readAll()

g.write_registers(1000, [0b0000100100000000, 0, (200<<8)+150])
time.sleep(1.0)
g.readAll()