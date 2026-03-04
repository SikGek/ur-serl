import pyrealsense2 as rs

def list_realsense_cameras():
    # Create a context object, which is the entry point to the rs-camera API
    ctx = rs.context()
    
    # Get a list of all connected devices
    devices = ctx.devices
    
    if len(devices) == 0:
        print("No RealSense cameras found.")
        return []

    print(f"Found {len(devices)} RealSense camera(s).")
    serial_numbers = []
    
    for i, device in enumerate(devices):
        # Retrieve the serial number (S/N) for each device
        serial_number = device.get_info(rs.camera_info.serial_number)
        model_name = device.get_info(rs.camera_info.name)
        
        print(f"Camera {i+1}: Model: {model_name}, Serial Number: {serial_number}")
        serial_numbers.append(serial_number)
        
    return serial_numbers

if __name__ == "__main__":
    camera_serials = list_realsense_cameras()
    # You can now use these serial numbers to select specific cameras in other parts of your code.
