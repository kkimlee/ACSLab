import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile
from pynput.keyboard import Listener, Key, KeyCode

# Use below in settings.json with Blocks environment
"""
{
	"SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
	"SettingsVersion": 1.2,
	"SimMode": "Multirotor",
	"ClockSpeed": 1,
	
	"Vehicles": {
		"Drone1": {
		  "VehicleType": "SimpleFlight",
		  "X": 4, "Y": 0, "Z": -2
		},
		"Drone2": {
		  "VehicleType": "SimpleFlight",
		  "X": 8, "Y": 0, "Z": -2
		},
		"Drone3": {
		  "VehicleType": "SimpleFlight",
		  "X": 4, "Y": 4, "Z": -2
		}

    }
}
"""

def handlePress(key):
	global drone, x, y, z
	
	state = client.getMultirotorState(vehicle_name=drone)
	yaw = state.rc_data.yaw
	print(yaw)
	
	print(state1.rc_data)
	
	if key == KeyCode(char='a'):
		client.moveByVelocityAsync(0, -1, 0, 0.5, vehicle_name=drone)

		
	elif key == KeyCode(char='d'):
		client.moveByVelocityAsync(0, 1, 0, 0.5, vehicle_name=drone)
		
	elif key == KeyCode(char='w'):
		client.moveByVelocityAsync(1, 0, 0, 0.5, vehicle_name=drone)
		

	elif key == KeyCode(char='s'):
		client.moveByVelocityAsync(-1, 0, 0, 0.5, vehicle_name=drone)
		
		
	elif key == KeyCode(char='z'):
		client.moveByVelocityAsync(1, 0, -1, 0.5, vehicle_name=drone)
	
	elif key == KeyCode(char='x'):
		client.moveByVelocityAsync(-1, 0, 1, 0.5, vehicle_name=drone)
		
	elif key == KeyCode(char='q'):
		client.rotateByYawRateAsync(-18, 0.5, vehicle_name=drone)
		
	elif key == KeyCode(char='e'):
		client.rotateByYawRateAsync(18, 0.5, vehicle_name=drone)
		
	elif key == KeyCode(char='1'):
		drone = 'Drone1'
	elif key == KeyCode(char='2'):
		drone = 'Drone2'
	elif key == KeyCode(char='3'):
		drone = 'Drone3'
	elif key == KeyCode(char='p'):
		client.armDisarm(False, "Drone1")
		client.armDisarm(False, "Drone2")
		client.armDisarm(False, "Drone3")
		client.reset()

		# that's enough fun for now. let's quit cleanly
		client.enableApiControl(False, "Drone1")
		client.enableApiControl(False, "Drone2")
		client.enableApiControl(False, "Drone3")

		return False

		
def handleRelease(key):
	print('Released: {}'.format(key))
	
	if key == Key.esc:
		return False

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone1")
client.enableApiControl(True, "Drone2")
client.enableApiControl(True, "Drone3")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Drone2")
client.armDisarm(True, "Drone3")


airsim.wait_key('Press any key to takeoff')

f1 = client.takeoffAsync(vehicle_name="Drone1")
f2 = client.takeoffAsync(vehicle_name="Drone2")
f3 = client.takeoffAsync(vehicle_name="Drone3")

f1.join()
f2.join()
f3.join()


state1 = client.getMultirotorState(vehicle_name="Drone1")
s = pprint.pformat(state1)
print("state: %s" % s)
state2 = client.getMultirotorState(vehicle_name="Drone2")
s = pprint.pformat(state2)
print("state: %s" % s)
state3 = client.getMultirotorState(vehicle_name="Drone3")
s = pprint.pformat(state3)
print("state: %s" % s)


drone = 'Drone1'
with Listener(on_press=handlePress, on_release=handleRelease) as listener:
    listener.join()


