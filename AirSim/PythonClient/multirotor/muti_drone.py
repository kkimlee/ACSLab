import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile
from pynput.keyboard import Listener, Key, KeyCode
from threading import Thread

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

cameraTypeMap = { 
 "depth": airsim.ImageType.DepthVis,
 "segmentation": airsim.ImageType.Segmentation,
 "seg": airsim.ImageType.Segmentation,
 "scene": airsim.ImageType.Scene,
 "disparity": airsim.ImageType.DisparityNormalized,
 "normals": airsim.ImageType.SurfaceNormals
}

def handlePress(key):
	global drone, x, y, z
		
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


def key_event():
	with Listener(on_press=handlePress, on_release=handleRelease) as listener:
    		listener.join()

def show_img():
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	w = 256
	h = 144
	out = cv2.VideoWriter('ouput.avi', fourcc, 30.0, (w, h))
	
	idx = 0
	while True:
		rawImage = client.simGetImage("0", cameraTypeMap["scene"], vehicle_name=drone)
		if (rawImage == None):
			print("Camera is not returning image, please check airsim for error messages")
		else:
			png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
			png = png[:, :, :3]
			print(png.shape)
			cv2.imshow("img", png)
			
			if idx < 10:
				strIdx = "00"+str(idx)
			elif idx < 100:
				strIdx = "0"+str(idx)
			else:
				strIdx = str(idx)
			
			cv2.imwrite('frame/frame' + strIdx + '.jpg', png)
			out.write(png)
			
		idx += 1
		if cv2.waitKey(1) == ord('p'):
			break
	
	cv2.destroyAllWindows()
	out.release()
	
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

# with Listener(on_press=handlePress, on_release=handleRelease) as listener:
#   listener.join()

Listener(on_press=handlePress, on_release=handleRelease).start()
Thread(target=show_img).start()
