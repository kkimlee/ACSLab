from pynput.keyboard import Listener, Key, KeyCode

def handlePress(key):
	print('Press: {}'.format(key))
	
	if key == KeyCode(char='a'):
		print('A')
		
def handleRelease(key):
	print('Released: {}'.format(key))
	
	if key == Key.esc:
		return False
		
with Listener(on_press=handlePress, on_release=handleRelease) as listener:
    listener.join()
