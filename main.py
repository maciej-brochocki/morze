import cv2
import numpy as np
import pafy
import argparse


class Flow(object):
	previousFrame = None  # store previous frame

	def compute_optical_flow(self, frame):
		# get optical flow
		flow = cv2.calcOpticalFlowFarneback(self.previousFrame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		fx, fy = flow[:, :, 0], flow[:, :, 1]
		# become independent of camera moves
		fx -= np.median(fx)
		fy -= np.median(fy)
		# convert to polar as more useful
		mag, ang = cv2.cartToPolar(fx, fy)
		mag = np.uint8(mag)
		return fx, fy, ang, mag

	def detect_flow(self, frame):
		new_frame = frame
		if self.previousFrame is not None:
			fx, fy, ang, mag = self.compute_optical_flow(frame)
			# fancy colours
			hsv = np.zeros(frame.shape + (3,), dtype=np.uint8)
			hsv[..., 0] = ang*180/np.pi/2
			hsv[..., 1] = 255
			# hsv[..., 2] = np.minimum(mag*4, 255)
			hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
			new_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		self.previousFrame = frame
		return new_frame


flow = Flow()


def mode0(frame):
	# do nothing, just pass input
	return frame


def mode1(frame):
	# convert to grayscale
	return cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)


def mode2(frame):
	# convert to grayscale and histogram equalization
	frame = mode1(frame)
	return cv2.equalizeHist(frame)


def mode3(frame):
	# convert to grayscale and histogram normalization
	frame = mode1(frame)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	return clahe.apply(frame)


def mode4(frame):
	# convert to grayscale and calculate motion vectors
	frame = mode3(frame)
	return flow.detect_flow(frame)


mode5previousFrame = None  # store previous frame
def mode5(frame):
	# simple diff
	global mode5previousFrame
	if mode5previousFrame is not None:
		current_frame = frame - mode5previousFrame
	else:
		current_frame = frame
	mode5previousFrame = frame
	return current_frame


def mode6(frame):
	# edge detection
	return cv2.Canny(frame,100,200)


mode7previousFrame = None  # store previous frame
mode7previousFrame2 = None  # store previous previous frame
mode7previousFrame3 = None  # store previous previous previous frame
def mode7(frame):
	# buffered diff
	global mode7previousFrame
	global mode7previousFrame2
	global mode7previousFrame3
	if mode7previousFrame is not None:
		current_frame = (frame - mode7previousFrame - mode7previousFrame2 - mode7previousFrame3)/4 - frame
		mode7previousFrame3 = mode7previousFrame2
		mode7previousFrame2 = mode7previousFrame
	else:
		current_frame = frame
		mode7previousFrame2 = frame
		mode7previousFrame3 = frame
	mode7previousFrame = frame
	return current_frame


def mode8(frame):
	# do nothing, just pass input
	return frame


def mode9(frame):
	# do nothing, just pass input
	return frame


if __name__ == '__main__':
	# Parse input parameters
	parser = argparse.ArgumentParser()
	input = parser.add_mutually_exclusive_group()
	input.add_argument("-camera", dest='camera', type=int, help="Number of a computer camera to use as an input, for example: 0", default=0, nargs='?')
	input.add_argument("-file", dest='file', type=str, help="Name of a file to use as an input, for example: movie.mp4", nargs='?')
	input.add_argument("-stream", dest='stream', type=str, help="Youtube video id to use as an input, for example: WHPEKLQID4U", nargs='?')
	parser.add_argument("-mode", dest="mode", type=int, help="Visualization mode", choices=range(10), default=0, nargs='?')
	args = parser.parse_args()

	# Configure input
	if args.stream:
		camera_source = pafy.new(args.stream).getbest()  # use youtube videos, sometimes live cameras can be problematic
		capture = cv2.VideoCapture(camera_source.url)
	elif args.file:
		capture = cv2.VideoCapture(args.file)
	else:
		capture = cv2.VideoCapture(args.camera)

	mode = args.mode
	modes = [mode0, mode1, mode2, mode3, mode4, mode5, mode6, mode7, mode8, mode9]
	while (True):
		# Capture frame-by-frame
		ret, current_frame = capture.read()
		
		if ret:
			current_frame = modes[mode](current_frame)
			cv2.imshow('frame', current_frame)
		else:
			break

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		if key >= ord('0') and key <= ord('9'):
			mode = key - ord('0')
	capture.release()
	cv2.destroyAllWindows()
