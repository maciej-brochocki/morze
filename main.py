import cv2
import numpy as np
import pafy



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


if __name__ == '__main__':
	url = "https://youtu.be/kNcV9Gy5lQc"
	camera_source = pafy.new(url).getbest()  #.allstreams[0]
	capture = cv2.VideoCapture(camera_source.url)
	flow = Flow()
	while (True):
		# Capture frame-by-frame
		ret, current_frame = capture.read()
		
		if ret:
			gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
			# gray = cv2.equalizeHist(gray)
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray = clahe.apply(gray)
			current_frame = flow.detect_flow(gray)
			cv2.imshow('frame', current_frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	capture.release()
	cv2.destroyAllWindows()
