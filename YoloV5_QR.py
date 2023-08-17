import numpy as np
import torch
import cv2
from time import time



class İhaTakip:

	def __init__(self, capture_index, model_name):
		"""
		hangi kamerayı kullancağımız, hangi modeli kullanacağımız ekran kartı mı yoksa işlemci mi kullanacağız
		ve bazı değişkenlere atama yapıyoruz
		"""
		self.capture_index = capture_index
		self.model = self.load_model(model_name)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.qrCodeDetector = cv2.QRCodeDetector()
		print("Using Device: ", self.device) 
		self.qr_enable = True
		self.x = 0
		
	def get_video_capture(self):
		"""
		kameradan görüntü alıyoruz
		"""
	  
		return cv2.VideoCapture(self.capture_index)

	def load_model(self, model_name):
		"""
		Pytorch hub'dan Yolov5 modelini indiriyoruz
		ve bunu modüle geri döndürüyoruz 
		"""
		if model_name:
			model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=False)
		else:
			model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
		return model

	def score_frame(self, frame):
		"""
		kameradan aldığı görüntüyü modele sokarak ondan tahmin oranı alıyoruz 
		"""
		self.model.to(self.device)
		frame = [frame]
		results = self.model(frame)
		labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
		return labels, cord

	def plot_boxes(self, results, frame):
		"""
		aranan objenin hangi konumlar içinde olduğunu buluyoruz.
		"""
		labels, cord = results
		n = len(labels)
		x_shape, y_shape = frame.shape[1], frame.shape[0]
		for i in range(n):
			row = cord[i]
			if row[4] >= 0.8:
				x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
				bgr = (0, 0, 255)
				cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
				x = int((x1+x2)/2)
				y = int((y1+y2)/2)

				cv2.line(frame,(512,384),(x,y),(255,255,255),2)
				cv2.line(frame,(512,384),(x,y),(255,255,255),2)

		return frame

	def read_Qr(self, frame):
		decodedText, points, _ = self.qrCodeDetector.detectAndDecode(frame)
		if points is not None:
			sol1,üst1=int(points[0][0][0]),int(points[0][0][1])
			sağ1,üst2=int(points[0][1][0]),int(points[0][1][1])
			sağ2,alt1=int(points[0][2][0]),int(points[0][2][1])
			sol2,alt2=int(points[0][3][0]),int(points[0][3][1])

			if len(decodedText)!=0:
				cv2.line(frame,(sol1,üst1),(sağ1,üst2),(255,0,0),4) #sol üst-sağ üst
				cv2.line(frame,(sağ1,üst2),(sağ2,alt1),(255,0,0),4) #sağ üst-sağ alt
				cv2.line(frame,(sağ2,alt1),(sol2,alt2),(255,0,0),4) #sağ alt-sol alt
				cv2.line(frame,(sol2,alt2),(sol1,üst1),(255,0,0),4) #sol alt-sol üst
				cv2.putText(frame,decodedText,(sağ1,üst2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
		
			if self.x==0 and len(decodedText)!=0:
				gonderilecek_metin=decodedText
				print(gonderilecek_metin)
				self.x=1
			
		return frame
			
	def __call__(self):
		"""
		kameramızı açarak aranan nesnenin nerede olduğunu hangi nesne olduğunu ve % kaç olasılıkla onun olduğunu yazıyoruz.
		"""       
		cap = self.get_video_capture()
		assert cap.isOpened()
		while True:
			
			ret ,frame = cap.read()
			if ret:
			#assert ret
				
				if not self.qr_enable:
					frame = cv2.resize(frame, (1024,768))
					frame = self.read_Qr(frame)
				else:
					start_time = time()
					frame = cv2.resize(frame, (1024,768))
					results = self.score_frame(frame)

					frame = self.plot_boxes(results, frame)
					end_time = time()
					fps = 1/np.round(end_time - start_time, 2)
					cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
					
					cv2.rectangle(frame,(256,76),(768,692),(255,0,0), 2)
					cv2.putText(frame,'.',(505,380),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),10)  				
				
				cv2.imshow('YOLOv5 Detection', frame)
				if cv2.waitKey(1) & 0xFF==ord('q'):
					self.qr_enable= not self.qr_enable
				
				elif cv2.waitKey(1)== ord('w'):
					break
				
			else:
				break

		cap.release()
		cv2.destroyAllWindows()

a=İhaTakip(capture_index=0,model_name="newbest.pt")
a()