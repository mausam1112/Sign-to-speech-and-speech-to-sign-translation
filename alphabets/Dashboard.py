
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore						#importing pyqt5 libraries

from PyQt5.QtCore import QTimer,Qt
from PyQt5 import QtGui
from tkinter import filedialog					#for file export module
from tkinter import *
import tkinter as tk
from matplotlib import pyplot as plt 			#for gesture viewer
from matplotlib.widgets import Button
import sys										#for pyqt
import os										#for removal of files
import cv2										#for the camera operations
import numpy as np 								#proceesing on images
import qimage2ndarray 							#convers images into matrix
import win32api
import winGuiAuto
import win32gui
import win32con									#for removing title cv2 window and always on top
import keyboard									#for pressing keys
import pyttsx3									#for tts assistance
import shutil									#for removal of directories

engine = pyttsx3.init()							#engine initialization for audio tts assistance


image_x, image_y = 64,64						#image resolution

from keras.models import load_model
classifier = load_model('Trained_model.h5')			#loading the model

def nothing(x):
	pass

def openimg():
	"""displays predefined gesture images at right most window"""
	cv2.namedWindow("Image", cv2.WINDOW_NORMAL )
	image = cv2.imread('template.png')
	cv2.imshow("Image",image)
	cv2.setWindowProperty("Image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.resizeWindow("Image",298,430)
	cv2.moveWindow("Image", 1052,214)


def removeFile():
	"""Removes the temp.txt and tempgest directory if any stop button is pressed oor application is closed"""
	try:
		os.remove("temp.txt")
	except:
		pass

def clearfunc(cam):
	"""shut downs the opened camera and calls removeFile() Func"""
	cam.release()
	cv2.destroyAllWindows()
	removeFile()

def clearfunc2(cam):
	"""shut downs the opened camera"""
	cam.release()
	cv2.destroyAllWindows()

def saveBuff(self,cam,finalBuffer):
	"""Save the file as temp.txt if save button is pressed in sentence formation through gui"""
	cam.release()
	cv2.destroyAllWindows()
	if(len(finalBuffer)>=1):
		f=open("temp.txt","w")
		for i in finalBuffer:
			f.write(i)
		f.close()
	self.textBrowser_4.setText("")


def checkFile():
	"""retrieve the content of temp.txt for export module """
	checkfile=os.path.isfile('temp.txt')
	if(checkfile==True):
		fr=open("temp.txt","r")
		content=fr.read()
		fr.close()
	else:
		content="No Content Available"
	return content


def to_speech():
    content = checkFile()
    engine.say(str(content).lower())
    try:
        engine.runAndWait()
    except:
        pass


def controlTimer(self):
	# if timer is stopped
	self.timer.isActive()
	# create video capture
	self.cam = cv2.VideoCapture(0)
	# start timer
	self.timer.start(20)
	
	
def backspace(self, new_text):
    if (len(new_text) >= 1):
        f = open(('temp.txt','w'))
        for char in new_text:
            if char == new_text[-1]:
                continue
            f.write(char)
        f.close()
    if (os.path.isfile('temp.txt')):
        fr = open('temp.txt')
        new_text = fr.read()
        fr.close()
        self.textBrowser_4.setText(new_text)


        


def predictor():
	""" Depending on model loaded and customgesture saved prediction is made by checking array or through SiFt algo"""
	import numpy as np
	from keras.preprocessing import image
	test_image = image.load_img('1.png', target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	gesname=''
	'''
	fileEntry=fileSearch()
	for i in range(len(fileEntry)):
		image_to_compare = cv2.imread("./SampleGestures/"+fileEntry[i])
		original = cv2.imread("1.png")
		sift = cv2.xfeatures2d.SIFT_create()
		kp_1, desc_1 = sift.detectAndCompute(original, None)
		kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

		index_params = dict(algorithm=0, trees=5)
		search_params = dict()
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		matches = flann.knnMatch(desc_1, desc_2, k=2)

		good_points = []
		ratio = 0.6
		for m, n in matches:
		      if m.distance < ratio*n.distance:
		             good_points.append(m)
		if(abs(len(good_points)+len(matches))>20):			#goodpoints and matcches sum from 1.png and customgestureimages is grater than 20
			gesname=fileEntry[i]
			gesname=gesname.replace('.png','')
			if(gesname=='sp'):								#sp is replaced with <space>
				gesname=' '
			return gesname
	'''

	if result[0][0] == 1:
		  return 'A'
	elif result[0][1] == 1:
		  return 'B'
	elif result[0][2] == 1:
		  return 'C'
	elif result[0][3] == 1:
		  return 'D'
	elif result[0][4] == 1:
		  return 'E'
	elif result[0][5] == 1:
		  return 'F'
	elif result[0][6] == 1:
		  return 'G'
	elif result[0][7] == 1:
		  return 'H'
	elif result[0][8] == 1:
		  return 'I'
	elif result[0][9] == 1:
		  return 'J'
	elif result[0][10] == 1:
		  return 'K'
	elif result[0][11] == 1:
		  return 'L'
	elif result[0][12] == 1:
		  return 'M'
	elif result[0][13] == 1:
		  return 'N'
	elif result[0][14] == 1:
		  return 'O'
	elif result[0][15] == 1:
		  return 'P'
	elif result[0][16] == 1:
		  return 'Q'
	elif result[0][17] == 1:
		  return 'R'
	elif result[0][18] == 1:
		  return 'S'
	elif result[0][19] == 1:
		  return 'T'
	elif result[0][20] == 1:
		  return 'U'
	elif result[0][21] == 1:
		  return 'V'
	elif result[0][22] == 1:
		  return 'W'
	elif result[0][23] == 1:
		  return 'X'
	elif result[0][24] == 1:
		  return 'Y'
	elif result[0][25] == 1:
		  return 'Z'



class Dashboard(QtWidgets.QMainWindow):
	def __init__(self):
		super(Dashboard, self).__init__()
		self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.FramelessWindowHint)

		self.setWindowIcon(QtGui.QIcon('icons/windowLogo.png'))
		self.title = 'Sign language Recognition'
		uic.loadUi('UI_Files/dash.ui', self)
		self.setWindowTitle(self.title)
		self.timer = QTimer()
		
		self.scan_sen.clicked.connect(self.scanSent)

		self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.exit_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

		self.exit_button.clicked.connect(self.quitApplication)
		self._layout = self.layout()
		self.label_3 = QtWidgets.QLabel()
		movie = QtGui.QMovie("icons/dashAnimation.gif")
		self.label_3.setMovie(movie)
		self.label_3.setGeometry(0,160,780,441)
		movie.start()
		self._layout.addWidget(self.label_3)
		self.setObjectName('Message_Window')

	def quitApplication(self):
		"""shutsdown the GUI window along with removal of files"""
		userReply = QMessageBox.question(self, 'Quit Application', "Are you sure you want to quit this app?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if userReply == QMessageBox.Yes:
			clearfunc(self.cam)
			keyboard.press_and_release('alt+F4')

	def scanSent(self):
		"""sentence formation module """
		try:
			clearfunc(self.cam)
		except:
			pass
		uic.loadUi('UI_Files/scan_sent.ui', self)
		self.setWindowTitle(self.title)
		
		self.exit_button_2.clicked.connect(self.quitApplication)
		if(self.scan_sen.clicked.connect(self.scanSent)):
			controlTimer(self)
		self.speech_gen.clicked.connect(to_speech)
		try:
			self.stop.clicked.connect(lambda:clearfunc2(self.cam))
		except:
			pass
		self.linkButton.clicked.connect(openimg)

		self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.exit_button_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.stop.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.save.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.speech_gen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		
		#try:
			#self.exit_button.clicked.connect(lambda:clearfunc(self.cam))
		#except:
			#pass
		
		img_text = ''
		append_text=''
		new_text=''
		finalBuffer=[]
		counts=0
		while True:
			ret, frame =self.cam.read()
			frame = cv2.flip(frame,1)
			try:
				frame=cv2.resize(frame,(331,310))

				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				img = cv2.rectangle(frame, (150,50),(300,200), (0,255,0), thickness=2, lineType=8, shift=0)
			except:
				keyboard.press_and_release('esc')
				keyboard.press_and_release('esc')

			height, width, channel = img.shape
			step = channel * width
                        # create QImage from image
			qImg = QImage(img.data, width, height, step, QImage.Format_RGB888)
                        # show image in img_label
			try:
				self.label_3.setPixmap(QPixmap.fromImage(qImg))
				slider=self.trackbar.value()
			except:
				pass

			lower_blue = np.array([0, slider, 0])
			upper_blue = np.array([179, 255, 255])
			imcrop = img[52:198, 152:298]
			hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
			mask1 = cv2.inRange(hsv, lower_blue, upper_blue)


			cv2.namedWindow("mask", cv2.WINDOW_NORMAL )
			cv2.imshow("mask", mask1)
			cv2.setWindowProperty("mask",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			cv2.resizeWindow("mask",118,108)
			cv2.moveWindow("mask", 713,264)

			img_name = "1.png"
			save_img = cv2.resize(mask1, (image_x, image_y))
			cv2.imwrite(img_name, save_img)
			img_text=predictor()

			hwnd = winGuiAuto.findTopWindow("mask")
			win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

			try:
				self.textBrowser.setText("\n      "+str(img_text))
			except:
				pass


			if cv2.waitKey(1) & 0xFF == ord('c'):
					try:
						counts+=1
						append_text += img_text
						new_text += img_text
						
						self.textBrowser_4.setText(new_text)
					except:
						append_text+=''

					if(len(append_text)>1):
						finalBuffer.append(append_text)
						append_text=''
					else:
						finalBuffer.append(append_text)
						append_text=''

			try:
				self.save.clicked.connect(lambda: saveBuff(self,self.cam,finalBuffer))
			except:
				pass
			if cv2.waitKey(1) == 27:
				break

			if keyboard.is_pressed('shift+s'):
				if(len(finalBuffer)>=1):
					f=open("temp.txt","w")
					for i in finalBuffer:
						f.write(i)
					f.close()
				break


		self.cam.release()
		cv2.destroyAllWindows()

		if os.path.exists('temp.txt'):
			QtWidgets.QMessageBox.about(self, "Information", "File is temporarily saved. Please click Speech Button ")
		try:
			self.textBrowser.setText("		 ")
		except:
			pass

	def help_func(self):
		""" help info shows in new dialog box"""
		try:
			clearfunc(self.cam)
		except:
			pass
		
		
		

	def exportFile(self):
		"""export file module with tts assistance and gesturre viewer"""
		try:
			clearfunc2(self.cam)
		except:
			pass
		clearfunc(self.cam)
		
				
	def on_click(self):
		"""Opens tkinter window to save file at desired location """
		content=checkFile()
		

	def gestureViewer(self):
		"""gesture viewer through matplotlib """
		pass

	def copy_paste_func(self):
		"""gesture viewer through matplotlib """
		clearfunc(self.cam)
		pass
		


	def scanSingle(self):
		"""Single gesture scanner """
		try:
			clearfunc(self.cam)
		except:
			pass
		

app = QtWidgets.QApplication([])
win = Dashboard()
win.show()
sys.exit(app.exec())
