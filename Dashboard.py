
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

import pyaudio
import wave
import scipy.io.wavfile as wav
from deepspeech import Model

engine = pyttsx3.init()							#engine initialization for audio tts assistance


image_x, image_y = 64,64						#image resolution

from keras.models import load_model
classifier = load_model('Trained_model_20epoch.h5')			#loading the model

MODEL_PATH = 'H:\deepspeech-0.7.4-models.pbmm'
SCORER_PATH = 'H:\deepspeech-0.7.4-models.scorer'

frameWidth = 640
frameHeight = 480

#image path for alphabets
a_path, b_path = 'H:/alphabets/a.jpg', 'H:/alphabets/b.jpg'
c_path, d_path = 'H:/alphabets/c.jpg', 'H:/alphabets/d.jpg'
e_path, f_path = 'H:/alphabets/e.jpg', 'H:/alphabets/f.jpg'
g_path, h_path = 'H:/alphabets/g.jpg', 'H:/alphabets/h.jpg'
i_path, j_path = 'H:/alphabets/i.jpg', 'H:/alphabets/j.jpg'
k_path, l_path = 'H:/alphabets/k.jpg', 'H:/alphabets/l.jpg'
m_path, n_path = 'H:/alphabets/m.jpg', 'H:/alphabets/n.jpg'
o_path, p_path = 'H:/alphabets/o.jpg', 'H:/alphabets/p.jpg'
q_path, r_path = 'H:/alphabets/q.jpg', 'H:/alphabets/r.jpg'
s_path, t_path = 'H:/alphabets/s.jpg', 'H:/alphabets/t.jpg'
u_path, v_path = 'H:/alphabets/u.jpg', 'H:/alphabets/v.jpg'
w_path, x_path = 'H:/alphabets/w.jpg', 'H:/alphabets/x.jpg'
y_path, z_path = 'H:/alphabets/y.jpg', 'H:/alphabets/z.jpg'

#aphabet dictionaries
alphabets = {'a':a_path, 'b':b_path, 'c':c_path, 'd':d_path, 'e':e_path,
            'f':f_path, 'g':g_path, 'h':h_path, 'i':i_path, 'j':j_path,
            'k':k_path, 'l':l_path, 'm':m_path, 'n':n_path, 'o':o_path,
            'p':p_path, 'q':q_path, 'r':r_path, 's':s_path, 't':t_path,
            'u':u_path, 'v':v_path, 'w':w_path, 'x':x_path, 'y':y_path,
            'z':z_path}

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

def info(self):
        try:
                self.label_info.setText("\t Recording Started.")
        except:
                pass


def predictor():
	""" Depending on model loaded and customgesture saved prediction is made by checking array or through SiFt algo"""
	import numpy as np
	from keras.preprocessing import image
	test_image = image.load_img('1.png', target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	gesname=''

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
	elif result[0][26] == 1:
		  return 'space'

        
def stackImages(imgArray, scale, lables=[]):
    sizeW = imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])

    # added to display images if num of images in each column differs
    for i in range(1, rows):
        cols_nxt_row = len(imgArray[i])
        if cols_nxt_row > cols:
            cols = cols_nxt_row

    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def show_images(text):
    kernel = np.ones((5, 5), np.uint8)

    imgBlank = np.zeros((200, 200), np.uint8)

    '''
    # make a list of images of all letters in a word
    # make list of images of every words.
    '''

    # At Maximum 8 word can be displayed.
    # text = 'This implementation blew my mind away due to its difficulties.'
    text = text.lower()

    # Count the number of words
    word_num = text.count(' ') + 1

    list1, list2, list3, list4, list5 = [], [], [], [], []
    list6, list7, list8, list9, list10 = [], [], [], [], []

    empty_lists = [list1, list2, list3, list4, list5, list6, list7, list8, list9, list10]
    main_list = []
    list_len = []

    # create empty list with number equal to the word_num
    # empty_lists[:word_num]

    splited_to_sentence = text.split('.')

    for sentence in splited_to_sentence:
        splited_to_words = sentence.split()  # SPLIT EACH SENTENCE TO LIST OF WORDS
        for list_name, word in zip(empty_lists[:word_num], splited_to_words):
            for letter in word:
                img = cv2.imread(alphabets[letter])
                list_name.append(img)  # Appends image of letters in a word to single list
            main_list.append(list_name)  # Appends list of words to main list
            list_len.append(len(list_name))  # List appends length of each words
        # print(main_list)

    # print([len(lst) for lst in main_list])

    len_main = len(main_list)
    max_len = max(list_len)
    # print('len ' + str(len(main_list)))

    # better logic
    for list_name, i in zip(main_list, range(len_main)):
        if list_len[i] < max_len:
            diff = max_len - list_len[i]
            for j in range(diff):
                list_name.append(imgBlank)

    # print([len(lst) for lst in main_list])

    div, mod = int(len_main / 7), len_main % 7

    # print(div, mod)

    rate = 7  # controls the number of rows of imagaes to be shown in one window
    mod_copy = mod
    index_3 = len_main - mod
    for i in range(div + 1, 0, -1):
        if mod != 0:
            win_name = 'Gestures_{}'.format(i)
            index = len_main - mod
            StackedImages = stackImages(([lst for lst in main_list[index:]]), 0.6)
            cv2.imshow(win_name, StackedImages)
            mod = 0
            continue
        win_name = 'Gestures_{}'.format(i)
        index_2 = len_main - mod_copy - rate
        StackedImages = stackImages(([lst for lst in main_list[index_2:index_3]]), 0.6)
        cv2.imshow(win_name, StackedImages)
        rate += 7
        index_3 = index_2

    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Dashboard(QtWidgets.QMainWindow):
	def __init__(self):
		super(Dashboard, self).__init__()
		self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.FramelessWindowHint)

		self.setWindowIcon(QtGui.QIcon('icons/windowLogo.png'))
		self.title = 'Sign language Recognition'
		uic.loadUi('UI_Files/dash.ui', self)
		self.setWindowTitle(self.title)
		self.timer = QTimer()
		
		self.signtospeech.clicked.connect(self.SignToSpeech)
		self.speechtosign.clicked.connect(self.SpeechToSign)
		self.help_btn.clicked.connect(self.help_dash)
		self.exit_button.clicked.connect(self.quitApplication)
		self.clear_dash.clicked.connect(self.clear_dashboard)

		self.signtospeech.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.speechtosign.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.help_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.exit_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.clear_dash.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

		
		self._layout = self.layout()
		#self.label_3 = QtWidgets.QLabel()
		#movie = QtGui.QMovie("icons/dashAnimation.gif")
		#self.label_3.setMovie(movie)
		#self.label_3.setGeometry(0,160,780,441)
		#movie.start()
		#self._layout.addWidget(self.label_3)
		self.setObjectName('Message_Window')

	def quitApplication(self):
		"""shutsdown the GUI window along with removal of files"""
		userReply = QMessageBox.question(self, 'Quit Application', "Are you sure you want to quit this app?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if userReply == QMessageBox.Yes:
			clearfunc(self.cam)
			keyboard.press_and_release('alt+F4')

	def SignToSpeech(self):
		"""sentence formation module """
		try:
			clearfunc(self.cam)
		except:
			pass
		uic.loadUi('UI_Files/signtospeech.ui', self)
		self.setWindowTitle(self.title)
		
		self.exit_button_2.clicked.connect(self.quitApplication)
		self.speechtosign.clicked.connect(self.SpeechToSign)
		if(self.signtospeech.clicked.connect(self.SignToSpeech)):
			controlTimer(self)
		self.speech_gen.clicked.connect(to_speech)
		try:
			self.stop.clicked.connect(lambda:clearfunc2(self.cam))
		except:
			pass
		self.linkButton.clicked.connect(openimg)

		self.signtospeech.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.speechtosign.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.exit_button_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.stop.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.save.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.speech_gen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

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
			cv2.moveWindow("mask", 713,244)

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
		help_info = "\tPlease click Record Button to start \n\tvoice recording for 5 seconds. Then \n\twait for the gestures to be \n\tdisplayed for few seconds in new window."
		try:
			self.label_help.setText(help_info)
		except:
			pass

	def clear_help_info(self):		
		try:
			self.label_help.setText('')
		except:
			pass

	def arbi(self):
		"""add something here """
		pass


	def SpeechToSign(self):
		#converts audio file to ASL
		try:
			clearfunc2(self.cam)
		except:
			pass
		uic.loadUi('UI_Files/speech_sign.ui', self)
		self.setWindowTitle(self.title)

		self.signtospeech.clicked.connect(self.SignToSpeech)
		if(self.speechtosign.clicked.connect(self.SpeechToSign)):
			controlTimer(self)
		self.help_2.clicked.connect(self.help_func)
		self.clear.clicked.connect(self.clear_help_info)
		self.exit_button.clicked.connect(self.quitApplication)

		self.signtospeech.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.speechtosign.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.record.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.help_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.clear.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
		self.exit_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

		if not os.path.exists('AudioFile'):
			os.mkdir('AudioFile')
		output_stream_file = 'AudioFile/speech_stream.wav'
		self.record.clicked.connect(self.record_voice_and_predict_text)
		
		

	def record_voice_and_predict_text(self):
		"""Records the speech and predicts its text """
		#Recording the speech
		
		stream_file_name = 'AudioFile/speech_stream.wav'
		stream_format = pyaudio.paInt16  # Sampling size and format
		no_of_channels = 1  # Number of audio channels
		sampling_rate = 16000  # Sampling rate in Hertz
		frames_count = 1024  # Number of frames per buffer
		record_seconds = 5

		stream = pyaudio.PyAudio()

		stream_data = stream.open(format=stream_format,
                                          channels=no_of_channels,
                                          rate=sampling_rate,
                                          input=True,
                                          frames_per_buffer=frames_count)
		frames = [stream_data.read(frames_count) for i in range(0, int(sampling_rate / frames_count * record_seconds))]
		stream_data.stop_stream()
		stream_data.close()
		stream.terminate()

		wave_file = wave.open(stream_file_name, 'wb')
		wave_file.setnchannels(no_of_channels)
		wave_file.setsampwidth(stream.get_sample_size(stream_format))
		wave_file.setframerate(sampling_rate)
		wave_file.writeframes(b''.join(frames))
		wave_file.close()

		try:
			self.label_info.setText('Recording completed.')
		except:
			pass
		
		#Text prediction Part
		alpha = 0.75
		beta = 1.85
		beam_width = 500

		# Initialize the model
		speech_model = Model(MODEL_PATH)

		# set beam width. A larger beam width value generates better results at the cost of decoding time.
		speech_model.setBeamWidth(beam_width)

		# Enable language scorer to improve the accuracy
		speech_model.enableExternalScorer(SCORER_PATH)
		# You can play with setting the model Beam Width, Scorer language model weight and word insertion weight

		# Set hyperparameters alpha and beta of the external scorer.
		# alpha: Language model weight.
		# beta: Word insertion weight
		speech_model.setScorerAlphaBeta(alpha, beta)

		# Use scipy to covert wav file into numpy array
		_, audio = wav.read(stream_file_name)
		text = speech_model.stt(audio)
		try:
			self.text_pred.setText(text)
		except:
			pass
		show_images(text)
		#print(speech_model.stt(audio))

		
		
		
				
	def help_dash(self):
		"""display information about help """
		help_value = ' Click Speech-Sign to convert speech to sign \n Sign-Speech to convet sign gestures to speech.'
		try:
			self.help_info.setText(help_value)
		except:
			pass

	def clear_dashboard(self):
		"""clear the field of help """
		try:
			self.help_info.setText('')
		except:
			pass
		

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
