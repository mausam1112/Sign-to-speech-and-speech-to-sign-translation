#this code runs for only single sentence
#without seperator other than space

import cv2
import numpy as np

frameWidth = 640
frameHeight = 480

#image path for alphabets
a_path = 'H:/alphabets/a.jpg'
b_path = 'H:/alphabets/b.jpg'
c_path = 'H:/alphabets/c.jpg'
d_path = 'H:/alphabets/d.jpg'
e_path = 'H:/alphabets/e.jpg'
f_path = 'H:/alphabets/f.jpg'
g_path = 'H:/alphabets/g.jpg'
h_path = 'H:/alphabets/h.jpg'
i_path = 'H:/alphabets/i.jpg'
j_path = 'H:/alphabets/j.jpg'
k_path = 'H:/alphabets/k.jpg'
l_path = 'H:/alphabets/l.jpg'
m_path = 'H:/alphabets/m.jpg'
n_path = 'H:/alphabets/n.jpg'
o_path = 'H:/alphabets/o.jpg'
p_path = 'H:/alphabets/p.jpg'
q_path = 'H:/alphabets/q.jpg'
r_path = 'H:/alphabets/r.jpg'
s_path = 'H:/alphabets/s.jpg'
t_path = 'H:/alphabets/t.jpg'
u_path = 'H:/alphabets/u.jpg'
v_path = 'H:/alphabets/v.jpg'
w_path = 'H:/alphabets/w.jpg'
x_path = 'H:/alphabets/x.jpg'
y_path = 'H:/alphabets/y.jpg'
z_path = 'H:/alphabets/z.jpg'

#aphabet dictionaries
alphabets = {'a':a_path, 'b':b_path, 'c':c_path, 'd':d_path, 'e':e_path,
            'f':f_path, 'g':g_path, 'h':h_path, 'i':i_path, 'j':j_path,
            'k':k_path, 'l':l_path, 'm':m_path, 'n':n_path, 'o':o_path, 
            'p':p_path, 'q':q_path, 'r':r_path, 's':s_path, 't':t_path,
            'u':u_path, 'v':v_path, 'w':w_path, 'x':x_path, 'y':y_path,
            'z':z_path}


def stackImages(imgArray,scale,lables=[]):
    sizeH = imgArray[0][0].shape[0]
    sizeW= imgArray[0][0].shape[1]
    
    rows = len(imgArray)
    cols = len(imgArray[0])

    #added to display images if num of images in each column differs
    for i in range(1,rows):
        cols_nxt_row = len(imgArray[i])
        if cols_nxt_row > cols:
            cols = cols_nxt_row

    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
        
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver



#success, img = cap.read()
kernel = np.ones((5,5),np.uint8)
#print(kernel)
imgBlank = np.zeros((200,200),np.uint8)
    
'''
# make a list of images of all letters in a word
# make list of images of every words.
'''

# At Maximum 8 word can be displayed.
text = 'This implementation.'
text = text.lower()

#Count the number of words
word_num = text.count(' ') + 1

list1, list2, list3, list4, list5 = [],[],[],[],[]
list6, list7, list8, list9, list10 = [],[],[],[],[]

empty_lists = [list1, list2, list3, list4, list5, list6, list7, list8, list9, list10]
main_list = []
list_len = []
    
#create empty list with number equal to the word_num
#empty_lists[:word_num]
    
splited_to_sentence = text.split('.')

for sentence in splited_to_sentence:
    splited_to_words = sentence.split( )        #SPLIT EACH SENTENCE TO LIST OF WORDS
    for list_name,word in zip(empty_lists[:word_num],splited_to_words): 
        for letter in word:
            img = cv2.imread(alphabets[letter])
            list_name.append(img)               # Appends image of letters in a word to single list 
        main_list.append(list_name)             # Appends list of words to main list
        list_len.append(len(list_name))         # List appends length of each words 
    #print(main_list)

#print([len(lst) for lst in main_list])

len_main = len(main_list)                
max_len = max(list_len)
#print('len ' + str(len(main_list)))


#better logic           
for list_name, i in zip(main_list,range(len_main)):
    if list_len[i] < max_len:
        diff = max_len - list_len[i]
        for j in range(diff):
            list_name.append(imgBlank)
                
#print([len(lst) for lst in main_list])

div, mod = int(len_main/7), len_main%7

#print(div, mod)

rate = 7    #controls the number of rows of imagaes to be shown in one window 
mod_copy = mod
index_3 = len_main - mod
for i in range(div+1,0,-1):
    if mod != 0:
        win_name = 'Gestures_{}'.format(i+1)
        index = len_main - mod
        StackedImages = stackImages(([lst for lst in main_list[index:]]),0.6)
        cv2.imshow(win_name, StackedImages)
        mod = 0
        continue
    win_name = 'Gestures_{}'.format(i)
    index_2 = len_main - mod_copy - rate
    StackedImages = stackImages(([lst for lst in main_list[index_2:index_3]]),0.6)
    cv2.imshow(win_name, StackedImages)
    rate += 7
    index_3 = index_2

'''
if len(main_list) > 8:
    StackedImages = stackImages(([lst for lst in main_list[8:]]),0.6)
    cv2.imshow("Corresponding hand gesture 2.", StackedImages)

StackedImages = stackImages(([lst for lst in main_list[:8]]),0.6)
cv2.imshow("Corresponding hand gesture 1.", StackedImages)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()


#test_image set
'''
path = "H:/alphabets/a.jpg"
img =  cv2.imread(path)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny = cv2.Canny(imgBlur,100,200)
imgDilation = cv2.dilate(imgCanny,kernel , iterations = 2)
imgEroded = cv2.erode(imgDilation,kernel,iterations=2)
'''

#test code
'''
lst1 = [img,img,img]
lst2 = [img,img,img,img]
lst = [lst1,lst2]


lst_len = [len(lst1),len(lst2)]
max_len = max(lst_len)
print(len(lst[0]))

for i in range(len(lst_len)):
    if lst_len[i] < max_len:
        diff = max_len - lst_len[i]
        for j in range(diff):
            lst[i].append(imgBlank)
print(len(lst[0]))
'''
