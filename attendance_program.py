import cv2
import numpy as npy
import face_recognition as face_rec
import os
import pyttsx3 as textspeech
from datetime import datetime

                                #isme abhi date according store krne h name aur runtime pr image runtime pr lene h

engine=textspeech.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

def resize(img,size):
    width=int(img.shape[1]*size)
    height=int(img.shape[0]*size)
    dimension=(width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)

path='sample_images'
studentimg=[]
studentName=[]
myList=os.listdir(path)
#print(myList)

for cl in myList:
    curImg=cv2.imread(f'{path}\{cl}')
    studentimg.append(curImg)
    studentName.append(os.path.splitext(cl)[0])


def finEncoding(images):
    encoding_list=[]
    for img in images:
        img=resize(img,0.50)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        encodeimg=face_rec.face_encodings(img)[0]
        encoding_list.append(encodeimg)
    return encoding_list


def markattendance(name):
    with open('attendance.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            now=datetime.now()
            timestr=now.strftime('%H: %M')
            f.writelines(f'\n{name}, {timestr}')
            engine.say('Welcome to class '+name)
            engine.runAndWait()


encode_list=finEncoding(studentimg)

vid=cv2.VideoCapture(0)

while True:
    success,frame=vid.read()
    smaller_frames=cv2.resize(frame,(0,0),None,0.25,0.25)

# we are finding the face in frame and encode it, in encoding we have to give the source and face location

    facesInframe= face_rec.face_locations(smaller_frames)
    encodefacesInframe= face_rec.face_encodings(smaller_frames,facesInframe)

# for loop for multiple faces in frame and we want to detect all by comparing with our original data

# we using zip bcoz we are using two values from two individual lists together
 
    for encodeFace, Faceloc in zip(encodefacesInframe,facesInframe) :
        matches=face_rec.compare_faces(encode_list,encodeFace)

        # facedis difference measure kar rha h ki kiska kiske encoding m kitna difference h

        facedis=face_rec.face_distance(encode_list,encodeFace)
        print(facedis)

        # jiska bhi minimum difference hoga wo( original sample ka encode) idhar index bn jayega

        matchindex=npy.argmin(facedis)

        if matches[matchindex]:
            name=studentName[matchindex].upper()
            y1,x2,y2,x1=Faceloc

            #we do multiplication with 4 becoz in smaller frames we reduce the size of 
            # frames to 0.25(1/4) so to restore it's size we do this
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.rectangle(frame,(x1,y2-25),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)



    cv2.imshow('video',frame)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()

