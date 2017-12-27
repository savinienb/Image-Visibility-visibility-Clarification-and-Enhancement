#coded with the help of ://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python/11449901#11449901
import cv2
import time
import numpy as np

import sys
sys.path.insert(0, '/home/sav/Desktop/CODE_BACK/codes')
import haze
import dehaze

def print_details(t,gpu,model):

    print '                             '
    print '------------------------'
    print 'gpu= ', gpu
    print 'cnn model= ', model
    print 'fps= ', np.uint8(1/t)
    print 'dehaze time= ', t,'s'
    print '------------------------'#
    print '                             '


cv2.namedWindow("cleaned_image")
cv2.namedWindow("original_image")
cv2.namedWindow("hazy_image")
vc = cv2.VideoCapture(0)

#model parameter
template_dir='template/'
model=['DeMisty_FI','DeMisty_RI']
templateFile=[]
caffe_model=[]
for mod in range(len(model)):
    templateFile.append(template_dir+model[mod]+'_template.prototxt')
    caffe_model.append(model[mod]+'.caffemodel')


gpu=0
mod=0

#haze parameter out=in*e^(beta)+A(1-e^(beta))
A=0.6 #0.6 1
beta=1.8#0.2 2.5
#var perlin #0.1 0.6
haze_noise=0


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    dh=dehaze.Dehaze(frame,templateFile[mod],template_dir,caffe_model[mod],gpu)
    perlin_haze=np.ones(frame.shape)*1.0
else:
    rval = False

while rval:

    frame=1.0*frame/255
    hazy_frame=haze.perlin_haze(frame,perlin_haze,beta,A,haze_noise)
   
    #hazy_frame=haze.haze(frame,beta,A,haze_noise)

    a=time.time()
    cleaned_frame=dh.dehaze(hazy_frame)
    t=time.time()-a 

    print_details(t,gpu,model[mod])

    cv2.imshow("original_image", frame)
    cv2.imshow("hazy_image", hazy_frame)
    cv2.imshow("cleaned_image", cleaned_frame)

    rval, frame = vc.read()
    key = cv2.waitKey(20)

    if key == 1048675: # C
        perlin_haze=np.ones(frame.shape)
    if key == 1048686: # N
        perlin_haze=haze.perlin_gen(frame)
    if key == 1113938: # UP
        haze_noise=abs(1-haze_noise)
    if key == 27: # exit on ESC
        break
    if key == 1114082: # SHIFT
        mod+=1
        if mod>=len(caffe_model):
            mod=0
        dh=dehaze.Dehaze(frame,templateFile[mod],template_dir,caffe_model[mod],gpu)
    if key == 1113603: # altgr
        gpu=abs(1-gpu)
        dh=dehaze.Dehaze(frame,templateFile[mod],gpu)


cv2.destroyWindow("original_image")
cv2.destroyWindow("hazy_image")
cv2.destroyWindow("cleaned_image")
vc.release()