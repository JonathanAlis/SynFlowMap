import cv2
import numpy as np
import math
import time
import os
import re
from moviepy.editor import VideoFileClip,ImageSequenceClip

def getNum(str):
    numbers = re.findall('[0-9]+', str)
    return int(numbers[0])    
def getAlpha(str):
    numbers = re.findall('[0-9]+', str)
    return int(numbers[-2])

datapath='./momag-database/resultVideos/moveHalfLarge_downsize_4/'
tmpdir='tmp'
resultsPaths=['./data/VideoAccelerationResults/momag-database/large_motion/',
              './data/VideoAccelerationResults/momag-database/large_motion_halfmag/'
              ]
for resultsPath in resultsPaths:
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)


    videos=os.listdir(datapath)

    orig=[v for v in videos if v.startswith('orig')]
    orig.sort(key=getNum)
    mag=[v for v in videos if v.startswith('mag')]
    mag.sort(key=getNum)
    dyn=False
    for i in range(len(orig)):
        #orig21_plainbg_moveXY_amp2.0_Large_fm_2_alpha_4_pylevel_4_kernel_INT
        imgpath=resultsPath+orig[i][:-4]+'/im_write/'
        print("frames path:",imgpath)
        
        alpha=getAlpha(mag[i])
        
        if resultsPath.find('halfMag')>=0:
            alpha//=2
        #out_name=resultsPath+orig[i][:-4]+'_alpha'+str(alpha)+'_notdynamic.mp4'
        outfilename=resultsPath+orig[i][:-4]+'_fm_2_alpha_'+str(alpha)+'_pylevel_4_kernel_INT.mp4'
        print(f"Trying saving video {outfilename}.....")
        if os.path.exists(outfilename):
            print(outfilename,'already exists')
            video = cv2.VideoCapture(outfilename)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if num_frames==90:
                print('And it has 90 frames. Skipping')
                continue
        
        print('saving' ,outfilename)
        
        frame_exists=[]
        for fn in range(90):
            frame=f'{imgpath}fr{fn+1}.png'
            frame_exists.append(os.path.isfile(frame))
            if not frame_exists[-1]:
                print(f'    Warning: Frame {frame} not found. run generateAccelerationDatabase2.mat on MATLAB')    
                continue

        imgs=[]
        for fn in range(90):
            frame=f'{imgpath}fr{fn+1}.png'
            imgs.append(frame)        
        clip = ImageSequenceClip(imgs, fps=30)
        clip.write_videofile(outfilename)
        print('         Done. ')
        if os.path.exists(outfilename):
            pass
            #os.remove(outfilename[:-4]+'.avi')


