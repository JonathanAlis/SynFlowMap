import cv2
import numpy as np
import math
import time
import os
import re

datapath='./momag-database/resultVideos/moveOnly2_downsize_4/'
tmpdir='tmp'
resultsPaths=['./data/deepmag/momag-database/moveOnly2_downsize_4/',
              './data/deepmag/momag-database/moveOnly2_downsize_4_halfMag/']
for resultsPath in resultsPaths:
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    def getNum(str):
        numbers = re.findall('[0-9]+', str)
        return int(numbers[0])    
    def getAlpha(str):
        numbers = re.findall('[0-9]+', str)
        return int(numbers[-2])

    videos=os.listdir(datapath)
    orig=[v for v in videos if v.startswith('orig')]
    orig.sort(key=getNum)
    mag=[v for v in videos if v.startswith('mag')]
    mag.sort(key=getNum)
    dyn=False
    for i in range(len(orig)):
        print("salvando frames:",orig[i])
        status='dynamic'
        v=datapath+orig[i]
        frameCount=0
        vCap=cv2.VideoCapture(v)
        vw= int(vCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh= int(vCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps= (vCap.get(cv2.CAP_PROP_FPS))    
        while(vCap.isOpened()):
            ret1, vFrame = vCap.read()
            frameCount=frameCount+1
            if not ret1:            
                break
            tmppath="./deep_motion_mag-master/data/vids/mag"+str(i)+"/"
            if not os.path.exists(tmppath):
                os.makedirs(tmppath)
            imname=tmppath+'{:06d}'.format(frameCount)+'.png'
            cv2.imwrite(imname,vFrame)
            #print('Saving frame '+str(frameCount)+' to '+imname)
        
        print("processando videos")
        alpha=getAlpha(mag[i])
        
        if resultsPath.find('halfMag')>=0:
            alpha//=2
    #FLAGS="--phase=run --vid_dir=$VID_DIR --out_dir=$OUT_DIR --amplification_factor=$AMPLIFICATION_FACTOR"
    #if [ "$DYNAMIC_MODE" = yes ] ; then
    #    FLAGS="$FLAGS"" --velocity_mag"
    #fi
    #pip3 install setproctitle
    #python3 main.py --config_file=configs/"$EXP_NAME".conf \
    #    $FLAGS
        flags='--phase=run --vid_dir={} --out_dir={} --amplification_factor={}'.format(tmpdir,tmpdir,alpha)
        if dyn:
            flags+=' --velocity_mag'
        command='sh run_on_test_videos.sh o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds '+'mag'+str(i)+' '+str(alpha)+' '+('yes ' if v[3] == 'dynamic' else 'no ')+flags
        print(command)    

        status=os.system('cd deep_motion_mag-master && pwd && '+command)  
        #256: error saving the frames
        if status==256:
            frames_folder='./deep_motion_mag-master/data/output/mag'+str(i)+'/'            
            out_name=resultsPath+orig[i][:-4]+'_alpha'+str(alpha)+'_notdynamic.mp4'
            frame=frames_folder+'{:06d}.png'.format(1)
            print(frame)
            img = cv2.imread(frame)
            writer = cv2.VideoWriter(out_name,cv2.VideoWriter_fourcc(*'FMP4'), 30,(img.shape[1],img.shape[0]))#MJPG 4.3Mb, H264 didnt save, FMP4,MP4V,AVC1...

            for fn in range(90):
                frame=frames_folder+'{:06d}.png'.format(fn+1)
                frame = cv2.imread(frame)
                writer.write(frame)   
            writer.release()


