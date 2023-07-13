import cv2
import numpy as np
import math
import time
from moviepy.editor import VideoFileClip,ImageSequenceClip

class manage_videos:
    def __init__(self,original_file,mag_file,scale=1,bilatfil_size=1,alpha=1,winsize=15):
        self.sfm=SynFlowMap(scale,bilatfil_size,alpha,winsize)
        self.orig_file=original_file
        self.mag_file=mag_file
        self.currentOrig=None
        self.currentMag=None
        self.currentFlow=None
        self.currentProposed=None
        self.started=False

    def start(self,verbose=True):
        self.originalCap=cv2.VideoCapture(self.orig_file)
        self.magnifyedCap=cv2.VideoCapture(self.mag_file)
        #raise error if cant open files
        self.frame_count=0
        oWidth= int(self.originalCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oHeight= int(self.originalCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        oFps=  int(self.originalCap.get(cv2.CAP_PROP_FPS))
        mWidth= int(self.magnifyedCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        mHeight= int(self.magnifyedCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mFps=  int(self.magnifyedCap.get(cv2.CAP_PROP_FPS))
        if verbose:
            print('starting video')
            print('original video size ',oWidth,'x',oHeight,', at ',oFps,' fps.')
            print('magnifyed video size ',mWidth,'x',mHeight,', at ',mFps,' fps.')
        
        #cv2.namedWindow('proposed',cv2.WND_PROP_FULLSCREEN)
        self.started=True
        return True

    def nextFrame(self):
        if not self.originalCap.isOpened() and not self.magnifyedCap.isOpened():
            print('error: start first the video')
            return        
        #lastFrameTime = thisFrameTime
        ret1, self.currentOrig = self.originalCap.read()#OS: original size
        ret2, self.currentMag = self.magnifyedCap.read()
        if not ret1 or not ret2:
            return (False,np.zeros((100,100,3), np.uint8))
        self.currentOrig=cv2.resize(self.currentOrig,(self.currentMag.shape[1],self.currentMag.shape[0]))
        
        orig_proc=self.sfm.preprocessFrame(self.currentOrig)
        mag_proc=self.sfm.preprocessFrame(self.currentMag)
        self.currentFlow=self.sfm.calculateFlow(orig_proc,mag_proc)
        result=self.sfm.remap(self.currentOrig,self.currentFlow)       
        self.frame_count=self.frame_count+1
        return (True,result)
    def has_started(self):
        return self.started
    def has_finished(self):        
        return not self.originalCap.isOpened() or not self.magnifyedCap.isOpened()
        


class SynFlowMap:
    

    def __init__(self,scale=1,bilatfil_size=1,alpha=1,winsize=15):
        self.scale_range=[1,2,4,8]
        self.bilatfil_size_range=[1,3,5,7,9,11,13,15]
        self.alpha_limits=(-3,3)
        self.winsize_range=[3,5,7,9,11,13,15,17,19,21,23,25]
        self._scale=scale #1 2 4 8        
        self._bilatfil_size=1 #1 21?
        self._alpha=alpha #-2 2
        self._winsize=winsize #3 25
        self.h1=None
        self.w1=None


    @staticmethod
    def get_closest(value, list):
        diff = lambda x : abs(x - value)
        ret=min(list, key=diff)
        #print('Received',value,', setting as ',ret)
        return ret

    @staticmethod
    def limit(value, limits):
        if value> limits[1]:
            ret= limits[1]
        elif value < limits[0]:
            ret=limits[0]
        else:
            ret =value
        #print('Received',value,', setting as ',ret)
        return ret

    @property
    def scale(self):
        return self._scale
    @scale.setter    
    def scale(self, value):        
        self._scale = SynFlowMap.get_closest(value,self.scale_range)
        
    @property
    def bilatfil_size(self):
        return self._bilatfil_size    
    @bilatfil_size.setter    
    def bilatfil_size(self, value):        
        self._bilatfil_size = SynFlowMap.get_closest(value,self.bilatfil_size_range)

    @property
    def alpha(self):
        return self._alpha    
    @alpha.setter    
    def alpha(self, value):        
        self._alpha = SynFlowMap.limit(value,self.alpha_limits)

    @property
    def winsize(self):
        return self._winsize    
    @winsize.setter    
    def winsize(self, value):        
        self._winsize = SynFlowMap.get_closest(value,self.winsize_range)

    @staticmethod
    def drawFlow(img, flow, step=8):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        #print(img.shape)
        vis = img.copy()#cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img, lines, 0, (255, 255, 0))
        #for (x1, y1), (x2, y2) in lines:
            #cv2.circle(vis, (x1, y1), 1, (255, 255, 0), -1)
        return vis

    def preprocessFrame(self,frame):        
        self.h1=frame.shape[0]
        self.w1=frame.shape[1]
        self.h=int(self.h1/self._scale)
        self.w=int(self.w1/self._scale)
        frame=cv2.resize(frame,(self.w,self.h))
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def calculateFlow(self,orig,mag,alpha=None):
        if alpha is not None:
            self._alpha=alpha
        flow = cv2.calcOpticalFlowFarneback(orig,mag, None, 0.5, 3, self._winsize, 3, 5, 1.2, 0)*-self._alpha
        if self._bilatfil_size==1:
            flow[:,:,0] = self._scale*flow[:,:,0]
            flow[:,:,1] = self._scale*flow[:,:,1]
        else:
            flow[:,:,0] = self._scale*cv2.bilateralFilter(flow[:,:,0], self._bilatfil_size,75,75)
            flow[:,:,1] = self._scale*cv2.bilateralFilter(flow[:,:,1], self._bilatfil_size,75,75)
        
        flowOS=np.zeros((self.h1,self.w1,2),dtype='float32')
        flowOS[:,:,0]=cv2.resize(flow[:,:,0],(self.w1,self.h1))
        flowOS[:,:,1]=cv2.resize(flow[:,:,1],(self.w1,self.h1))
        self._flow=flowOS
        return flowOS
        #flowView=draw_flow(cv2.cvtColor(oFrameOS, cv2.COLOR_BGR2GRAY),flowOS)
    def combineFlows(self,flows,alphas):
        if not isinstance(flows, list) or not isinstance(alphas, list):
            raise Exception('flows and alphas must be lists')
        if len(flows) != len(alphas):
            raise Exception('flows and alphas must have same size')
        flow=np.zeros((self.h1,self.w1,2),dtype='float32')
        for i in range(len(flows)):
            flow+=alphas[i]*flows[i]
        return flow

    def nullMap(self,h,w):        
        self._map_x = np.zeros((h,w),np.float32)
        self._map_y = np.zeros((h,w),np.float32)
        for i in range(0,h):
            for j in range(0,w):#
                self._map_x[i,j] = j
                self._map_y[i,j] = i
        return 

    def remap(self,frame,flow):
        if frame.shape[0]!=flow.shape[0] and frame.shape[1]!=flow.shape[1]:
            raise Exception('frame and flow must have same dimensions')
        self.h1=frame.shape[0]
        self.w1=frame.shape[1]
        calcNullMap=True
        if hasattr(self,'_map_x') and hasattr(self,'_map_y'):
            if self._map_x.shape[0]==self.h1 and self._map_y.shape[0]==self.w1:
                calcNullMap=False
        if calcNullMap:
            self.nullMap(self.h1,self.w1)

        proposedFrame=np.empty(((self.h1,self.w1,3)))
        proposedFrame=cv2.remap( frame, self._map_x+flow[:,:,0],self._map_y+flow[:,:,1], cv2.INTER_LINEAR,proposedFrame,cv2.BORDER_REPLICATE)
        #((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) && map2.empty()) || (map1.type() == CV_32FC1 && map2.type() == CV_32FC1) in function 'remap'
        return proposedFrame

    def singleFlowMap(self,orig,mag,alpha=None):
        if alpha is None:
            self._alpha=alpha
        orig_proc=self.preprocessFrame(orig)
        mag_proc=self.preprocessFrame(mag)
        flow=self.calculateFlow(orig_proc,mag_proc,self._alpha)
        result=remap(orig_proc,flow)
        return result

    def multiFlowMap(self,orig,mags,alphas=None):
        if alpha is None:
            alphas=[]
            for i in range(len(mags)):
                alphas.append(self._alpha)
        orig_proc=preprocessFrame(self,orig)
        flows=[]
        for i in range(len(mags)):
            mag_proc=self.preprocessFrame(mags[i])
            flows.append(self.calculateFlow(orig_proc,mag_proc,1))
        flow=self.combineFlows(flows,alphas)  
        result=self.remap(orig_proc,flow)
        return result



def flowSeg(flowX,flowY):
    #otsu only works in uint8, bcos uses histogram
    flowXY=np.sqrt(np.square(flowX)+np.square(flowY))
    #flowuint8=flowXY/np.max(flowXY)*255
    flowuint8=flowXY.astype(np.uint8)
    r,otsu=cv2.threshold(flowuint8,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th3 = cv2.adaptiveThreshold(flowuint8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    cv2.imshow('otsu',otsu*255)


def nothing(val):
    pass

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (255, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        pass
        #cv2.circle(vis, (x1, y1), 1, (255, 255, 0), -1)
    return vis

def printInstructions():
    print('\nProposed Motion magnification method\n')
    print('Instructions:')
    print('p->pause')
    print('i->invert motion')
    print('r->restart')
    print('mouse clicking in first image->select a slice to acompany along the time')
    print('s->save video and slice time images (if selected)')
    print('h->show these instructions')

def lineToFolow(event,x,y,flags,param):
    global ix,iy,fx,fy,drawing, restart
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy=x,y
        fx,fy=x,y
        drawing=True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            fx,fy=x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        restart=True
        fx,fy=x,y
    if abs(fx-ix)>abs(fy-iy): #keep y
        fy=iy
    else:
        fx=ix

    
def drawLine(img,ix,iy,fx,fy,drawing):
    if fx>0 and fy>0:
        if drawing:
            cv2.line(img,(ix,iy),(fx,fy),(0,0,255),5)            
        if not drawing:
            cv2.line(img,(ix,iy),(fx,fy),(255,0,0),5)            
       



def gui():
    vNames=[('baby','data/baby.mp4','data/PhaseBasedResults/baby-differenceOfIIR-band0.04-0.40-sr1-alpha20-mp0-sigma5-scale0.80-frames1-301-quarterOctave.mp4',20, 'phase-based'),\
        ('baby','data/baby.mp4','data/riez/baby.mp4',20, 'riez-pyramids'),\
        ('cat_toy','data/cat_toy.mp4','data/VideoAccelerationResults/cat_toy_fm_4_alpha_8_pylevel_4_kernel_DOG.mp4',8,'acceleration magnification'),\
        ('cat_toy2','data/cat_toy.mp4','data/deepmag/cat_toy_7_dynamic.mp4',7, 'deepmag dynamic'),\
        ('cat_toy3','data/cat_toy.mp4','data/deepmag/cat_toy_8_dynamic.mp4',8, 'deepmag dynamic'),\
        ('gunshot3','data/gun_shot.mp4','data/deepmag/gunshot1_8_dynamic.mp4',8, 'deepmag dynamic'),\
        ('subway1','data/subway.mp4','data/VideoAccelerationResults/subway_fm_4_alpha_8_pylevel_4_kernel_INT.mp4',8,'acceleration magnification'),\
        ('subway2','data/subway.mp4','data/PhaseBasedResults/subway-FIRWindowBP-band3.60-6.20-sr30-alpha60-mp0-sigma3-scale1.00-frames1-243-octave.mp4',60, 'phase-based'),\
        ('subway3','data/subway.mp4','data/deepmag/subway1_8_static.mp4',8,'deepmag static'),\
        ('subway4','data/subway.mp4','data/deepmag/subway2_8_dynamic.mp4',8, 'deepmag dynamic'),\
        ('drum','data/drum.avi','data/deepmag/drum_10_static.mp4',10, 'deepmag static'),\
        ('balance1','data/balance.mp4','data/riez/balance.mp4',10, 'riez pyramids'),\
        ('balance2','data/balance.mp4','data/deepmag/balance_10_1_8_300_Butter.mp4',10, 'deepmag static'),\
        ('car_engine','data/car_engine.mp4','data/PhaseBasedResults/car_engine-FIRWindowBP-band15.00-25.00-sr400-alpha30-mp0-sigma3-scale1.00-frames1-300-octave.mp4',30, 'phase-based'),\
        ('throat','data/throat.mp4','data/PhaseBasedResults/throat-FIRWindowBP-band90.00-110.00-sr1900-alpha100-mp0-sigma3-scale0.67-frames1-300-octave.avi',100, 'phase-based'),\
        ('crane_crop1','data/crane_crop.mp4','data/PhaseBasedResults/crane_crop-FIRWindowBP-band0.20-0.25-sr24-alpha75-mp0-sigma5-scale1.00-frames1-215-octave.mp4',75, 'phase-based'),\
        ('crane_crop2','data/crane_crop.mp4','data/PhaseBasedResults/crane_crop-FIRWindowBP-band0.20-0.25-sr24-alpha75-mp0-sigma5-scale1.00-frames1-215-halfOctave.mp4',75, 'phase-based'),\
        ('crane_crop3','data/crane_crop.mp4','data/PhaseBasedResults/crane_crop-FIRWindowBP-band0.20-0.25-sr24-alpha75-mp0-sigma5-scale1.00-frames1-215-quarterOctave.mp4',75, 'phase-based'),\
        ('crane_crop4','data/crane_crop.mp4','data/riez/crane_crop.mp4',75, 'riez-pyramids'),\
        ('eye','data/eye.mp4','data/PhaseBasedResults/eye-FIRWindowBP-band30.00-50.00-sr500-alpha75-mp0-sigma4-scale0.40-frames1-600-octave.avi',75, 'phase-based'),\
        ('bottle_moving','data/bottle_moving.mp4','data/VideoAccelerationResults/bottle_moving_fm_4_alpha_8_pylevel_4_kernel_INT.mp4',8,'acceleration magnification'),\
        ('bottle_moving2','data/bottle_moving.mp4','data/deepmag/bottle_moving_8_dynamic.mp4',8, 'deepmag dynamic'),\
        ('drone','data/drone_raw.mp4','data/PhaseBasedResults/drone_raw-differenceOfButterworths-band3.00-7.00-sr30-alpha5-mp0-sigma3-scale1.00-frames1-451-halfOctave.mp4',5, 'phase-based'),\
        ('drone','data/drone_raw.mp4','data/deepmag/drone_10_dynamic.mp4',10, 'deepmag dynamic'),\
        ('womanf1','data/woman.mp4','data/PhaseBasedResults/woman-FIRWindowBP-band0.35-0.71-sr60-alpha15-mp1-sigma3-scale1.00-frames1-600-halfOctave.mp4',15, 'phase-based'),\
        ('womanf2','data/woman.mp4','data/PhaseBasedResults/woman-FIRWindowBP-band1.00-1.90-sr60-alpha15-mp1-sigma3-scale1.00-frames1-600-halfOctave.mp4',15, 'phase-based'),\
        ('womanf3','data/woman.mp4','data/PhaseBasedResults/woman-FIRWindowBP-band3.00-6.00-sr60-alpha15-mp1-sigma3-scale1.00-frames1-600-halfOctave.mp4',15, 'phase-based'),\
        ('basballOnCity','data/basballOnCity.mp4','data/PhaseBasedResults/basballOnCity-FIRWindowBP-band0.10-10.00-sr30-alpha100-mp0-sigma5-scale1.00-frames1-150-halfOctave.mp4',100, 'phase-based'),\
        ('basballOnBlack','data/basballOnBlack.mp4','data/PhaseBasedResults/basballOnBlack-FIRWindowBP-band0.10-10.00-sr30-alpha100-mp0-sigma5-scale1.00-frames1-150-halfOctave.mp4',100, 'phase-based'),\
    ]    
    vNames = [v for v in vNames if os.path.isfile(v[1]) and os.path.isfile(v[2])]

    #print(vNames)
    alphaDefault=1
    alpha=alphaDefault
    invert=1
    scaleDefault=1
    scale=scaleDefault
    ix,iy,fx,fy=0,0,0,0
    fpsSum=0
    OF_pos_default=6
    scale_pos_default=0
    alpham_pos_default=100
    bifilt_pos_default=0
    isSaving=False
    alpha_pos=alpham_pos_default
    scale_pos=scale_pos_default
    bilfilt_pos=bifilt_pos_default
    OF_pos=OF_pos_default
    #buttons: restart, next video, invert alpha, save image, save result video(next iteration), save comparison
    v=0
    while True:
        printInstructions()
        manager=manage_videos(vNames[v][1],vNames[v][2])
        manager.start()
        cv2.namedWindow('proposed',cv2.WND_PROP_FULLSCREEN)
        cv2.createTrackbar('OF window size','proposed',OF_pos,11,nothing)
        cv2.createTrackbar('scale','proposed',scale_pos,3,nothing)
        cv2.createTrackbar('alpha','proposed',alpha_pos,300,nothing)
        cv2.createTrackbar('bilfilter','proposed',bilfilt_pos,7,nothing)
        cv2.createTrackbar('video','proposed',v,len(vNames)-1,nothing)
        param=[]
        cv2.setMouseCallback('proposed',lineToFolow,param)

        drawing=False
        restart=True
        paused=False
        imgs=[]
        print('processing videos',vNames[v][1],vNames[v][2])
        while(not manager.has_finished()):
            if not paused:
                lastFrameTime = time.time()    
                alpha_pos=cv2.getTrackbarPos('alpha','proposed')
                scale_pos=cv2.getTrackbarPos('scale','proposed')
                bilfilt_pos=cv2.getTrackbarPos('bilfilter','proposed')
                OF_pos=cv2.getTrackbarPos('OF window size','proposed')
                manager.sfm.alpha=alpha_pos/100*invert
                manager.sfm.scale=2**scale_pos
                manager.sfm.bilatfil_size=bilfilt_pos*2+1
                manager.sfm.winsize=OF_pos*2+3
                
                video_num=cv2.getTrackbarPos('video','proposed')
                if cv2.getTrackbarPos('video','proposed')!=v:
                    print(v)
                    v=cv2.getTrackbarPos('video','proposed')
                    print(v)
                    break
                ret,frame=manager.nextFrame()
                if not ret:
                    if isSaving:
                        clip = ImageSequenceClip(imgs, fps=int(manager.originalCap.get(cv2.CAP_PROP_FPS)))
                        clip.write_videofile(vidName)
                        print('         Done. ')
                        print('saving video as', vidName)
                        isSaving=False
                    break
                thisFrameTime = time.time()
                
                fps=1/(thisFrameTime-lastFrameTime)
                orig=manager.currentOrig
                flow=manager.sfm.drawFlow(orig,manager.currentFlow)
                cv2.putText(flow,"Original "+manager.orig_file,(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.putText(orig,"Optical Flow",(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)

                mag=manager.currentMag
                cv2.putText(mag,f"Eulerian Magnification",(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.putText(mag,f"Method: {vNames[v][4]}",(3,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.putText(mag,f"alpha: {vNames[v][3]}",(3,75), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                
                cv2.putText(frame,f"SynFlowMap (ours)",(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.putText(frame,f"alpha:{vNames[v][3]*manager.sfm.alpha}",(3,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.putText(frame,f"scale:{manager.sfm.scale}",(3,75), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                
                
                
                toDisplay=cv2.hconcat([flow,mag])
                toDisplay=cv2.vconcat([toDisplay,cv2.hconcat([orig,frame])])
                
                #cv2.putText(toDisplay,"{:2.1f} fps".format(fps)+", frame n "+str(manager.frame_count),(400,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.imshow('proposed',toDisplay)
                #if cv2.waitKey(int(1000/oFps)) & 0xFF == ord('q'):
                if isSaving:
                    imgs.append(cv2.cvtColor(toDisplay, cv2.COLOR_BGR2RGB))


            key=cv2.waitKey(int(1)) & 0xFF
            if key == ord('q'):
                return
            
            elif key  == ord('r'):
                print('restarting')
                print(alpham_pos_default)
                alpha_pos=alpham_pos_default
                scale_pos=scale_pos_default
                bilfilt_pos=bifilt_pos_default
                OF_pos=OF_pos_default
                cv2.namedWindow('proposed',cv2.WND_PROP_FULLSCREEN)
                cv2.createTrackbar('OF window size','proposed',OF_pos,11,nothing)
                cv2.createTrackbar('scale','proposed',scale_pos,3,nothing)
                print(alpha_pos)
                cv2.createTrackbar('alpha','proposed',alpha_pos,300,nothing)
                cv2.createTrackbar('bilfilter','proposed',bilfilt_pos,7,nothing)
                cv2.createTrackbar('video','proposed',v,len(vNames)-1,nothing)
                invert=1
                manager.start()
                frameCount=0
                fpsSum=0
                continue
            elif key  == ord('i'):
                invert=-invert
            elif key == ord('s'):
                if not paused:
                    print('restarting and saving the result...')
                    manager.start()
                    imgs=[]
                    vidName='data/proposedResults/SFM_'+vNames[v][0]
                    if alpha!=alphaDefault:
                        vidName=vidName+'_alpha'+str(manager.sfm.alpha)
                    if scale!=scaleDefault:
                        vidName=vidName+'_scale'+str(manager.sfm.scale)  
                    #    manager.sfm.bilatfil_size                  
                    #manager.sfm.winsize
                    vidName=vidName+'.mp4'
                    
                    isSaving=True
                    restart=True
                    continue

            elif key == ord('p'):
                paused=not paused
            elif key == ord('h'):
                printInstructions()

            
                
if __name__ == "__main__":
    import os
    gui()