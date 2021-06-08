from synflowmap import SynFlowMap
import cv2
import os
import time

def nothing(val):
    pass
originalName='data/woman.mp4'
magnifyedNameLow='data/PhaseBasedResults/woman-FIRWindowBP-band0.35-0.71-sr60-alpha15-mp1-sigma3-scale1.00-frames1-600-halfOctave.mp4'
magnifyedNameMid='data/PhaseBasedResults/woman-FIRWindowBP-band1.00-1.90-sr60-alpha15-mp1-sigma3-scale1.00-frames1-600-halfOctave.mp4'
magnifyedNameHigh='data/PhaseBasedResults/woman-FIRWindowBP-band3.00-6.00-sr60-alpha15-mp1-sigma3-scale1.00-frames1-600-halfOctave.mp4'
ovcap=cv2.VideoCapture(originalName)
f1vcap=cv2.VideoCapture(magnifyedNameLow)
f2vcap=cv2.VideoCapture(magnifyedNameMid)
f3vcap=cv2.VideoCapture(magnifyedNameHigh)
alpha=1
scale=2
cv2.namedWindow('multi frequency',cv2.WND_PROP_FULLSCREEN)
cv2.createTrackbar('alpha f1','multi frequency',200,400,nothing)
cv2.createTrackbar('alpha f2','multi frequency',200,400,nothing)
cv2.createTrackbar('alpha f3','multi frequency',200,400,nothing)
cv2.createTrackbar('scale','multi frequency',0,4,nothing)
sfm=SynFlowMap()

while ovcap.isOpened() and f1vcap.isOpened() and f2vcap.isOpened() and f3vcap.isOpened():
    lastFrameTime = time.time()    

    ret0, oframe = ovcap.read()#OS: original size
    ret1, f1frame = f1vcap.read()
    ret2, f2frame = f2vcap.read()
    ret3, f3frame = f3vcap.read()

    if not ret0 or not ret1 or not ret2 or not ret3:
        break
    
    a1=(cv2.getTrackbarPos('alpha f1','multi frequency')-200)/100
    a2=(cv2.getTrackbarPos('alpha f2','multi frequency')-200)/100
    a3=(cv2.getTrackbarPos('alpha f3','multi frequency')-200)/100
    sfm.scale=2**cv2.getTrackbarPos('scale','multi frequency')

    alphas=[a1,a2,a3]
    mags=[f1frame,f2frame,f3frame]

    orig_proc=sfm.preprocessFrame(oframe)
    flows=[]
    for i in range(len(mags)):
        mag_proc=sfm.preprocessFrame(mags[i])
        flows.append(sfm.calculateFlow(orig_proc,mag_proc,1))
    flow=sfm.combineFlows(flows,alphas)  
    result=sfm.remap(oframe,flow)

    thisFrameTime = time.time()
    
    fps=1/(thisFrameTime-lastFrameTime)
    
    cv2.putText(oframe,"Original",(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(f1frame,"Frequency 1",(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(f2frame,"Frequency 2",(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(f3frame,"Frequency 3",(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(result,"Result, {:2.1f} fps".format(fps),(3,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)

    hh,wh=oframe.shape[0]//2,oframe.shape[1]//2
    cv2.resize(oframe,(wh,hh))
    toDisplay=cv2.hconcat([cv2.resize(oframe,(wh,hh)),cv2.resize(f1frame,(wh,hh))])
    toDisplay=cv2.vconcat([toDisplay,cv2.hconcat([cv2.resize(f2frame,(wh,hh)),cv2.resize(f3frame,(wh,hh))])])
    toDisplay=cv2.hconcat([toDisplay,result])
    
    cv2.imshow('multi frequency',toDisplay)
    #if cv2.waitKey(int(1000/oFps)) & 0xFF == ord('q'):
    key=cv2.waitKey(int(1)) & 0xFF
    if key == ord('q'):
        break
    
