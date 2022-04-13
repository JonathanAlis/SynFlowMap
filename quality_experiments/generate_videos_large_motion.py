import cv2
import os
import re
import numpy as np
from moviepy.editor import VideoFileClip,ImageSequenceClip
import pandas as pd

def getNum(str):
    numbers = re.findall('[0-9]+', str)
    return int(numbers[0])  

def draw_text(frame, text, grid=(1,1), pos=(0,0), color=(255,255,255), thickness=2, size=1,):
    cellx=frame.shape[0]//grid[0]
    celly=frame.shape[1]//grid[1]
    x=cellx*pos[0]+30
    y=celly*pos[1]+50
    if x is not None and y is not None:
        cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness,cv2.LINE_AA)

def size_to_reshape(size, width = None, height = None):
    (h, w) = size
    if width is None and height is None:
        return size
    if width is None:
        r = height / float(h)
        return  (height,int(w * r))
    else:
        r = width / float(w)
        return  (int(h * r),width)

def videofiles(folder,startswith=None):
    videos=os.listdir(folder)
    if startswith:
        filenames=[v for v in videos if v.startswith(startswith)]
    else:
        filenames=[v for v in videos if v.endswith('.mp4')]

    filenames.sort(key=getNum)
    print(filenames)
    return [folder+filename for filename in filenames]
    

save_frames=[43, 50]
video_save_frames=[10]
biframes=[]
#if os.path.exists('videos/tmp_imgs')
os.makedirs('videos/tmp_imgs',exist_ok=True)
os.makedirs('videos/tmp_csvs',exist_ok=True)
os.makedirs('videos/frames',exist_ok=True)

mag_ratios=([16]*3+[8]*3+[4]*3)*3

landscape=True
savename="video/demo3.mp4"
#max_size=[616*2,920*3]
if landscape:
    max_size= [1080,1920]
    niw,nih=5,3
else:
    max_size= [1920,1080]
    niw,nih=3,5

wxh=max_size[1]/max_size[0]

print('max size:',max_size)
folder_original='../momag-database/resultVideos/moveHalfLarge_downsize_4/'

origf=videofiles(folder_original,startswith='orig')
gt=videofiles(folder_original,startswith='mag')

folders_acc=['../data/VideoAccelerationResults/momag-database/large_motion/',           
             '../data/VideoAccelerationResults/momag-database/large_motion_halfmag/']           
folders_results_acc=['../data/proposedResults/from_ACC/default/large_motion/',           
                 '../data/proposedResults/from_ACC/default/large_motion_halfmag/']
folders_best_acc=['../data/proposedResults/from_ACC/best/large_motion/',
              '../data/proposedResults/from_ACC/best/large_motion_halfmag/']

acc=videofiles(folders_acc[0],startswith='orig')
acchm=videofiles(folders_acc[1],startswith='orig')
acc_df=videofiles(folders_results_acc[0])
acchm_df=videofiles(folders_results_acc[1])
acc_best=videofiles(folders_best_acc[0])
acchm_best=videofiles(folders_best_acc[1])

folders_dmd=['../data/deepmag/momag-database/large_motion/',           
             '../data/deepmag/momag-database/large_motion_halfmag/']           
folders_results_dmd=['../data/proposedResults/from_DMD/default/large_motion/',           
                 '../data/proposedResults/from_DMD/default/large_motion_halfmag/']
folders_best_dmd=['../data/proposedResults/from_DMD/best/large_motion/',
              '../data/proposedResults/from_DMD/best/large_motion_halfmag/']
dmd=videofiles(folders_dmd[0])
dmdhm=videofiles(folders_dmd[1])
dmd_df=videofiles(folders_results_dmd[0])
dmdhm_df=videofiles(folders_results_dmd[1])
dmd_best=videofiles(folders_best_dmd[0])
dmdhm_best=videofiles(folders_best_dmd[1])

#num_frames = int(groudtruthCap.get(cv2.CAP_PROP_FRAME_COUNT))
#imsize= (int(groudtruthCap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#        int(groudtruthCap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 

cv2.namedWindow('results', cv2.WINDOW_NORMAL)          
cv2.namedWindow('output video', cv2.WINDOW_NORMAL)          
v_shapes=[]
im_counter=0
for i in range(len(origf)):
    
    captures=[]
    methodnames=[('Original','Original'), 
                 ('Ground Truth Magnification','Ground Truth'),
                 ('Acceleration magnification','ACC'),
                 ('DeepMag Dynamic mode','DMD'),
                 ('SynFlowMap from Acceleration: default','SFMD from ACC'),
                 ('SynFlowMap from DeepMag: default','SFMD from DMD'),
                 ('SynFlowMap from Acceleration: best','SFMB from ACC'),
                 ('SynFlowMap from DeepMag: best','SFMB from DMD'),
                 ('SynFlowMap from half-magnification Acc: default','SFMD2 from ACC'),
                 ('SynFlowMap from half-magnification DeepMag: default','SFMD2 from DMD'),
                 ('SynFlowMap from half-magnification Acc: best','SFMB2 from ACC'),
                 ('SynFlowMap from half-magnification DeepMag: best','SFMB2 from DMD'),
                 ]
                 
    captures=[]
    captures.append(cv2.VideoCapture(origf[i]))
    captures.append(cv2.VideoCapture(gt[i]))
    captures.append(cv2.VideoCapture(acc[i]))
    captures.append(cv2.VideoCapture(dmd[i]))

    captures.append(cv2.VideoCapture(acc_df[i]))
    captures.append(cv2.VideoCapture(dmd_df[i]))
    captures.append(cv2.VideoCapture(acc_best[i]))
    captures.append(cv2.VideoCapture(dmd_best[i]))

    captures.append(cv2.VideoCapture(acchm_df[i]))
    captures.append(cv2.VideoCapture(dmdhm_df[i]))
    captures.append(cv2.VideoCapture(acchm_best[i]))
    captures.append(cv2.VideoCapture(dmdhm_best[i]))
    
    clips=[]
    first=True
    while True:
        ok=True
        frames=[]
        for cap in captures:
            ret, frame = cap.read()

            ok=ok and ret
            frames.append(frame)

        if not ok:
            v_shapes.append(video_shape)
            print(f'v{i} complete, {frame_counter} frames')
            break

        if first:            
            frame_counter=0
            shape=frames[0].shape
            ratio=shape[1]*niw/(shape[0]*nih)
            if ratio>wxh:
                #reshape to match width
                lr=False
                video_shape=size_to_reshape((int(1.0*shape[0]*nih),shape[1]*niw),width=max_size[1])
                video_shape=[(video_shape[0]//2)*2,(video_shape[1]//2)*2]
                black_shape=[(max_size[0]-video_shape[0])//2,max_size[1]]                

            else:
                lr=True
                #reshape to match height
                video_shape=size_to_reshape((shape[0]*nih,int(1.0*shape[1]*niw)),height=max_size[0])
                video_shape=[(video_shape[0]//2)*2,(video_shape[1]//2)*2]
                black_shape=[max_size[0],(max_size[1]-video_shape[1])//2]                
            first=False
        if False:
            v_shapes.append(video_shape)
            break
        
        for idx,f in enumerate(frames):
            draw_text(f,methodnames[idx][1])
        
        frame_counter+=1
        
        if i in video_save_frames:
            if frame_counter in save_frames:                
                print(f"frame {frame_counter} of video {i}")
                for counter, f in enumerate(frames):
                    if frame_counter==save_frames[0]:
                        biframes.append(f)
                    if frame_counter==save_frames[1]:
                        horizontal_cut=True
                        if horizontal_cut:        
                            biframes[counter][f.shape[0]//2:,:,:]=f[f.shape[0]//2:,:,:]
                            cv2.line(biframes[counter], (0, f.shape[0]//2), (f.shape[1], f.shape[0]//2), (0, 0, 0), thickness=2)
                            
                        else:
                            biframes[counter][:,f.shape[1]//2,:]=f[:,f.shape[1]//2,:]
                            cv2.line(biframes[counter], (f.shape[1]//2,0), (f.shape[0]//2, f.shape[1]), (0, 0, 0), thickness=2)
                        #get slice
                        #draw_text(f,methodnames[idx][1])
                        #
                        cv2.imwrite(f"videos/frames/v{i}_frames{save_frames[0]}_{save_frames[1]}_{methodnames[counter][1].replace(' ','-')}.png", 
                                    biframes[counter])
                    frame_filename=f"v{i}_frame{frame_counter}_{methodnames[counter][1].replace(' ','-')}.png"
                    cv2.imwrite(f'videos/frames/{frame_filename}', f)

        info=np.zeros(frames[0].shape, np.uint8)
        draw_text(info, f'Magnification:{str(mag_ratios[i])}x')
        
        if landscape:
            filler=np.zeros(frames[0].shape, np.uint8)
            origs=cv2.hconcat([frames[0],frames[1],info,filler,filler])
            accs=cv2.hconcat([frames[2],frames[4],frames[6],frames[8],frames[10]])            
            dmds=cv2.hconcat([frames[3],frames[5],frames[7],frames[9],frames[11]])
            results=cv2.vconcat([origs,accs,dmds])
            
        else:
            filler=np.zeros(frames[0].shape, np.uint8)
            origs=cv2.vconcat([frames[0],frames[1],filler,filler,filler])
            accs=cv2.vconcat([frames[2],frames[4],frames[6],frames[8],frames[10]])            
            dmds=cv2.vconcat([frames[3],frames[5],frames[7],frames[9],frames[11]])
            results=cv2.hconcat([origs,accs,dmds])
            
        cv2.imshow('results',results)
        
        if lr:
            black = np.zeros((black_shape[0],black_shape[1],3), np.uint8)
            tovideo=cv2.hconcat([black,cv2.resize(results, (video_shape[1],video_shape[0])),black])
        else:
            black = np.zeros((black_shape[0],black_shape[1],3), np.uint8)            
            tovideo=cv2.vconcat([black,cv2.resize(results, (video_shape[1],video_shape[0])),black])
        cv2.imshow('output video',tovideo)
        cv2.imwrite(f'videos/tmp_imgs/frame_{im_counter}.png', tovideo)   
        im_counter+=1
        if tovideo.shape[0]!=1080 or tovideo.shape[1]!=1920:
            print(tovideo.shape)
        key=cv2.waitKey(int(1000//30)) & 0xFF

        if key == ord('q'):
            v_shapes.append(video_shape)
            break

        
        
imgs=os.listdir(f'videos/tmp_imgs/')
imgs.sort(key=getNum)
imgs=[f'videos/tmp_imgs/'+img for img in imgs]

from os.path import exists
def quality(v_path,vref_path,outfile): 
    print(f'generating {outfile}')
    outfile=f'videos/tmp_csvs/{outfile}'
    if exists(outfile):
        aux=pd.read_csv(outfile)
        if len(aux)==90:
            return
        else:
            print(f'missing data: {outfile}')

    cmd='ffmpeg -i ' 
    cmd+=v_path
    cmd+=' -i '
    cmd+=vref_path
    cmd+=' -lavfi libvmaf=model_path="../vmaf/model/vmaf_v0.6.1.json":psnr=1:ssim=1:ms_ssim=1:log_fmt=csv:log_path='
    cmd+=outfile
    cmd+=' -f null -'
    print(cmd)
    os.system(cmd)

for i,v_ref in enumerate(gt):
    v_acc=acc[i]
    v_acc_df=acc_df[i]
    v_acc_best=acc_best[i]
    v_acchm_df=acchm_df[i]
    v_acchm_best=acchm_best[i]
    v_dmd=dmd[i]
    v_dmd_df=dmd_df[i]
    v_dmd_best=dmd_best[i]
    v_dmdhm_df=dmdhm_df[i]
    v_dmdhm_best=dmdhm_best[i]
    quality(v_acc,v_ref,f'acc_{i}.csv')
    quality(v_acc_df,v_ref,f'acc_df_{i}.csv')
    quality(v_acc_best,v_ref,f'acc_best_{i}.csv')
    quality(v_acchm_df,v_ref,f'acchm_df_{i}.csv')
    quality(v_acchm_best,v_ref,f'acchm_best_{i}.csv')
    quality(v_dmd,v_ref,f'dmd_{i}.csv')
    quality(v_dmd_df,v_ref,f'dmd_df_{i}.csv')
    quality(v_dmd_best,v_ref,f'dmd_best_{i}.csv')
    quality(v_dmdhm_df,v_ref,f'dmdhm_df_{i}.csv')
    quality(v_dmdhm_best,v_ref,f'dmdhm_best_{i}.csv')


acc_df=pd.DataFrame()
acc_df_df=pd.DataFrame()
acc_best_df=pd.DataFrame()
acchm_df_df=pd.DataFrame()
acchm_best_df=pd.DataFrame()
dmd_df=pd.DataFrame()
dmd_df_df=pd.DataFrame()
dmd_best_df=pd.DataFrame()
dmdhm_df_df=pd.DataFrame()
dmdhm_best_df=pd.DataFrame()

mean_df=pd.DataFrame(columns=['psnr','ssim','ms_ssim','vmaf'])
print(len(imgs))
for i in range(0,len(imgs),90):
    vid_num=i//90
    acc_i=pd.read_csv(f'videos/tmp_csvs/acc_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    acc_df=acc_df.append(acc_i,ignore_index=True)    
    acc_df_i=pd.read_csv(f'videos/tmp_csvs/acc_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    acc_df_df=acc_df_df.append(acc_df_i,ignore_index=True)
    acc_best_i=pd.read_csv(f'videos/tmp_csvs/acc_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    acc_best_df=acc_best_df.append(acc_best_i,ignore_index=True)
    acchm_df_i=pd.read_csv(f'videos/tmp_csvs/acchm_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    acchm_df_df=acchm_df_df.append(acchm_df_i,ignore_index=True)
    acchm_best_i=pd.read_csv(f'videos/tmp_csvs/acchm_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    acchm_best_df=acchm_best_df.append(acchm_best_i,ignore_index=True)
    dmd_i=pd.read_csv(f'videos/tmp_csvs/dmd_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dmd_df=dmd_df.append(dmd_i,ignore_index=True)    
    dmd_df_i=pd.read_csv(f'videos/tmp_csvs/dmd_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dmd_df_df=dmd_df_df.append(dmd_df_i,ignore_index=True)
    dmd_best_i=pd.read_csv(f'videos/tmp_csvs/dmd_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dmd_best_df=dmd_best_df.append(dmd_best_i,ignore_index=True)
    dmdhm_df_i=pd.read_csv(f'videos/tmp_csvs/dmdhm_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dmdhm_df_df=dmdhm_df_df.append(dmdhm_df_i,ignore_index=True)
    dmdhm_best_i=pd.read_csv(f'videos/tmp_csvs/dmdhm_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dmdhm_best_df=dmdhm_best_df.append(dmdhm_best_i,ignore_index=True)

    mean_list=[acc_i.mean(),acc_df_i.mean(),acc_best_i.mean(),acchm_df_i.mean(),acchm_best_i.mean(),dmd_i.mean(),dmd_df_i.mean(),dmd_best_i.mean(),dmdhm_df_i.mean(),dmdhm_best_i.mean()]
    index_names=[f'acc v{vid_num}',f'sfm-acc v{vid_num}',f'sfm-acc best v{vid_num}',f'sfm-acc halfmag v{vid_num}',f'sfm-acc halfmag best v{vid_num}',
                 f'dmd v{vid_num}',f'sfm-dmd v{vid_num}',f'sfm-dmd best v{vid_num}',f'sfm-dmd halfmag v{vid_num}',f'sfm-dmd halfmag best v{vid_num}']
    df_means = pd.DataFrame(mean_list, index=index_names)
    print(df_means)
    mean_df=mean_df.append(df_means)
#list_of_series = [pd.Series([1,2],index=cols), pd.Series([3,4],index=cols)]

def text_pos(shape,grid=(3,5),pos=(1,0), bottom=True, add_pixels=(0,0)):
    minisize=(shape[0]//grid[0],shape[1]//grid[1])
    
    if shape[0]==1080:
        h_start=(1920-shape[1])//2
        if bottom:
            return (pos[0]*minisize[1]+h_start+add_pixels[1],
                    pos[1]*minisize[0]+add_pixels[0]+int(minisize[0]*0.9))
        else:
            return (pos[0]*minisize[1]+h_start+add_pixels[1],
                    pos[1]*minisize[0]+add_pixels[0])

    else:
        h_start=(1080-shape[0])//2
        h_end=h_start+shape[0]
        if bottom:
            return (pos[0]*minisize[1]+add_pixels[1],
                    pos[1]*minisize[0]+h_start+add_pixels[0]+int(minisize[0]*0.9))
        else:
            return (pos[0]*minisize[1]+add_pixels[1],
                    pos[1]*minisize[0]+h_start+add_pixels[0])

print(acc_df)

def saveImgTxt(imfile, text, pos, text_mean, pos_mean):
    frame_i=cv2.imread(imfile)
    cv2.putText(frame_i, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1,cv2.LINE_AA)        
    cv2.putText(frame_i, text_mean, pos_mean, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1,cv2.LINE_AA)    
    cv2.imwrite(imfile,frame_i)

print('dataframe lens:',len(acc_df),len(acc_df_df),len(acchm_df_df),len(acchm_best_df),len(dmd_df),len(dmd_df_df),len(dmdhm_df_df),len(dmdhm_best_df))
for i,imfile in enumerate(imgs):
    v_num=i//90
    if i%90==0:
        print(v_num)    
    shape=(v_shapes[v_num][0],v_shapes[v_num][1])
    text=f"PSNR: {acc_df.at[i,'psnr']:.1f}, SSIM: {acc_df.at[i,'ssim']:.3f}, MS-SSIM: {acc_df.at[i,'ms_ssim']:.3f}, VMAF: {acc_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'acc v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'acc v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'acc v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'acc v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(0,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(0,1),add_pixels=(0,0)))    
    text=f"PSNR: {acc_df_df.at[i,'psnr']:.1f}, SSIM: {acc_df_df.at[i,'ssim']:.3f}, MS-SSIM: {acc_df_df.at[i,'ms_ssim']:.3f}, VMAF: {acc_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-acc v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-acc v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-acc v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-acc v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(1,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(1,1),add_pixels=(0,0)))
    text=f"PSNR: {acc_best_df.at[i,'psnr']:.1f}, SSIM: {acc_best_df.at[i,'ssim']:.3f}, MS-SSIM: {acc_best_df.at[i,'ms_ssim']:.3f}, VMAF: {acc_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-acc best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-acc best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-acc best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-acc best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(2,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(2,1),add_pixels=(0,0)))
    text=f"PSNR: {acchm_df_df.at[i,'psnr']:.1f}, SSIM: {acchm_df_df.at[i,'ssim']:.3f}, MS-SSIM: {acchm_df_df.at[i,'ms_ssim']:.3f}, VMAF: {acchm_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-acc halfmag v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-acc halfmag v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-acc halfmag v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-acc halfmag v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(3,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(3,1),add_pixels=(0,0)))
    text=f"PSNR: {acchm_best_df.at[i,'psnr']:.1f}, SSIM: {acchm_best_df.at[i,'ssim']:.3f}, MS-SSIM: {acchm_best_df.at[i,'ms_ssim']:.3f}, VMAF: {acchm_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-acc halfmag best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-acc halfmag best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-acc halfmag best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-acc halfmag best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(4,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(4,1),add_pixels=(0,0)))
    text=f"PSNR: {dmd_df.at[i,'psnr']:.1f}, SSIM: {dmd_df.at[i,'ssim']:.3f}, MS-SSIM: {dmd_df.at[i,'ms_ssim']:.3f}, VMAF: {dmd_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'dmd v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'dmd v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'dmd v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'dmd v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(0,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(0,2),add_pixels=(0,0)))
    text=f"PSNR: {dmd_df_df.at[i,'psnr']:.1f}, SSIM: {dmd_df_df.at[i,'ssim']:.3f}, MS-SSIM: {dmd_df_df.at[i,'ms_ssim']:.3f}, VMAF: {dmd_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dmd v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dmd v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dmd v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dmd v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(1,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(1,2),add_pixels=(0,0)))
    text=f"PSNR: {dmd_best_df.at[i,'psnr']:.1f}, SSIM: {dmd_best_df.at[i,'ssim']:.3f}, MS-SSIM: {dmd_best_df.at[i,'ms_ssim']:.3f}, VMAF: {dmd_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dmd best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dmd best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dmd best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dmd best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(2,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(2,2),add_pixels=(0,0)))
    text=f"PSNR: {dmdhm_df_df.at[i,'psnr']:.1f}, SSIM: {dmdhm_df_df.at[i,'ssim']:.3f}, MS-SSIM: {dmdhm_df_df.at[i,'ms_ssim']:.3f}, VMAF: {dmdhm_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dmd halfmag v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dmd halfmag v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dmd halfmag v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dmd halfmag v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(3,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(3,2),add_pixels=(0,0)))
    text=f"PSNR: {dmdhm_best_df.at[i,'psnr']:.1f}, SSIM: {dmdhm_best_df.at[i,'ssim']:.3f}, MS-SSIM: {dmdhm_best_df.at[i,'ms_ssim']:.3f}, VMAF: {dmdhm_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dmd halfmag best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dmd halfmag best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dmd halfmag best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dmd halfmag best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(4,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(4,2),add_pixels=(0,0)))
    

clip = ImageSequenceClip(imgs, fps=30)
clip.write_videofile('videos/SFM_large_motion.mp4')

import shutil
try: 
    shutil.rmtree(f'videos/tmp_imgs/')
except FileNotFoundError:
    print ("Folder has been deleted already and Video is ready")  
#folder_original_large='../momag-database/resultVideos/moveHalfLarge_downsize_4/'

