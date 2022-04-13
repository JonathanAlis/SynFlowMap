import cv2
import os
import re
import numpy as np
from moviepy.editor import VideoFileClip,ImageSequenceClip

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
    

#if os.path.exists('videos/tmp_imgs')
save_frames=[3, 11]
video_save_frames=[19]
biframes=[]

os.makedirs('videos/tmp_imgs',exist_ok=True)
os.makedirs('videos/tmp_csvs',exist_ok=True)
os.makedirs('videos/frames',exist_ok=True)

mag_ratios=([160]*3+[80]*3+[40]*3)*3

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
folder_original='../momag-database/resultVideos/moveOnly2_downsize_4/'

origf=videofiles(folder_original,startswith='orig')
gt=videofiles(folder_original,startswith='mag')


folders_pb=['../data/PhaseBasedResults/momag-database/small_motion/',
           '../data/PhaseBasedResults/momag-database/small_motion_quarter/',
           '../data/PhaseBasedResults/momag-database/small_motion_halfmag/',
           '../data/PhaseBasedResults/momag-database/small_motion_quarter_halfmag/']

pb=videofiles(folders_pb[0],startswith='orig')
pbhm=videofiles(folders_pb[2],startswith='orig')

folders_results_pb=['../data/proposedResults/from_PB/default/small_motion/',
           '../data/proposedResults/from_PB/default/small_motion_quarter/',
           '../data/proposedResults/from_PB/default/small_motion_halfmag/',
           '../data/proposedResults/from_PB/default/small_motion_quarter_halfmag/']
           
pb_df=videofiles(folders_results_pb[0])
pbhm_df=videofiles(folders_results_pb[2])

folders_best_pb=['../data/proposedResults/from_PB/best/small_motion/',
           '../data/proposedResults/from_PB/best/small_motion_quarter/',
           '../data/proposedResults/from_PB/best/small_motion_halfmag/',
           '../data/proposedResults/from_PB/best/small_motion_quarter_halfmag/']

pb_best=videofiles(folders_best_pb[0])
pbhm_best=videofiles(folders_best_pb[2])

folders_dms=['../data/deepmag/momag-database/small_motion/',           
           '../data/deepmag/momag-database/small_motion_halfmag/']


dms=videofiles(folders_dms[0])
dmshm=videofiles(folders_dms[1])
                    
folders_results_dms=['../data/proposedResults/from_DMS/default/small_motion/',           
           '../data/proposedResults/from_DMS/default/small_motion_halfmag/']

dms_df=videofiles(folders_results_dms[0])
dmshm_df=videofiles(folders_results_dms[1])

folders_best_dms=['../data/proposedResults/from_DMS/best/small_motion/',
           '../data/proposedResults/from_DMS/best/small_motion_halfmag/']

dms_best=videofiles(folders_best_dms[0])
dmshm_best=videofiles(folders_best_dms[1])

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
                 ('Phase-Based','PB'),
                 ('DeepMag Static mode','DMD'),
                 ('SynFlowMap from Phase-Based: default','SFMD from PB'),
                 ('SynFlowMap from DeepMag: default','SFMD from DMS'),
                 ('SynFlowMap from Phase-Based: best','SFMB from PB'),
                 ('SynFlowMap from DeepMag: best','SFMB from DMS'),
                 ('SynFlowMap from half-magnification Phase-Based: default','SFMD2 from PB'),
                 ('SynFlowMap from half-magnification DeepMag: default','SFMD2 from DMS'),
                 ('SynFlowMap from half-magnification Phase-Based: best','SFMB2 from PB'),
                 ('SynFlowMap from half-magnification DeepMag: best','SFMB2 from DMS'),
                 ]
                 
    captures=[]
    captures.append(cv2.VideoCapture(origf[i]))
    captures.append(cv2.VideoCapture(gt[i]))
    captures.append(cv2.VideoCapture(pb[i]))
    captures.append(cv2.VideoCapture(dms[i]))

    captures.append(cv2.VideoCapture(pb_df[i]))
    captures.append(cv2.VideoCapture(dms_df[i]))
    captures.append(cv2.VideoCapture(pb_best[i]))
    captures.append(cv2.VideoCapture(dms_best[i]))

    captures.append(cv2.VideoCapture(pbhm_df[i]))
    captures.append(cv2.VideoCapture(dmshm_df[i]))
    captures.append(cv2.VideoCapture(pbhm_best[i]))
    captures.append(cv2.VideoCapture(dmshm_best[i]))
    
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
            print(f'v{i} complete')
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
                
        frame_counter+=1
        
        if i in video_save_frames:
            if frame_counter in save_frames:                
                print(f"frame {frame_counter} of video {i}")
                for counter, f in enumerate(frames):
                    if frame_counter==save_frames[0]:
                        biframes.append(f)
                    if frame_counter==save_frames[1]:
                        horizontal_cut=False
                        if horizontal_cut:        
                            biframes[counter][f.shape[0]//2:,:,:]=f[f.shape[0]//2:,:,:]
                            cv2.line(biframes[counter], (0, f.shape[0]//2), (f.shape[1], f.shape[0]//2), (0, 0, 0), thickness=2)
                            cv2.imwrite(f"videos/frames/v{i}_frames{save_frames[0]}_{save_frames[1]}_{methodnames[counter][1].replace(' ','-')}.png", 
                                        biframes[counter])
                        else:
                            biframes[counter][:,f.shape[1]//2:,:]=f[:,f.shape[1]//2:,:]
                            cv2.line(biframes[counter], (f.shape[1]//2,0), (f.shape[1]//2, f.shape[0]), (0, 0, 0), thickness=2)
                            cv2.imwrite(f"videos/frames/v{i}_frames{save_frames[0]}_{save_frames[1]}_{methodnames[counter][1].replace(' ','-')}.png", 
                                        biframes[counter])
                    frame_filename=f"v{i}_frame{frame_counter}_{methodnames[counter][1].replace(' ','-')}.png"
                    cv2.imwrite(f'videos/frames/{frame_filename}', f)
        
        for idx,f in enumerate(frames):
            draw_text(f,methodnames[idx][1])
        
        info=np.zeros(frames[0].shape, np.uint8)
        draw_text(info, f'Magnification:{str(mag_ratios[i])}x')
        if landscape:
            filler=np.zeros(frames[0].shape, np.uint8)
            origs=cv2.hconcat([frames[0],frames[1],info,filler,filler])
            pbs=cv2.hconcat([frames[2],frames[4],frames[6],frames[8],frames[10]])            
            dmss=cv2.hconcat([frames[3],frames[5],frames[7],frames[9],frames[11]])
            if 0:
                print(video_shape[0]-3*frame[0].shape[0], video_shape, frame[0].shape)
                infobar1=np.zeros((video_shape[0]-3*frame[0].shape[0],frame[0].shape[1]), np.uint8)
                draw_text(infobar1,'Other methods')
                infobar2=np.zeros((video_shape[0]-3*frame[0].shape[0],frame[0].shape[1]), np.uint8)
                draw_text(infobar2,'SFM default')
                infobar3=np.zeros((video_shape[0]-3*frame[0].shape[0],frame[0].shape[1]), np.uint8)
                draw_text(infobar3,'SFM best')
                infobar4=np.zeros((video_shape[0]-3*frame[0].shape[0],frame[0].shape[1]), np.uint8)
                draw_text(infobar4,'SFM halfmag default')
                infobar5=np.zeros((video_shape[0]-3*frame[0].shape[0],frame[0].shape[1]), np.uint8)
                draw_text(infobar4,'SFM halfmag best')
                infobar=cv2.hconcat([infobar1,infobar2,infobar3,infobar4,infobar5])
                results=cv2.vconcat([origs,pbs,dmss,infobar])
            results=cv2.vconcat([origs,pbs,dmss])
            
        else:
            filler=np.zeros(frames[0].shape, np.uint8)
            origs=cv2.vconcat([frames[0],frames[1],filler,filler,filler])
            pbs=cv2.vconcat([frames[2],frames[4],frames[6],frames[8],frames[10]])            
            dmss=cv2.vconcat([frames[3],frames[5],frames[7],frames[9],frames[11]])
            results=cv2.hconcat([origs,pbs,dmss])
            
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
        return

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
    v_pb=pb[i]
    v_pb_df=pb_df[i]
    v_pb_best=pb_best[i]
    v_pbhm_df=pbhm_df[i]
    v_pbhm_best=pbhm_best[i]
    v_dms=dms[i]
    v_dms_df=dms_df[i]
    v_dms_best=dms_best[i]
    v_dmshm_df=dmshm_df[i]
    v_dmshm_best=dmshm_best[i]
    quality(v_pb,v_ref,f'pb_{i}.csv')
    quality(v_pb_df,v_ref,f'pb_df_{i}.csv')
    quality(v_pb_best,v_ref,f'pb_best_{i}.csv')
    quality(v_pbhm_df,v_ref,f'pbhm_df_{i}.csv')
    quality(v_pbhm_best,v_ref,f'pbhm_best_{i}.csv')
    quality(v_dms,v_ref,f'dms_{i}.csv')
    quality(v_dms_df,v_ref,f'dms_df_{i}.csv')
    quality(v_dms_best,v_ref,f'dms_best_{i}.csv')
    quality(v_dmshm_df,v_ref,f'dmshm_df_{i}.csv')
    quality(v_dmshm_best,v_ref,f'dmshm_best_{i}.csv')
import pandas as pd

pb_df=pd.DataFrame()
pb_df_df=pd.DataFrame()
pb_best_df=pd.DataFrame()
pbhm_df_df=pd.DataFrame()
pbhm_best_df=pd.DataFrame()
dms_df=pd.DataFrame()
dms_df_df=pd.DataFrame()
dms_best_df=pd.DataFrame()
dmshm_df_df=pd.DataFrame()
dmshm_best_df=pd.DataFrame()

mean_df=pd.DataFrame(columns=['psnr','ssim','ms_ssim','vmaf'])
print(len(imgs))
for i in range(0,len(imgs),90):
    vid_num=i//90
    pb_i=pd.read_csv(f'videos/tmp_csvs/pb_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    pb_df=pb_df.append(pb_i,ignore_index=True)    
    pb_df_i=pd.read_csv(f'videos/tmp_csvs/pb_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    pb_df_df=pb_df_df.append(pb_df_i,ignore_index=True)
    pb_best_i=pd.read_csv(f'videos/tmp_csvs/pb_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    pb_best_df=pb_best_df.append(pb_best_i,ignore_index=True)
    pbhm_df_i=pd.read_csv(f'videos/tmp_csvs/pbhm_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    pbhm_df_df=pbhm_df_df.append(pbhm_df_i,ignore_index=True)
    pbhm_best_i=pd.read_csv(f'videos/tmp_csvs/pbhm_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    pbhm_best_df=pbhm_best_df.append(pbhm_best_i,ignore_index=True)
    dms_i=pd.read_csv(f'videos/tmp_csvs/dms_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dms_df=dms_df.append(dms_i,ignore_index=True)    
    dms_df_i=pd.read_csv(f'videos/tmp_csvs/dms_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dms_df_df=dms_df_df.append(dms_df_i,ignore_index=True)
    dms_best_i=pd.read_csv(f'videos/tmp_csvs/dms_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dms_best_df=dms_best_df.append(dms_best_i,ignore_index=True)
    dmshm_df_i=pd.read_csv(f'videos/tmp_csvs/dmshm_df_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dmshm_df_df=dmshm_df_df.append(dmshm_df_i,ignore_index=True)
    dmshm_best_i=pd.read_csv(f'videos/tmp_csvs/dmshm_best_{vid_num}.csv')[['psnr','ssim','ms_ssim','vmaf']]
    dmshm_best_df=dmshm_best_df.append(dmshm_best_i,ignore_index=True)

    mean_list=[pb_i.mean(),pb_df_i.mean(),pb_best_i.mean(),pbhm_df_i.mean(),pbhm_best_i.mean(),dms_i.mean(),dms_df_i.mean(),dms_best_i.mean(),dmshm_df_i.mean(),dmshm_best_i.mean()]
    index_names=[f'pb v{vid_num}',f'sfm-pb v{vid_num}',f'sfm-pb best v{vid_num}',f'sfm-pb halfmag v{vid_num}',f'sfm-pb halfmag best v{vid_num}',
                 f'dms v{vid_num}',f'sfm-dms v{vid_num}',f'sfm-dms best v{vid_num}',f'sfm-dms halfmag v{vid_num}',f'sfm-dms halfmag best v{vid_num}']
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

print(pb_df)

def saveImgTxt(imfile, text, pos, text_mean, pos_mean):
    frame_i=cv2.imread(imfile)
    cv2.putText(frame_i, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1,cv2.LINE_AA)        
    cv2.putText(frame_i, text_mean, pos_mean, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1,cv2.LINE_AA)    
    cv2.imwrite(imfile,frame_i)

for i,imfile in enumerate(imgs):
    v_num=i//90
    if i%90==0:
        print(v_num)    
    shape=(v_shapes[v_num][0],v_shapes[v_num][1])
    text=f"PSNR: {pb_df.at[i,'psnr']:.1f}, SSIM: {pb_df.at[i,'ssim']:.3f}, MS-SSIM: {pb_df.at[i,'ms_ssim']:.3f}, VMAF: {pb_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'pb v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'pb v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'pb v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'pb v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(0,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(0,1),add_pixels=(0,0)))    
    text=f"PSNR: {pb_df_df.at[i,'psnr']:.1f}, SSIM: {pb_df_df.at[i,'ssim']:.3f}, MS-SSIM: {pb_df_df.at[i,'ms_ssim']:.3f}, VMAF: {pb_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-pb v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-pb v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-pb v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-pb v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(1,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(1,1),add_pixels=(0,0)))
    text=f"PSNR: {pb_best_df.at[i,'psnr']:.1f}, SSIM: {pb_best_df.at[i,'ssim']:.3f}, MS-SSIM: {pb_best_df.at[i,'ms_ssim']:.3f}, VMAF: {pb_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-pb best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-pb best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-pb best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-pb best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(2,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(2,1),add_pixels=(0,0)))
    text=f"PSNR: {pbhm_df_df.at[i,'psnr']:.1f}, SSIM: {pbhm_df_df.at[i,'ssim']:.3f}, MS-SSIM: {pbhm_df_df.at[i,'ms_ssim']:.3f}, VMAF: {pbhm_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-pb halfmag v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-pb halfmag v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-pb halfmag v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-pb halfmag v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(3,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(3,1),add_pixels=(0,0)))
    text=f"PSNR: {pbhm_best_df.at[i,'psnr']:.1f}, SSIM: {pbhm_best_df.at[i,'ssim']:.3f}, MS-SSIM: {pbhm_best_df.at[i,'ms_ssim']:.3f}, VMAF: {pbhm_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-pb halfmag best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-pb halfmag best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-pb halfmag best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-pb halfmag best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(4,1),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(4,1),add_pixels=(0,0)))
    text=f"PSNR: {dms_df.at[i,'psnr']:.1f}, SSIM: {dms_df.at[i,'ssim']:.3f}, MS-SSIM: {dms_df.at[i,'ms_ssim']:.3f}, VMAF: {dms_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'dms v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'dms v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'dms v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'dms v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(0,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(0,2),add_pixels=(0,0)))
    text=f"PSNR: {dms_df_df.at[i,'psnr']:.1f}, SSIM: {dms_df_df.at[i,'ssim']:.3f}, MS-SSIM: {dms_df_df.at[i,'ms_ssim']:.3f}, VMAF: {dms_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dms v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dms v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dms v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dms v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(1,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(1,2),add_pixels=(0,0)))
    text=f"PSNR: {dms_best_df.at[i,'psnr']:.1f}, SSIM: {dms_best_df.at[i,'ssim']:.3f}, MS-SSIM: {dms_best_df.at[i,'ms_ssim']:.3f}, VMAF: {dms_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dms best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dms best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dms best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dms best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(2,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(2,2),add_pixels=(0,0)))
    text=f"PSNR: {dmshm_df_df.at[i,'psnr']:.1f}, SSIM: {dmshm_df_df.at[i,'ssim']:.3f}, MS-SSIM: {dmshm_df_df.at[i,'ms_ssim']:.3f}, VMAF: {dmshm_df_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dms halfmag v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dms halfmag v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dms halfmag v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dms halfmag v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(3,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(3,2),add_pixels=(0,0)))
    text=f"PSNR: {dmshm_best_df.at[i,'psnr']:.1f}, SSIM: {dmshm_best_df.at[i,'ssim']:.3f}, MS-SSIM: {dmshm_best_df.at[i,'ms_ssim']:.3f}, VMAF: {dmshm_best_df.at[i,'vmaf']:.1f}"
    text_mean=f"AVG: PSNR: {mean_df.at[f'sfm-dms halfmag best v{v_num}','psnr']:.1f}, SSIM: {mean_df.at[f'sfm-dms halfmag best v{v_num}','ssim']:.3f}, MS-SSIM: {mean_df.at[f'sfm-dms halfmag best v{v_num}','ms_ssim']:.3f}, VMAF: {mean_df.at[f'sfm-dms halfmag best v{v_num}','vmaf']:.1f}"
    saveImgTxt(imfile,text,text_pos(shape,pos=(4,2),add_pixels=(20,0)),text_mean,text_pos(shape,pos=(4,2),add_pixels=(0,0)))
    

clip = ImageSequenceClip(imgs, fps=30)
clip.write_videofile('videos/SFM_small_motion.mp4')

import shutil
try: 
    shutil.rmtree(f'videos/tmp_imgs/')
except FileNotFoundError:
    print ("Folder has been deleted already and Video is ready")  
#folder_original_large='../momag-database/resultVideos/moveHalfLarge_downsize_4/'

