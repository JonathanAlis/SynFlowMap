import sys
sys.path.append("..")
import synflowmap
import cv2
import os
import re
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import shutil
from pathlib import Path

def getNum(str):
    numbers = re.findall('[0-9]+', str)
    return int(numbers[0])    
def getAlpha(str):
    str=str[str.find('alpha'):]
    numbers = re.findall('[0-9]+', str)
    return int(numbers[0])
def getAvgMetrics(csv_file):
    df=pd.read_csv(csv_file)
    psnr=df['psnr'].mean(axis = 0)
    ssim=df['ssim'].mean(axis = 0)
    ms_ssim=df['ms_ssim'].mean(axis = 0)
    vmaf=df['vmaf'].mean(axis = 0)
    return (psnr,ssim,ms_ssim,vmaf)
def calculateMetrics(v_path,vref_path,out):
    if not os.path.exists(out):            
        if not os.path.exists('./csvs/'):
            os.makedirs('./csvs/')
        cmd='ffmpeg -i ' 
        cmd+=v_path
        cmd+=' -i '
        cmd+=vref_path
        cmd+=' -lavfi libvmaf=model_path="../vmaf/model/vmaf_v0.6.1.json":psnr=1:ssim=1:ms_ssim=1:log_fmt=csv:log_path='
        cmd+=out
        cmd+=' -f null -'
        print(cmd)
        os.system(cmd)
    return getAvgMetrics(out)
def overallScore(psnr,ssim,ms_ssim,vmaf):
    return (ssim+ms_ssim+vmaf/100)/3
def countFrames(file):
    cap = cv2.VideoCapture(file)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

results_file='SynFlowMap_from_PB.csv'

if os.path.isfile(results_file):
    all_params=pd.read_csv(results_file)
else:
    columns=['video_num','status','orig_file','gt_file','mag_file','sfm_default_file','sfm_best_file',
            'pb_settings','pb_alpha','alpha_2',
            'psnr_mag','ssim_mag','ms_ssim_mag','vmaf_mag','score_mag',
            'psnr','ssim','ms_ssim','vmaf','score'
            'best_OFwinsize','best_scale','best_bilatfil_size',
            'psnr_best','ssim_best','ms_ssim_best','vmaf_best','score']
    all_params=pd.DataFrame(columns=columns)

#if 
# -----
folder_original='../momag-database/resultVideos/moveOnly2_downsize_4/'
videos=os.listdir(folder_original)
origf=[v for v in videos if v.startswith('orig')]
origf.sort(key=getNum)
gt=[v for v in videos if v.startswith('mag')] #gt= ground thruth
gt.sort(key=getNum)
print('original videos:', origf)

default_scale=1
default_win=15
default_bilatfil=1


folders_pb=['../data/PhaseBasedResults/momag-database/small_motion/',
           '../data/PhaseBasedResults/momag-database/small_motion_quarter/',
           '../data/PhaseBasedResults/momag-database/small_motion_halfmag/',
           '../data/PhaseBasedResults/momag-database/small_motion_quarter_halfmag/']
folders_results=['./from_PB/small_motion/',
           './from_PB/small_motion_quarter/',
           './from_PB/small_motion_halfmag/',
           './from_PB/small_motion_quarter_halfmag/']
folders_default=['../data/proposedResults/from_PB/default/small_motion/',
                 '../data/proposedResults/from_PB/default/small_motion_quarter/',
                 '../data/proposedResults/from_PB/default/small_motion_halfmag/',
                 '../data/proposedResults/from_PB/default/small_motion_quarter_halfmag/']
folders_best    =['../data/proposedResults/from_PB/best/small_motion/',
                 '../data/proposedResults/from_PB/best/small_motion_quarter/',
                 '../data/proposedResults/from_PB/best/small_motion_halfmag/',
                 '../data/proposedResults/from_PB/best/small_motion_quarter_halfmag/']

scale_range=[1,2,4,8]
winsize_range=[7,15,23]
#winsize_range=[7,11,15,19,23]

bilatfil_size_range=[1]

remove=False
#generate all videos
for pnum, path_pb in enumerate(folders_pb): 
    print(f"On folder {path_pb}")

    #creating folders
    Path(folders_results[pnum]).mkdir(parents=True, exist_ok=True)
    Path(folders_default[pnum]).mkdir(parents=True, exist_ok=True)
    Path(folders_best[pnum]).mkdir(parents=True, exist_ok=True)



    #check PB path
    if 'halfmag' in path_pb:
        PB_settings='halfmag'    
    else:
        PB_settings=''#+path_pb[path_pb.find('4'):-2]
    if 'quarter' in path_pb:
        PB_settings+='_quarter'    
    
    videos=os.listdir(path_pb)
    pb=[v for v in videos if v.startswith('orig') and v.endswith('.avi')]
    pb.sort(key=getNum)

    print(pb)
    best_win=[0]*len(pb)
    best_bil=[0]*len(pb)
    best_scale=[0]*len(pb)
    default_scores=[{'psnr':0,'ssim':0,'ms_ssim':0,'vmaf':0,'score':0}]*len(pb)
    best_scores=[{'psnr':0,'ssim':0,'ms_ssim':0,'vmaf':0,'score':0}]*len(pb)
    default_file=''
    best_file=''
    for i,pbv in enumerate(pb):
    
        ### STEP 1: calculate the metrics of the PB video
        pb_metrics_file='./csvs/'+'PB_metrics_{}_v{}.csv'.format(PB_settings,i)
        PB_metrics=calculateMetrics(path_pb+pb[i],folder_original+gt[i],pb_metrics_file)
        PB_score=overallScore(*PB_metrics)
        print("mag metrics:", PB_metrics)
        print("mag score:", PB_score)
        
        ### STEP 2: create the synflowmap video
        if not os.path.exists(folders_results[pnum]):
            os.makedirs(folders_results[pnum])

        best_score=-1.0
        current_best=''
        for s in scale_range:
            for b in bilatfil_size_range:
                for w in winsize_range:
        
                    out_name=f"{folders_results[pnum]}v{i}_fromPB{PB_settings}_s{s}_b{b}_w{w}.mp4"
                    sfm_metrics_file=f"./csvs/SFM_PB_metrics_v{i}_s{s}_b{b}_w{w}.csv"
                    alpha_2=1
                    if path_pb.find('halfmag')>=0:
                        alpha_2=2
                        sfm_metrics_file=f"./csvs/SFM_PB_metrics_v{i}_s{s}_b{b}_w{w}_hm.csv"
                        out_name=f"{folders_results[pnum]}v{i}_fromPB{PB_settings}_s{s}_b{b}_w{w}_hm.mp4"


                    if not os.path.exists(out_name) or countFrames(out_name)!=90:# or not os.path.exists(sfm_metrics_file):
                        print(f'generating {out_name}')
                        print(f'Original: {folder_original+origf[i]}')
                        print(f'Ground Truth: {folder_original+gt[i]}')
                        print(f"    generating video for parameters: scale={s}, bilateral filter size={b} and window size={w}")#,end=' ')                  
                        groudtruthCap=cv2.VideoCapture(folder_original+gt[i])
                        num_frames = int(groudtruthCap.get(cv2.CAP_PROP_FRAME_COUNT))
                        imsize= (int(groudtruthCap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(groudtruthCap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
                        manager=synflowmap.manage_videos(folder_original+origf[i],path_pb+pb[i],
                                                        s,b,alpha_2,w)
                        writer = cv2.VideoWriter(out_name,
                                                cv2.VideoWriter_fourcc(*'FMP4'), 
                                                30,
                                                imsize) #MJPG 4.3Mb, H264 didnt save, FMP4,MP4V,AVC1...
                        manager.start(verbose=False)
                        frameCounter=0
                        while(not manager.has_finished()):
                            ok,frame=manager.nextFrame()
                            frameCounter+=1
                            if not ok:
                                break
                            writer.write(frame)
                        print(f"         {frameCounter} frames written")
                        if frameCounter<90:
                            raise "error, not enough frames..."
                        writer.release()
                    else:
                        pass
                        #if has already calculated and is already in the csv file, break
                        #if out_name in all_params['sfm_default_file'].values:
                            #pass
                        #    break
                    ### STEP 3: calculate metric of generated video
                    SFM_metrics=calculateMetrics(out_name,folder_original+gt[i],sfm_metrics_file)
                    SFM_score=overallScore(*SFM_metrics)
                    print("sfm metrics:", SFM_metrics)
                    print("sfm score:", SFM_score)
                    print()
                    print()
                    if SFM_score>=best_score:
                        best_score=SFM_score
                        print(f"NEW BEST SCORE: {best_score}, with metrics: {SFM_metrics}")
                        #os.rename(out_name, out_name[:-4]+'_BEST.mp4')
                        best_current=out_name#[:-4]+'_BEST.mp4'
                        best_overall=f"{folders_best[pnum]}v{i}_fromPB{PB_settings}_s{s}_b{b}_w{w}.mp4"
                        best_win[i]=w
                        best_bil[i]=b
                        best_scale[i]=s
                        best_scores[i]={'psnr':SFM_metrics[0],'ssim':SFM_metrics[1],'ms_ssim':SFM_metrics[2],'vmaf':SFM_metrics[3],'score':best_score}
                        best_file=out_name[:-4]+'_BEST.mp4'
                    if s==default_scale and b==default_bilatfil and w==default_win:
                        default_file=out_name[:-4]+'_DEFAULT.mp4'
                        shutil.copy(out_name, default_file)
                        default_scores[i]={'psnr':SFM_metrics[0],'ssim':SFM_metrics[1],'ms_ssim':SFM_metrics[2],'vmaf':SFM_metrics[3],'score':SFM_score}                        
                        shutil.copy(out_name, f"{folders_default[pnum]}v{i}_fromPB{PB_settings}_s{s}_b{b}_w{w}.mp4")

                    
                    if remove:
                        os.remove(out_name)
        shutil.copy(best_current,best_file) #copy the file to destination dir
        shutil.copy(best_current, best_overall)

        params=pd.DataFrame({'video_num':i,   
                           'status':[1],                        
                           'orig_file': [folder_original+origf[i]],
                           'gt_file': [folder_original+gt[i]],
                           'mag_file': [path_pb+pb[i]],
                           'sfm_default_file': [default_file],
                           'sfm_best_file':[best_file],
                           'pb_settings':[PB_settings],
                           'pb_alpha': [getAlpha(pb[i])],
                           'alpha_2':[alpha_2],
                           'psnr_mag': [PB_metrics[0]],
                           'ssim_mag': [PB_metrics[1]],
                           'ms_ssim_mag': [PB_metrics[2]],
                           'vmaf_mag': [PB_metrics[3]],
                           'score_mag': [PB_score],                           
                           'psnr':[default_scores[i]['psnr']],                           
                           'ssim':[default_scores[i]['ssim']],                           
                           'ms_ssim':[default_scores[i]['ms_ssim']],                           
                           'vmaf': [default_scores[i]['vmaf']],                           
                           'score': [default_scores[i]['score']],                           
                           'best_OFwinsize':best_win[i],
                           'best_scale':best_scale[i],
                           'best_bilatfil_size':best_bil[i],
                           'psnr_best':[best_scores[i]['psnr']],                           
                           'ssim_best':[best_scores[i]['ssim']],                           
                           'ms_ssim_best':[best_scores[i]['ms_ssim']],                           
                           'vmaf_best': [best_scores[i]['vmaf']],                           
                           'score_best': [best_scores[i]['score']],                           
                            })
        all_params=all_params.append(params, ignore_index=True)
                            #}, index=[i]))
        print(all_params)      
        all_params = all_params.loc[:, ~all_params.columns.str.contains('^Unnamed')]  
        all_params=all_params.reset_index(drop=True)
        all_params.to_csv(results_file)
