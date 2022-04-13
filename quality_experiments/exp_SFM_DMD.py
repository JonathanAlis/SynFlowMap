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
    

results_file='SynFlowMap_from_DMD.csv'

columns=['video_num','status','orig_file','gt_file','mag_file','sfm_default_file','sfm_best_file',
        'dmd_settings','dmd_alpha','alpha_2',
        'psnr_mag','ssim_mag','ms_ssim_mag','vmaf_mag','score_mag',
        'psnr','ssim','ms_ssim','vmaf','score'
        'best_OFwinsize','best_scale','best_bilatfil_size',
        'psnr_best','ssim_best','ms_ssim_best','vmaf_best','score']
all_params=pd.DataFrame(columns=columns)

#if 
# -----
folder_original='../momag-database/resultVideos/moveHalfLarge_downsize_4/'
videos=os.listdir(folder_original)
origf=[v for v in videos if v.startswith('orig')]
origf.sort(key=getNum)
gt=[v for v in videos if v.startswith('mag')] #gt= ground thruth
gt.sort(key=getNum)
print('original vidos:', origf)

default_scale=1
default_win=15
default_bilatfil=1

folders_dmd=['../data/deepmag/momag-database/large_motion/',           
             '../data/deepmag/momag-database/large_motion_halfmag/']   
folders_results=['./from_DMD/large_motion/',           
                 './from_DMD/large_motion_halfmag/']
folders_default=['../data/proposedResults/from_DMD/default/large_motion/',
                 '../data/proposedResults/from_DMD/default/large_motion_halfmag/']
folders_best    =['../data/proposedResults/from_DMD/best/large_motion/',
                 '../data/proposedResults/from_DMD/best/large_motion_halfmag/']

scale_range=[1,2,4,8]
winsize_range=[7,15,23]
#winsize_range=[7,11,15,19,23]

bilatfil_size_range=[1]

remove=False
#generate all videos
for pnum, path_dmd in enumerate(folders_dmd): 
    #creating folders
    Path(folders_results[pnum]).mkdir(parents=True, exist_ok=True)
    Path(folders_default[pnum]).mkdir(parents=True, exist_ok=True)
    Path(folders_best[pnum]).mkdir(parents=True, exist_ok=True)

    #check DMD path
    if 'halfmag' in path_dmd:
        DMD_settings='halfmag'    
    else:
        DMD_settings=''#+path_dmd[path_dmd.find('4'):-2]
    print("path:",path_dmd)
    videos=os.listdir(path_dmd)
    dmd=[v for v in videos if v.startswith('orig') and v.endswith('.mp4')]
    dmd.sort(key=getNum)

    best_win=[0]*len(dmd)
    best_bil=[0]*len(dmd)
    best_scale=[0]*len(dmd)
    default_scores=[{'psnr':0,'ssim':0,'ms_ssim':0,'vmaf':0,'score':0}]*len(dmd)
    best_scores=[{'psnr':0,'ssim':0,'ms_ssim':0,'vmaf':0,'score':0}]*len(dmd)
    default_file=''
    best_file=''
    for i,dmdv in enumerate(dmd):
    
        ### STEP 1: calculate the metrics of the DMD video
        dmd_metrics_file='./csvs/'+'DMD_metrics_{}_v{}.csv'.format(DMD_settings,i)
        DMD_metrics=calculateMetrics(path_dmd+dmd[i],folder_original+gt[i],dmd_metrics_file)
        DMD_score=overallScore(*DMD_metrics)
        print("mag metrics:", DMD_metrics)
        print("mag score:", DMD_score)
        
        ### STEP 2: create the synflowmap video
        if not os.path.exists(folders_results[pnum]):
            os.makedirs(folders_results[pnum])

        best_score=-1.0
        current_best=''
        for s in scale_range:
            for b in bilatfil_size_range:
                for w in winsize_range:
        
                    out_name=f"{folders_results[pnum]}v{i}_fromDMD{DMD_settings}_s{s}_b{b}_w{w}.mp4"
                    sfm_metrics_file=f"./csvs/SFM_DMD_metrics_v{i}_s{s}_b{b}_w{w}.csv"
                    alpha_2=1
                    if path_dmd.find('halfmag')>=0:
                        alpha_2=2
                        sfm_metrics_file=f"./csvs/SFM_DMD_metrics_v{i}_s{s}_b{b}_w{w}_hm.csv"
                        out_name=f"{folders_results[pnum]}v{i}_fromDMD{DMD_settings}_s{s}_b{b}_w{w}_hm.mp4"


                    if not os.path.exists(out_name) or countFrames(out_name)!=90:# or not os.path.exists(sfm_metrics_file):
                        print(f'generating {out_name}')
                        print(f'Original: {folder_original+origf[i]}')
                        print(f'Ground Truth: {folder_original+gt[i]}')
                        print(f"    generating video for parameters: scale={s}, bilateral filter size={b} and window size={w}")#,end=' ')                  
                        groudtruthCap=cv2.VideoCapture(folder_original+gt[i])
                        num_frames = int(groudtruthCap.get(cv2.CAP_PROP_FRAME_COUNT))
                        imsize= (int(groudtruthCap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(groudtruthCap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
                        manager=synflowmap.manage_videos(folder_original+origf[i],path_dmd+dmd[i],
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
                        best_overall=f"{folders_best[pnum]}v{i}_fromDMD{DMD_settings}_s{s}_b{b}_w{w}.mp4"
                        best_win[i]=w
                        best_bil[i]=b
                        best_scale[i]=s
                        best_scores[i]={'psnr':SFM_metrics[0],'ssim':SFM_metrics[1],'ms_ssim':SFM_metrics[2],'vmaf':SFM_metrics[3],'score':best_score}
                        best_file=out_name[:-4]+'_BEST.mp4'
                    if s==default_scale and b==default_bilatfil and w==default_win:
                        default_file=out_name[:-4]+'_DEFAULT.mp4'
                        shutil.copy(out_name, default_file)
                        default_scores[i]={'psnr':SFM_metrics[0],'ssim':SFM_metrics[1],'ms_ssim':SFM_metrics[2],'vmaf':SFM_metrics[3],'score':SFM_score}
                        shutil.copy(out_name, f"{folders_default[pnum]}v{i}_fromDMD{DMD_settings}_s{s}_b{b}_w{w}.mp4")

                    
                    if remove:
                        os.remove(out_name)
        shutil.copy(best_current,best_file) #copy the file to destination dir
        shutil.copy(best_current, best_overall)


        params=pd.DataFrame({'video_num':i,   
                           'status':[1],                        
                           'orig_file': [folder_original+origf[i]],
                           'gt_file': [folder_original+gt[i]],
                           'mag_file': [path_dmd+dmd[i]],
                           'sfm_default_file': [default_file],
                           'sfm_best_file':[best_file],
                           'dmd_settings':[DMD_settings],
                           'dmd_alpha': [getAlpha(dmd[i])],
                           'alpha_2':[alpha_2],
                           'psnr_mag': [DMD_metrics[0]],
                           'ssim_mag': [DMD_metrics[1]],
                           'ms_ssim_mag': [DMD_metrics[2]],
                           'vmaf_mag': [DMD_metrics[3]],
                           'score_mag': [DMD_score],                           
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
