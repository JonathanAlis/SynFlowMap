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


def getNum(str):
    numbers = re.findall('[0-9]+', str)
    return int(numbers[0])    
def getAlpha(str):
    str=str[str.find('alpha'):]
    numbers = re.findall('[0-9]+', str)
    return int(numbers[0])
def getAvgMetrics(csv_file):
    try:
        df=pd.read_csv(csv_file)
    except:
        print(csv_file, 'error')
        exit()
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
def oveallScore(psnr,ssim,ms_ssim,vmaf):
    return (ssim+ms_ssim+vmaf/100)/3

folder_original='../momag-database/resultVideos/moveHalfLarge_downsize_4/'
videos=os.listdir(folder_original)
origf=[v for v in videos if v.startswith('orig')]
origf.sort(key=getNum)
gt=[v for v in videos if v.startswith('mag')] #gt= ground thruth
gt.sort(key=getNum)


default_scale=1
default_win=15
default_bilatfil=1
folders_dmd=['../data/deepmag/momag-database/moveHalfLarge_downsize_4/',           
             '../data/deepmag/momag-database/moveHalfLarge_downsize_4_halfMag/']           
folders_results=['../data/proposedResults/from_DMD/default/moveHalfLarge_downsize_4/',           
                 '../data/proposedResults/from_DMD/default/moveHalfLarge_downsize_4_halfMag/']
folders_best=['../data/proposedResults/from_DMD/best/moveHalfLarge_downsize_4/',
              '../data/proposedResults/from_DMD/best/moveHalfLarge_downsize_4_halfMag/']
results_file='SynFlowMap_from_DMD.csv'

if os.path.isfile(results_file):
    all_params=pd.read_csv(results_file)
else:
    columns=['video_num','status','orig_file','gt_file','mag_file','sfm_default_file','sfm_best_file',
            'dmd_settings','dmd_alpha','alpha_2',
            'psnr_mag','ssim_mag','ms_ssim_mag','vmaf_mag',
            'psnr','ssim','ms_ssim','vmaf',
            'best_OFwinsize','best_scale','best_bilatfil_size',
            'psnr_best','ssim_best','ms_ssim_best','vmaf_best']
    all_params=pd.DataFrame(columns=columns)
    
    #generate videos with default
for pnum, path_dmd in enumerate(folders_dmd): 
    #check DMD path
    DMD_settings=''+path_dmd[path_dmd.find('4'):-2]
    print("path:",path_dmd)
    videos=os.listdir(path_dmd)
    dmd=[v for v in videos if v.startswith('orig')]
    dmd.sort(key=getNum)

    for i,dmdv in enumerate(dmd):
        ### STEP 1: calculate the metrics of the DMD video
        dmd_metrics_file='./csvs/'+'DMD_metrics_{}_v{}.csv'.format(DMD_settings,i)
        DMD_metrics=calculateMetrics(path_dmd+dmd[i],folder_original+gt[i],dmd_metrics_file)
        print(DMD_metrics)

        ### STEP 2: create the synflowmag video
        if not os.path.exists(folders_results[pnum]):
            os.makedirs(folders_results[pnum])
        if not os.path.exists(folders_best[pnum]):
            os.makedirs(folders_best[pnum])
        out_name=folders_results[pnum]+'v'+str(i)+'_fromDMD'+DMD_settings+'_default.mp4'
        sfm_metrics_file='./csvs/'+'SFM_metrics_{}_v{}.csv'.format(DMD_settings,i)
        alpha_2=1
        if path_dmd.find('halfMag')>=0:
            alpha_2=2
        if not os.path.exists(out_name) or not os.path.exists(sfm_metrics_file):
            print('generating',out_name)
            print("    generating video for default parameter: scale=1, bilateral filter size=1 and window size=15",end=' ')                  
            groudtruthCap=cv2.VideoCapture(folder_original+gt[i])
            num_frames = int(groudtruthCap.get(cv2.CAP_PROP_FRAME_COUNT))
            imsize= (int(groudtruthCap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(groudtruthCap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
            manager=synflowmap.manage_videos(folder_original+origf[i],path_dmd+dmd[i],
                                            default_scale,default_bilatfil,alpha_2,default_win)
            writer = cv2.VideoWriter(out_name,
                                     cv2.VideoWriter_fourcc(*'FMP4'), 
                                     30,
                                     imsize) #MJPG 4.3Mb, H264 didnt save, FMP4,MP4V,AVC1...
            manager.start(verbose=False)
            frameCounter=0
            while(not manager.has_finished()):
                ok,frame=manager.nextFrame()
                if not ok:
                    break
                writer.write(frame)
            writer.release()
        else:
            #if has already calcylated and is already in the csv file, break
            if out_name in all_params['sfm_default_file'].values:
                break

        ### STEP 3: calculate metric of generated video
        SFM_metrics=calculateMetrics(out_name,folder_original+gt[i],sfm_metrics_file)
        print(SFM_metrics)

        ### STEP 4: save metrics in a csv
        columns=['video_num','status','orig_file','gt_file','mag_file','result_file',
            'dmd_settings','dmd_alpha','alpha_2',
            'psnr_mag','ssim_mag','ms_ssim_mag','vmaf_mag',
            'psnr','ssim','ms_ssim','vmaf',
            'best_OFwinsize','best_scale','best_bilatfil_size',
            'psnr_best','ssim_best','ms_ssim_best','vmaf_best']
        best_params=pd.DataFrame({'video_num':[i],   
                           'status':[1],                        
                           'orig_file': [folder_original+origf[i]],
                           'gt_file': [folder_original+gt[i]],
                           'mag_file': [path_dmd+dmd[i]],
                           'sfm_default_file': [out_name],
                           'sfm_best_file':[''],
                           'dmd_settings':[DMD_settings],
                           'dmd_alpha': [getAlpha(dmd[i])],
                           'alpha_2':[alpha_2],
                           'psnr_mag': [DMD_metrics[0]],
                           'ssim_mag': [DMD_metrics[1]],
                           'ms_ssim_mag': [DMD_metrics[2]],
                           'vmaf_mag': [DMD_metrics[3]],
                           'psnr': [SFM_metrics[0]],
                           'ssim': [SFM_metrics[1]],
                           'ms_ssim': [SFM_metrics[2]],
                           'vmaf': [SFM_metrics[3]],
                           'best_OFwinsize':[0],
                           'best_scale':[0],
                           'best_bilatfil_size':[0],
                           'psnr_best': [0],
                           'ssim_best': [0],
                           'ms_ssim_best': [0],
                           'vmaf_best': [0]
                            })
        all_params=all_params.append(best_params, ignore_index=True)
                            #}, index=[i]))
        print(all_params)      
        all_params = all_params.loc[:, ~all_params.columns.str.contains('^Unnamed')]  
        all_params.to_csv(results_file)


###Step 5: after saving all, try to get the best parameters
scale_range=[1,2,4]
bilatfil_size_range=[1,5,9]
winsize_range=[7,11,15,19,23]
#status meaning:
#0: nothing has run
#1: only default has run
#2: not finished running part 2
#3: finnished running part 2
if not os.path.exists('./tmp/'):
    os.makedirs('./tmp/')
for i,row in all_params.iterrows():
    if row['status']==1 or row['status']==2:
        best_score=0   
        #out_name=folders_results[pnum]+'v'+str(i)+'_fromDMD'+DMD_settings+'_default.mp4'     
        for s in scale_range:
            for b in bilatfil_size_range:
                for w in winsize_range:
                    ### STEP 5.1: generate the videos for each param
                    if os.path.exists('./tmp/vtmp.mp4'):
                        os.remove('./tmp/vtmp.mp4')
                    print("    generating video for default parameter: scale={}, bilateral filter size={} and window size={}".format(s,b,w),end=' ')                  
                    alpha_2=row['alpha_2']
                    groudtruthCap=cv2.VideoCapture(row['gt_file'])
                    num_frames = int(groudtruthCap.get(cv2.CAP_PROP_FRAME_COUNT))
                    imsize= (int(groudtruthCap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(groudtruthCap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
                    manager=synflowmap.manage_videos(row['orig_file'],row['mag_file'],
                                                    s,b,alpha_2,w)
                    ### STEP 5.2: save videos in temporal folder
                    writer = cv2.VideoWriter('./tmp/vtmp.mp4',
                                                cv2.VideoWriter_fourcc(*'FMP4'), 
                                                30,
                                                imsize) #MJPG 4.3Mb, H264 didnt save, FMP4,MP4V,AVC1...
                    manager.start(verbose=False)
                    frameCounter=0
                    while(not manager.has_finished()):
                        ok,frame=manager.nextFrame()
                        if not ok:
                            break
                        writer.write(frame)
                    writer.release()
                    ### STEP 5.3: calculate metric of generated videos
                    sfm_metrics_file='./tmp/'+'SFM_best_metrics_{}_v{}_s{}_b{}_w{}.csv'.format(DMD_settings,i,s,b,w)
                    SFM_metrics=calculateMetrics('./tmp/vtmp.mp4',row['gt_file'],sfm_metrics_file)
                    print(SFM_metrics)
                    score=oveallScore(SFM_metrics[0],SFM_metrics[1],SFM_metrics[2],SFM_metrics[3])
                    if score>best_score:
                        ### STEP 5.4: save new scores and the video
                        print("NEW BEST SCORE!", score,'>',best_score)
                        print("params: scale={},bilfilsize={},winsize={}".format(s,b,w))
                        all_params.loc[i,'status']=2
                        best_path=row['sfm_default_file']
                        best_path=best_path.replace('default','best')
                        all_params.loc[i,'sfm_best_file']=best_path
                        os.rename('./tmp/vtmp.mp4',best_path)
                        all_params.loc[i,'best_OFwinsize']=w
                        all_params.loc[i,'best_scale']=s
                        all_params.loc[i,'best_bilatfil_size']=b
                        all_params.loc[i,'psnr_best']=SFM_metrics[0]
                        all_params.loc[i,'ssim_best']=SFM_metrics[1]
                        all_params.loc[i,'ms_ssim_best']=SFM_metrics[2]
                        all_params.loc[i,'vmaf_best']=SFM_metrics[3]
                        all_params = all_params.loc[:, ~all_params.columns.str.contains('^Unnamed')]  
                        all_params.to_csv(results_file)
                        best_score=score
                    else:
                        os.remove('./tmp/vtmp.mp4')
        all_params.loc[i,'status']=3              
        all_params = all_params.loc[:, ~all_params.columns.str.contains('^Unnamed')]  
        all_params.to_csv(results_file)
                        

                    
