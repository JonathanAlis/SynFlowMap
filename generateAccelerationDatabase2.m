addpath('acceleration-magnification','acceleration-magnification/third','acceleration-magnification/third/matlabPyrTools','acceleration-magnification/third/matlabPyrTools/MEX','acceleration-magnification/third/phaseCorrection');
warning('off')

inPath='./momag-database/resultVideos/moveHalfLarge_downsize_4/';
inOrigs=dir(strcat(inPath,'orig*.mp4'));
inMags=dir(strcat(inPath,'mag*.mp4'));
generate=1;
generatehalf=1;

resultsPath='./data/VideoAccelerationResults/momag-database/moveHalfLarge_downsize_4/';
mkdir(resultsPath);
if generate
    for i=1:length(inOrigs)
        inFile=inOrigs(i).name(1:end-4);
        %inFile=strcat(inPath,inFile);
        inMag=inMags(i).name;
        inMag=strcat(inPath,inMag);
        inMag=inMag(1:end-4);
        
        B= regexp(inMag,'\d*','Match');
        alpha=B{end};
        disp('phase amplify')
        disp(inFile)
        alpha=str2num(alpha);
        disp(alpha)
        fr90=strcat(resultsPath,inFile,'/im_write/fr90.png');
        if ~isfile(fr90)
            %orig21_plainbg_moveXY_amp2.0_Large_fm_2_alpha_4_pylevel_4_kernel_INT
            [vid_in,params] = setparameters(inFile, '.mp4', inPath, resultsPath, 2, alpha, 'INT');%fm,alpha,kernel
            motionamp(vid_in,params);
        else
            disp('already exists, skipping');
        end

    end
end


resultsPath='./data/VideoAccelerationResults/momag-database/moveHalfLarge_downsize_4_halfMag/';
mkdir(resultsPath);
if generatehalf
    for i=1:length(inOrigs)
        inFile=inOrigs(i).name(1:end-4);
        %inFile=strcat(inPath,inFile);
        inMag=inMags(i).name;
        inMag=strcat(inPath,inMag);
        inMag=inMag(1:end-4);

        B= regexp(inMag,'\d*','Match');
        alpha=B{end};
        disp('phase amplify')
        disp(inFile)
        alpha=str2num(alpha)/2;
        disp(alpha)
        
        %outfilename=strcat(resultsPath,inFile,'_fm_2_alpha_',int2str(alpha),'_pylevel_4_kernel_INT.avi');
        fr90=strcat(resultsPath,inFile,'/im_write/fr90.png');
        if ~isfile(fr90)
            %orig21_plainbg_moveXY_amp2.0_Large_fm_2_alpha_4_pylevel_4_kernel_INT
            [vid_in,params] = setparameters(inFile, '.mp4', inPath, resultsPath, 2, alpha, 'INT');%fm,alpha,kernel
            motionamp(vid_in,params);
        else
            disp('already exists, skipping');
        end
        

    end
end

