
addpath('PhaseBasedRelease','PhaseBasedRelease/PhaseBased','PhaseBasedRelease/matlabPyrTools','PhaseBasedRelease/pyrToolsExt','PhaseBasedRelease/Filters','PhaseBasedRelease/Linear','PhaseBasedRelease/Util')
inPath='./momag-database/resultVideos/moveOnly2_downsize_4/';
inOrigs=dir(strcat(inPath,'orig*.mp4'));
inMags=dir(strcat(inPath,'mag*.mp4'));

generateOctave=1;
generateQuarterOctave=1;
generateOctaveHalfMag=1;
generateQuarterHalfMag=1;

iterate=[generateOctave,generateQuarterOctave,generateOctaveHalfMag,generateQuarterHalfMag];
resultPaths={'./data/PhaseBasedResults/momag-database/moveOnly2_downsize_4/',...
             './data/PhaseBasedResults/momag-database/moveOnly2_downsize_4_quarterOctave/',...
             './data/PhaseBasedResults/momag-database/moveOnly2_downsize_4_halfMag/',...
             './data/PhaseBasedResults/momag-database/moveOnly2_downsize_4_quarterOctave_halfMag/'};
quarterOctave=[0,1,0,1];
halfMag=[0,0,1,1];

for j=1:length(iterate)
    
    if iterate(j)
        resultsPath=resultPaths{j};
        disp(resultsPath)
        mkdir(resultsPath);
        for i=1:length(inOrigs)
            inFile=inOrigs(i).name;
            inFile=strcat(inPath,inFile);
            inMag=inMags(i).name;
            inMag=strcat(inPath,inMag);
            inMag=inMag(1:end-4);

            samplingRate = 30; % Hz
            loCutoff = 1;    % Hz
            hiCutoff = 14;    % Hz
            sigma = 0;         % Pixels
            temporalFilter = @FIRWindowBP; 
            pyrType = 'octave';
            if quarterOctave(j)
                pyrType = 'octave';
            end
            B= regexp(inMag,'\d*','Match');
            alpha=B{end};
            disp('phase amplify')
            disp(inFile)
            alpha=str2num(alpha);
            if halfMag(j)
                alpha=alpha/2;
            end
            disp(alpha)

            phaseAmplify(inFile, alpha, loCutoff, hiCutoff, samplingRate, resultsPath,'sigma', sigma,'pyrType', pyrType,'scaleVideo', 1);
        end
    end
end