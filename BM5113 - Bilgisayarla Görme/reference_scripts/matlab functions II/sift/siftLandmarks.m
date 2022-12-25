%Compute SIFT features for input frame and perform match operation if there exists
%some previous frames for which SIFT is computed
function [points2,descr2,matches,tSpent]=siftLandmarks(frame,points1,descr1)
    matches = [];
    I=double(rgb2gray( frame ))./255;           
    minI = min(I(:));
    maxI = max(I(:));
    I=(I-minI)/maxI ;
    ts=tic;
    [points2,descr2] = sift( I ) ;
    tSpent =toc(ts);
    descr2=uint8(512*descr2) ;
    if ~isempty(points1) && ~isempty(points2)
        ts=tic ; 
%         whos descr1
%         descr1
%         whos descr2
%         descr2
        matches=siftmatch( uint8(descr1), descr2 ) ;
        tSpent =[tSpent toc(ts)];
    end
end