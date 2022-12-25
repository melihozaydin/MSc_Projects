function [I,x,y,K]=drawMatches(I1,I2,P1,P2,colorIDs,states,IDs,matches)
% PLOTMATCHES  Plot keypoint matches
%   PLOTMATCHES(I1,I2,P1,P2,MATCHES) plots the two images I1 and I2
%   and lines connecting the frames (keypoints) P1 and P2 as specified
%   by MATCHES.
%
%   P1 and P2 specify two sets of frames, one per column. The first
%   two elements of each column specify the X,Y coordinates of the
%   corresponding frame. Any other element is ignored.
%
%   MATCHES specifies a set of matches, one per column. The two
%   elementes of each column are two indexes in the sets P1 and P2
%   respectively.
%
%   The images I1 and I2 might be either both grayscale or both color
%   and must have DOUBLE storage class. If they are color the range
%   must be normalized in [0,1].
%
% AUTORIGHTS
% Copyright (c) 2006 The Regents of the University of California.
% All Rights Reserved.
% 
% Written by Andrea Vedaldi
% UCLA Vision Lab - Department of Computer Science
% Modified by Ceydanur Öztürk


% --------------------------------------------------------------------
%                                                           Do the job
% --------------------------------------------------------------------
global colors;
global axisFtrs;

[M1,N1,K1]=size(I1) ;
[M2,N2,K2]=size(I2) ;

N3=N1+N2 ;
M3=max(M1,M2) ;
oj=N1 ;
oi=0 ;

I=zeros(M3,N3,K1) ;

I(1:M1,1:N1,:) = I1 ;
I(oi+(1:M2),oj+(1:N2),:) = I2 ;

set(gcf,'CurrentAxes',axisFtrs);
%Clear object but dont lose it
cla(axisFtrs,'reset');
set(axisFtrs,'Tag','axesIm');
imagesc(I) ; colormap gray ; hold on ; axis image ; axis off ;

K = size(matches, 2); 
nans = NaN * ones(1,K) ;
if size(matches,1)>0
    x = [ P1(1,matches(1,:)) ;nans; P2(1,matches(2,:))+oj ; nans ] ;
    y = [ P1(2,matches(1,:)) ;nans; P2(2,matches(2,:))+oi ; nans ] ;

    for i=1:K
        if states(matches(1,i))==1
            plot(x(:,i),y(:,i),'Marker','.','Color',colors(:,colorIDs(matches(1,i)))') ;
            text(x(1,i)-2,y(1,i)-2,num2str(IDs(matches(1,i))), 'Color', 'y','FontSize',6);
            text(x(1,i)+oj-2,y(1,i)+oi-2, num2str(matches(2,i)), 'Color', 'y','FontSize',6);
        end
    end
    drawnow;
end
hold off;

% TO DRAW ONE BY ONE THE KEYPOINTS
%  for i=1:K
%         if(mod(i,2)==1)
%             c='r';
%         else
%             c='y';
%         end
%         plot(x(:,i),y(:,i),'Marker','.','Color',c) ;%colors(:,i)'
%         drawnow;
%         wait(0.5);
%         cla(h,'reset');
%     imagesc(I) ; colormap gray ; hold on ; axis image ; axis off ;
%     end

