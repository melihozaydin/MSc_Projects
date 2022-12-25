% SIFT_DEMO5  Demonstrate SIFT code (5)
%   Finding eggs.

I=imreadbw('data/nest.png') ;

[f,d,gss,dogss] = sift(I,'Verbosity',1,'BoundaryPoint',0,'Threshold',.0282,'FirstOctave',-1,'EdgeThreshold',0) ; 
d = uint8(512*d) ;

figure(1) ; clf ; colormap gray ;
imagesc(I) ; hold on ; axis equal ;
h=plotsiftframe(f) ;
set(h,'LineWidth',3) ;

figure(2); clf ; plotss(dogss) ; colormap gray;
