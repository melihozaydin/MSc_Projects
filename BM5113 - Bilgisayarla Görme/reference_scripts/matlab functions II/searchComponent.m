% labeledI global resmindeki 4 ya da 8 komþuluk bileþenlerini (type 
% deðiþkeniyle tanýmlý) tekrarlamalý yöntemle bulur.
function searchComponent(label,i,j,type)
	neighborsList = neighbors(i,j,label,type);
    listSize = size(neighborsList,2);
    
	for index=1:listSize
        searchComponent(label,neighborsList(1,index),neighborsList(2,index),type);
    end
end


function [neighborsList]=neighbors(i,j,label,type)
    global labeledI;
    neighborsList=[];
    [r, c] = size(labeledI);
    if type == 4        %4 komþuluk
        clockwiseNL = [-1 0 1  0;
                        0 1 0 -1];
    else                %8 komþuluk
        clockwiseNL = [-1 -1 -1 0 1 1  1  0;
                       -1  0  1 1 1 0 -1 -1];
    end
    for index= 1:size(clockwiseNL,2)
        x = i+clockwiseNL(1,index);
        y = j+clockwiseNL(2,index);
        %Deðeri sadece -1 olan komþularý dönmek için kontrol ve atama buraya
        %alýnmýþtýr. 
        if x>0 && x<=r && y>0 && y<=c && labeledI(x,y)==-1
            neighborsList = [neighborsList [x;y]];
            labeledI(x,y) = label;
        end
    end
end