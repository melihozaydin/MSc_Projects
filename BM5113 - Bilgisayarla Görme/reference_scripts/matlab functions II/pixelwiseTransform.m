
function [ImageTransformedLinear, ImageTransformedGamma, ImageTransformedShift]=pixelwiseTransform(Image,coef1, offset1, operation)

    ImageTransformedLinear =[];
    ImageTransformedGamma = [];
    ImageTransformedShift = [];

    switch operation
        case 'linear'
            ImageTransformedLinear = Image*coef1+offset1;
        case 'gamma'
            ImageTransformedGamma = Image.^coef1;
        otherwise
            ImageTransformedShift = Image+offset1;
    end


end