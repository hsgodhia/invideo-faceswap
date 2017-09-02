function resultImg = reconstructImg(indexes, red, green, blue, targetImg)
    
    %Get the size of the target image
    [nr,nc] = size(indexes);
    
    %Find the replacement pixels columnwise
    [rows,cols] = find(indexes);
    indices_replacement_pixels = sub2ind([nr nc],rows,cols);
    
    %Get the values for the red channel
    resultImg_red = targetImg(:,:,1);
    resultImg_red = resultImg_red(:);
    resultImg_red(indices_replacement_pixels) = red;

    %Get the values for the green channel
    resultImg_green = targetImg(:,:,2);
    resultImg_green = resultImg_green(:);
    resultImg_green(indices_replacement_pixels) = green;
    
    %Get the values for the blue channel
    resultImg_blue = targetImg(:,:,3);
    resultImg_blue = resultImg_blue(:);
    resultImg_blue(indices_replacement_pixels) = blue;
    
    %Reconstruct the target image
    resultImg = cat(3,reshape(resultImg_red,[nr nc]),reshape(resultImg_green,[nr nc]),reshape(resultImg_blue,[nr nc])); 
end