function indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY)
    
    %Find the indices of the pixels to be pasted in the source image
    [rows,cols] = find(mask);
    %Display an error prompt informing the user if the replacement region
    %extends beyond the boundaries of the target image
    if(max(rows+offsetY)>targetH || max(cols+offsetX)>targetW || min(rows+offsetY)<1 || min(cols+offsetX)<1)
        error('Replacement Region extends beyond the target image boundaries. Try resizing the source image or changing the offsets');
    end
    
    %Initialize a vector of zeros
    indexes = zeros(targetH,targetW);
    indexes = indexes(:);
    
    %Linear index the replacement pixels from top to bottom
    indices_in_target_image = sub2ind([targetH targetW],rows+offsetY,cols+offsetX);
    indexes(indices_in_target_image) = 1:size(rows,1);
    
    %Reshape indexes into a matrix of size of target image
    indexes = reshape(indexes,[targetH targetW]);
    
end
