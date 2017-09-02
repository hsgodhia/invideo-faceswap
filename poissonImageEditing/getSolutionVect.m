function solVector = getSolutionVect(indexes, source, target, offsetX, offsetY)

    %Get the size of the source and target images
    [nr_src,nc_src] = size(source);
    [nr,nc] = size(target);
    
    %Find the replacement pixels columnwise in the target and source images
    [rows,cols] = find(indexes);
    indices_replacement_pixels = sub2ind([nr nc],rows,cols);
    indices_pixels_source_image = sub2ind([nr_src nc_src],rows-offsetY,cols-offsetX);
    
    
    
    
    %Get the colour values of the neighbouring pixels for each pixel in the source image
    %Assume reflection padding for neighbours outside the boundary of the source
    %image
    left_neighbour = [source(:,1) source(:,1:(end-1))];
    right_neighbour = [source(:,2:end) source(:,end)];
    top_neighbour = [source(1,:); source(1:(end-1),:)];
    bottom_neighbour = [source(2:end,:); source(end,:)];
    
    %Get the gradient of each replacement pixel in the source image
    gradient = 4*double(source) - double(top_neighbour) - double(left_neighbour) - double(right_neighbour) - double(bottom_neighbour);
    gradient = gradient(:); 
    solVector = gradient(indices_pixels_source_image);
      
    
    %Get the values of the neighbouring elements for each element in indexes
    %Assume zero values for neighbours outside the boundary of indexes
    left_neighbour_indexes = [zeros(nr,1) indexes(:,1:(end-1))];
    left_neighbour_indexes = left_neighbour_indexes(:);
    right_neighbour_indexes = [indexes(:,2:end) zeros(nr,1)];
    right_neighbour_indexes = right_neighbour_indexes(:);
    top_neighbour_indexes = [zeros(1,nc); indexes(1:(end-1),:)];
    top_neighbour_indexes = top_neighbour_indexes(:);
    bottom_neighbour_indexes = [indexes(2:end,:); zeros(1,nc)];
    bottom_neighbour_indexes = bottom_neighbour_indexes(:); 
    
    %top_neighbours_outside is a logical map of top neighbours of replacement pixels that lie outside the
    %replacement region
    top_neighbours_of_replacement_pixels = top_neighbour_indexes(indices_replacement_pixels); 
    top_neighbours_outside = top_neighbours_of_replacement_pixels == 0;
    %bottom_neighbours_outside is a logical map of bottom neighbours of replacement pixels that lie outside the
    %replacement region
    bottom_neighbours_of_replacement_pixels = bottom_neighbour_indexes(indices_replacement_pixels);
    bottom_neighbours_outside = bottom_neighbours_of_replacement_pixels == 0;   
    %right_neighbours_outside is a logical map of right neighbours of replacement pixels that lie outside the
    %replacement region
    right_neighbours_of_replacement_pixels = right_neighbour_indexes(indices_replacement_pixels);
    right_neighbours_outside = right_neighbours_of_replacement_pixels == 0;
    %left_neighbours_outside is a logical map of left neighbours of replacement pixles that lie outside the
    %replacement region
    left_neighbours_of_replacement_pixels = left_neighbour_indexes(indices_replacement_pixels);
    left_neighbours_outside = left_neighbours_of_replacement_pixels == 0;
    
    %Get the values of the neighbouring pixels for each pixel in the target image
    %Assume reflection padding for neighbours outside the boundary of the target
    %image
    left_neighbour_target = [target(:,1) target(:,1:(end-1))];
    left_neighbour_target = left_neighbour_target(:);
    right_neighbour_target = [target(:,2:end) target(:,end)];
    right_neighbour_target = right_neighbour_target(:);
    top_neighbour_target = [target(1,:); target(1:(end-1),:)];
    top_neighbour_target = top_neighbour_target(:);
    bottom_neighbour_target = [target(2:end,:); target(end,:)];
    bottom_neighbour_target = bottom_neighbour_target(:);    
    
    %N is the num of replacement pixels
    N = size(rows,1);
    
    %replacement_pixels is a linear indicing of the replacement pixels columnwise 
    replacement_pixels = [1:N]';
    
    %For each neighbor of a replacement pixel in the target image that lies outside the
    %replacement region, add the neighbour's color value in the target image to the solution vector
    %for top neighbours
    top_neighbours_colour_values = top_neighbour_target(indices_replacement_pixels);
    solVector(replacement_pixels(top_neighbours_outside)) = solVector(replacement_pixels(top_neighbours_outside))...
        + double((top_neighbours_colour_values(top_neighbours_outside)));
    %for right neighbours
    right_neighbours_colour_values = right_neighbour_target(indices_replacement_pixels);
    solVector(replacement_pixels(right_neighbours_outside)) = solVector(replacement_pixels(right_neighbours_outside))...
        + double((right_neighbours_colour_values(right_neighbours_outside)));
    %for bottom neighbours
    bottom_neighbours_colour_values = bottom_neighbour_target(indices_replacement_pixels);
    solVector(replacement_pixels(bottom_neighbours_outside)) = solVector(replacement_pixels(bottom_neighbours_outside))...
        + double((bottom_neighbours_colour_values(bottom_neighbours_outside)));
    %for left neighbours
    left_neighbours_colour_values = left_neighbour_target(indices_replacement_pixels);
    solVector(replacement_pixels(left_neighbours_outside)) = solVector(replacement_pixels(left_neighbours_outside))...
        + double((left_neighbours_colour_values(left_neighbours_outside)));
    
end