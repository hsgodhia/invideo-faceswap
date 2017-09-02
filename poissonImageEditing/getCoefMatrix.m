function coefM = getCoefMatrix(indexes)
    
    %Get the size of the target image
    [nr,nc] = size(indexes);
    
    %Find the replacement pixels columnwise
    [rows,cols] = find(indexes);
    indices_replacement_pixels = sub2ind([nr nc],rows,cols);
    
    %Get the values of the neighbouring elements for each element in indexes
    %Assume zero values for neighbours outside the boundary of indexes
    left_neighbour = [zeros(nr,1) indexes(:,1:(end-1))];
    left_neighbour = left_neighbour(:);
    right_neighbour = [indexes(:,2:end) zeros(nr,1)];
    right_neighbour = right_neighbour(:);
    top_neighbour = [zeros(1,nc); indexes(1:(end-1),:)];
    top_neighbour = top_neighbour(:);
    bottom_neighbour = [indexes(2:end,:); zeros(1,nc)];
    bottom_neighbour = bottom_neighbour(:);    
    
    
    %top_neighbours_inside is a logical map of top neighbours of replacement pixles that lie within the
    %replacement region
    top_neighbours_of_replacement_pixels = top_neighbour(indices_replacement_pixels); 
    top_neighbours_inside = top_neighbours_of_replacement_pixels > 0;
    %bottom_neighbours_inside is a logical map of bottom neighbours of replacement pixles that lie within the
    %replacement region
    bottom_neighbours_of_replacement_pixels = bottom_neighbour(indices_replacement_pixels);
    bottom_neighbours_inside = bottom_neighbours_of_replacement_pixels > 0;   
    %right_neighbours_inside is a logical map of right neighbours of replacement pixles that lie within the
    %replacement region
    right_neighbours_of_replacement_pixels = right_neighbour(indices_replacement_pixels);
    right_neighbours_inside = right_neighbours_of_replacement_pixels > 0;
    %left_neighbours_inside is a logical map of left neighbours of replacement pixles that lie within the
    %replacement region
    left_neighbours_of_replacement_pixels = left_neighbour(indices_replacement_pixels);
    left_neighbours_inside = left_neighbours_of_replacement_pixels > 0;
    
    
    
    %Generate coefficients for each replacement pixel
    %size of coefM = N*N where N is the num of replacement pixels
    N = size(rows,1);
    
    %replacement_pixels is a linear indicing of the replacement pixels columnwise 
    replacement_pixels = [1:N]';
    
    %coefM_row_index and coefM_col_index are the row and column indices of
    %the coefficients of the replacement pixels and their neighbours
    %respectively
    coefM_row_index = [replacement_pixels;...
        replacement_pixels(top_neighbours_inside);...
        replacement_pixels(right_neighbours_inside);...
        replacement_pixels(bottom_neighbours_inside);...
        replacement_pixels(left_neighbours_inside)];
    coefM_col_index = [replacement_pixels;
        top_neighbours_of_replacement_pixels(top_neighbours_inside);...
        right_neighbours_of_replacement_pixels(right_neighbours_inside);...
        bottom_neighbours_of_replacement_pixels(bottom_neighbours_inside);...
        left_neighbours_of_replacement_pixels(left_neighbours_inside)];
    %coefficients is the respective coefficients of the replacement pixels
    %and their neighbours respectively
    coefficients = [4*ones(N,1);...
        -1*ones(nnz(top_neighbours_inside),1);...
        -1*ones(nnz(right_neighbours_inside),1);...
        -1*ones(nnz(bottom_neighbours_inside),1);...
        -1*ones(nnz(left_neighbours_inside),1)];
    
    %Create  a sparse matrix for coefM
    coefM = sparse(coefM_row_index,coefM_col_index,coefficients);
end
