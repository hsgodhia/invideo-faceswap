
function control_points = find_facial_keypoints(bbox,frame)
    
    load('codebook.mat');

    %Similarity Threshold for matching histogram features using chi-squared
    %distance
    threshold = 1;

    %Define the ratio of the cell size to the image size
    ratio = 32.0/100;
    
    %Find the cell size for the current bounding box while maintaining the ratio
    cell_width = round(ratio*bbox(3));
    cell_height = round(ratio*bbox(4));
    
    %figure;
    %imshow(frame);
    %axis image;
    
    %Initialize the maps for the 12 control points
    first_pt_hairline_map = zeros(size(frame,1),size(frame,2));
    second_pt_hairline_map = zeros(size(frame,1),size(frame,2));
    third_pt_hairline_map = zeros(size(frame,1),size(frame,2));
    first_pt_face_boundary_map = zeros(size(frame,1),size(frame,2));
    second_pt_face_boundary_map = zeros(size(frame,1),size(frame,2));
    third_pt_face_boundary_map = zeros(size(frame,1),size(frame,2));
    fourth_pt_face_boundary_map = zeros(size(frame,1),size(frame,2));
    fifth_pt_face_boundary_map = zeros(size(frame,1),size(frame,2));
    left_eye_map = zeros(size(frame,1),size(frame,2));
    right_eye_map = zeros(size(frame,1),size(frame,2));
    nose_map = zeros(size(frame,1),size(frame,2));
    mouth_map = zeros(size(frame,1),size(frame,2));
    
    %Divide the bounding box into overlapping cells
    for j=1:2*(round(bbox(4)/double(cell_height)))-1
        for i=1:2*(round(bbox(3)/double(cell_width)))-1
            x_upper_left = min(bbox(1)+(i-1)*(cell_width/2.0),bbox(3)+bbox(1));
            y_upper_left = min(bbox(2)+(j-1)*(cell_height/2.0),bbox(4)+bbox(2));
            x_center = x_upper_left + cell_width/2.0;
            y_center = y_upper_left + cell_height/2.0;
            
            %Extract HOG features for each cell
            [featureVector,validPoints,hogVisualization] = extractHOGFeatures(frame,[x_center y_center],'CellSize',[cell_width cell_height]);
            
            %Fetch the top k matches from the codebook
            k = 8;
            [IDX,D] = knnsearch(codebook,featureVector,'K',k,'Distance',@chi_squared_distance);
            IDX = IDX(1,D<threshold);
            D = D(1,D<threshold);
            
            %Build the control point maps
            for match_no=1:size(IDX,2)
                index_in_codebook = IDX(1,match_no);
                
                %Find the distance of the part it matched in the codebook from every other
                %part in the same codebook training image
                %If it is the first point on the hairline
                if(mod(index_in_codebook,12)==1)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = 0;
                    y_dist_to_first_pt_hairline = 0;
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook+1) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook+2) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook+3) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook+3) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook+4) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook+4) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook+5) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook+5) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook+6) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook+6) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook+7) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook+7) - y_coord(index_in_codebook);
                %If it is the second point on the hairline
                elseif(mod(index_in_codebook,12)==2)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = 0;
                    y_dist_to_second_pt_hairline = 0;
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook+1) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook+2) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook+3) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook+3) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook+4) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook+4) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook+5) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook+5) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook+6) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook+6) - y_coord(index_in_codebook);
                %If it is the third point on the hairline
                elseif(mod(index_in_codebook,12)==3)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = 0;
                    y_dist_to_third_pt_hairline = 0;
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook+1) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook+2) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook+3) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook+3) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook+4) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook+4) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook+5) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook+5) - y_coord(index_in_codebook);
                %If it is the first point on the face boundary
                elseif(mod(index_in_codebook,12)==4)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = 0;
                    y_dist_to_first_pt_face_boundary = 0;
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook+1) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook+2) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook+3) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook+3) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook+4) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook+4) - y_coord(index_in_codebook);
                %If it is the second point on the face boundary
                elseif(mod(index_in_codebook,12)==5)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-4) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = 0;
                    y_dist_to_second_pt_face_boundary = 0;
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook+1) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook+2) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook+3) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook+3) - y_coord(index_in_codebook);
                %If it is the third point on the face boundary
                elseif(mod(index_in_codebook,12)==6)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-5) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-4) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = 0;
                    y_dist_to_third_pt_face_boundary = 0;
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook+1) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook+2) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                %If it is the fourth point on the face boundary
                elseif(mod(index_in_codebook,12)==7)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-6) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-6) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-5) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = 0;
                    y_dist_to_fourth_pt_face_boundary = 0;
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook+1) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                %If it is the fifth point on the face boundary
                elseif(mod(index_in_codebook,12)==8)
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-7) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-7) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-6) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-6) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-4) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = 0;
                    y_dist_to_fifth_pt_face_boundary = 0;
                %If it is a left eye
                elseif(mod(index_in_codebook,12)==9)
                    x_dist_to_nose = x_coord(index_in_codebook+2) - x_coord(index_in_codebook); 
                    y_dist_to_nose = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                    x_dist_to_left_eye = 0; 
                    y_dist_to_left_eye = 0;
                    x_dist_to_right_eye = x_coord(index_in_codebook+1) - x_coord(index_in_codebook); 
                    y_dist_to_right_eye = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    x_dist_to_mouth = x_coord(index_in_codebook+3) - x_coord(index_in_codebook); 
                    y_dist_to_mouth = y_coord(index_in_codebook+3) - y_coord(index_in_codebook);
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-8) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-8) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-7) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-7) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-6) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = y_coord(index_in_codebook-6) - y_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-5) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook-4) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                %If it is a right eye
                elseif(mod(index_in_codebook,12)==10)
                    x_dist_to_nose = x_coord(index_in_codebook+1) - x_coord(index_in_codebook); 
                    y_dist_to_nose = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    x_dist_to_left_eye = x_coord(index_in_codebook-1) - x_coord(index_in_codebook);
                    y_dist_to_left_eye = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_right_eye = 0;
                    y_dist_to_right_eye = 0;
                    x_dist_to_mouth = x_coord(index_in_codebook+2) - x_coord(index_in_codebook); 
                    y_dist_to_mouth = y_coord(index_in_codebook+2) - y_coord(index_in_codebook);
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-9) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-9) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-8) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-8) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-7) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = y_coord(index_in_codebook-7) - y_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-6) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-6) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook-5) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook-4) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook-2) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                %If it is a nose
                elseif(mod(index_in_codebook,12)==11)
                    x_dist_to_nose = 0; 
                    y_dist_to_nose = 0;
                    x_dist_to_left_eye = x_coord(index_in_codebook-2) - x_coord(index_in_codebook); 
                    y_dist_to_left_eye = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_right_eye = x_coord(index_in_codebook-1) - x_coord(index_in_codebook); 
                    y_dist_to_right_eye = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_mouth = x_coord(index_in_codebook+1) - x_coord(index_in_codebook); 
                    y_dist_to_mouth = y_coord(index_in_codebook+1) - y_coord(index_in_codebook);
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-10) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-10) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-9) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-9) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-8) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = y_coord(index_in_codebook-8) - y_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-7) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-7) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook-6) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook-6) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook-5) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook-4) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                %If it is the mouth
                elseif(mod(index_in_codebook,12)==0)
                    x_dist_to_nose = x_coord(index_in_codebook-1) - x_coord(index_in_codebook); 
                    y_dist_to_nose = y_coord(index_in_codebook-1) - y_coord(index_in_codebook);
                    x_dist_to_left_eye = x_coord(index_in_codebook-3) - x_coord(index_in_codebook);
                    y_dist_to_left_eye = y_coord(index_in_codebook-3) - y_coord(index_in_codebook);
                    x_dist_to_right_eye = x_coord(index_in_codebook-2) - x_coord(index_in_codebook); 
                    y_dist_to_right_eye = y_coord(index_in_codebook-2) - y_coord(index_in_codebook);
                    x_dist_to_mouth = 0;
                    y_dist_to_mouth = 0;
                    % 3 points on the hairline
                    x_dist_to_first_pt_hairline = x_coord(index_in_codebook-11) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_hairline = y_coord(index_in_codebook-11) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_hairline = x_coord(index_in_codebook-10) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_hairline = y_coord(index_in_codebook-10) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_hairline = x_coord(index_in_codebook-9) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_hairline = y_coord(index_in_codebook-9) - y_coord(index_in_codebook);
                    % 5 points on the face boundary
                    x_dist_to_first_pt_face_boundary = x_coord(index_in_codebook-8) - x_coord(index_in_codebook);
                    y_dist_to_first_pt_face_boundary = y_coord(index_in_codebook-8) - y_coord(index_in_codebook);
                    x_dist_to_second_pt_face_boundary = x_coord(index_in_codebook-7) - x_coord(index_in_codebook);
                    y_dist_to_second_pt_face_boundary = y_coord(index_in_codebook-7) - y_coord(index_in_codebook);
                    x_dist_to_third_pt_face_boundary = x_coord(index_in_codebook-6) - x_coord(index_in_codebook);
                    y_dist_to_third_pt_face_boundary = y_coord(index_in_codebook-6) - y_coord(index_in_codebook);
                    x_dist_to_fourth_pt_face_boundary = x_coord(index_in_codebook-5) - x_coord(index_in_codebook);
                    y_dist_to_fourth_pt_face_boundary = y_coord(index_in_codebook-5) - y_coord(index_in_codebook);
                    x_dist_to_fifth_pt_face_boundary = x_coord(index_in_codebook-4) - x_coord(index_in_codebook);
                    y_dist_to_fifth_pt_face_boundary = y_coord(index_in_codebook-4) - y_coord(index_in_codebook);
                end
                
                %Maintain the same distance ratio as in the training image
                dist_width_ratio = double(bbox(3))/bounding_box(index_in_codebook,3);
                dist_height_ratio = double(bbox(4))/bounding_box(index_in_codebook,4);
                part_id = mod(index_in_codebook,12); 
                if(part_id==9||part_id==10||part_id==11||part_id==0)    
                    %Vote for the nose
                    x_nose_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_nose));
                    y_nose_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_nose));
                    if(x_nose_vote_pos<=size(frame,2) && y_nose_vote_pos<=size(frame,1) && x_nose_vote_pos>=1 && y_nose_vote_pos>=1)
                        nose_map(y_nose_vote_pos,x_nose_vote_pos) = nose_map(y_nose_vote_pos,x_nose_vote_pos) + exp(-D(1,match_no));
                        %Blur the nose map at the voted position using a 5*5 gaussian filter
                        x_nose_patch_coords = max(x_nose_vote_pos-5,1):min(x_nose_vote_pos+5,size(nose_map,2));
                        y_nose_patch_coords = max(y_nose_vote_pos-5,1):min(y_nose_vote_pos+5,size(nose_map,1));
                        nose_map(y_nose_patch_coords,x_nose_patch_coords) = imgaussfilt(nose_map(y_nose_patch_coords,x_nose_patch_coords),0.5,'FilterSize',5);
                    end

                    %Vote for the left eye
                    x_left_eye_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_left_eye));
                    y_left_eye_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_left_eye));
                    if(x_left_eye_vote_pos<=size(frame,2) && y_left_eye_vote_pos<=size(frame,1) && x_left_eye_vote_pos>=1 && y_left_eye_vote_pos>=1)
                        left_eye_map(y_left_eye_vote_pos,x_left_eye_vote_pos) = left_eye_map(y_left_eye_vote_pos,x_left_eye_vote_pos) + exp(-D(1,match_no));
                        %Blur the left eye map at the voted position using a 5*5 gaussian filter
                        x_left_eye_patch_coords = max(x_left_eye_vote_pos-5,1):min(x_left_eye_vote_pos+5,size(left_eye_map,2));
                        y_left_eye_patch_coords = max(y_left_eye_vote_pos-5,1):min(y_left_eye_vote_pos+5,size(left_eye_map,1));
                        left_eye_map(y_left_eye_patch_coords,x_left_eye_patch_coords) = imgaussfilt(left_eye_map(y_left_eye_patch_coords,x_left_eye_patch_coords),0.5,'FilterSize',5);
                    end

                    %Vote for the right eye
                    x_right_eye_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_right_eye));
                    y_right_eye_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_right_eye));
                    if(x_right_eye_vote_pos<=size(frame,2) && y_right_eye_vote_pos<=size(frame,1) && x_right_eye_vote_pos>=1 && y_right_eye_vote_pos>=1)
                        right_eye_map(y_right_eye_vote_pos,x_right_eye_vote_pos) = right_eye_map(y_right_eye_vote_pos,x_right_eye_vote_pos) + exp(-D(1,match_no));
                        %Blur the right eye map at the voted position using a 5*5 gaussian filter
                        x_right_eye_patch_coords = max(x_right_eye_vote_pos-5,1):min(x_right_eye_vote_pos+5,size(right_eye_map,2));
                        y_right_eye_patch_coords = max(y_right_eye_vote_pos-5,1):min(y_right_eye_vote_pos+5,size(right_eye_map,1));
                        right_eye_map(y_right_eye_patch_coords,x_right_eye_patch_coords) = imgaussfilt(right_eye_map(y_right_eye_patch_coords,x_right_eye_patch_coords),0.5,'FilterSize',5);
                    end
                    
                    %Vote for the mouth
                    x_mouth_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_mouth));
                    y_mouth_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_mouth));
                    if(x_mouth_vote_pos<=size(frame,2) && y_mouth_vote_pos<=size(frame,1) && x_mouth_vote_pos>=1 && y_mouth_vote_pos>=1)
                        mouth_map(y_mouth_vote_pos,x_mouth_vote_pos) = mouth_map(y_mouth_vote_pos,x_mouth_vote_pos) + exp(-D(1,match_no));
                        %Blur the mouth map at the voted position using a 5*5 gaussian filter
                        x_mouth_patch_coords = max(x_mouth_vote_pos-5,1):min(x_mouth_vote_pos+5,size(mouth_map,2));
                        y_mouth_patch_coords = max(y_mouth_vote_pos-5,1):min(y_mouth_vote_pos+5,size(mouth_map,1));
                        mouth_map(y_mouth_patch_coords,x_mouth_patch_coords) = imgaussfilt(mouth_map(y_mouth_patch_coords,x_mouth_patch_coords),0.5,'FilterSize',5);
                    end
                end
                
                %Vote for the first point on the hairline
                x_first_pt_hairline_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_first_pt_hairline));
                y_first_pt_hairline_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_first_pt_hairline));
                if(x_first_pt_hairline_vote_pos<=size(frame,2) && y_first_pt_hairline_vote_pos<=size(frame,1) && x_first_pt_hairline_vote_pos>=1 && y_first_pt_hairline_vote_pos>=1)
                    first_pt_hairline_map(y_first_pt_hairline_vote_pos,x_first_pt_hairline_vote_pos) = first_pt_hairline_map(y_first_pt_hairline_vote_pos,x_first_pt_hairline_vote_pos) + exp(-D(1,match_no));
                    %Blur the first point of hairline map at the voted position using a 5*5 gaussian filter
                    x_first_pt_hairline_patch_coords = max(x_first_pt_hairline_vote_pos-5,1):min(x_first_pt_hairline_vote_pos+5,size(first_pt_hairline_map,2));
                    y_first_pt_hairline_patch_coords = max(y_first_pt_hairline_vote_pos-5,1):min(y_first_pt_hairline_vote_pos+5,size(first_pt_hairline_map,1));
                    first_pt_hairline_map(y_first_pt_hairline_patch_coords,x_first_pt_hairline_patch_coords) = imgaussfilt(first_pt_hairline_map(y_first_pt_hairline_patch_coords,x_first_pt_hairline_patch_coords),0.5,'FilterSize',5);
                end
                
                %Vote for the second point on the hairline
                x_second_pt_hairline_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_second_pt_hairline));
                y_second_pt_hairline_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_second_pt_hairline));
                if(x_second_pt_hairline_vote_pos<=size(frame,2) && y_second_pt_hairline_vote_pos<=size(frame,1) && x_second_pt_hairline_vote_pos>=1 && y_second_pt_hairline_vote_pos>=1)
                    second_pt_hairline_map(y_second_pt_hairline_vote_pos,x_second_pt_hairline_vote_pos) = second_pt_hairline_map(y_second_pt_hairline_vote_pos,x_second_pt_hairline_vote_pos) + exp(-D(1,match_no));
                    %Blur the second point of hairline map at the voted position using a 5*5 gaussian filter
                    x_second_pt_hairline_patch_coords = max(x_second_pt_hairline_vote_pos-5,1):min(x_second_pt_hairline_vote_pos+5,size(second_pt_hairline_map,2));
                    y_second_pt_hairline_patch_coords = max(y_second_pt_hairline_vote_pos-5,1):min(y_second_pt_hairline_vote_pos+5,size(second_pt_hairline_map,1));
                    second_pt_hairline_map(y_second_pt_hairline_patch_coords,x_second_pt_hairline_patch_coords) = imgaussfilt(second_pt_hairline_map(y_second_pt_hairline_patch_coords,x_second_pt_hairline_patch_coords),0.5,'FilterSize',5);
                end
                
                %Vote for the third point on the hairline
                x_third_pt_hairline_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_third_pt_hairline));
                y_third_pt_hairline_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_third_pt_hairline));
                if(x_third_pt_hairline_vote_pos<=size(frame,2) && y_third_pt_hairline_vote_pos<=size(frame,1) && x_third_pt_hairline_vote_pos>=1 && y_third_pt_hairline_vote_pos>=1)
                    third_pt_hairline_map(y_third_pt_hairline_vote_pos,x_third_pt_hairline_vote_pos) = third_pt_hairline_map(y_third_pt_hairline_vote_pos,x_third_pt_hairline_vote_pos) + exp(-D(1,match_no));
                    %Blur the third point of hairline map at the voted position using a 5*5 gaussian filter
                    x_third_pt_hairline_patch_coords = max(x_third_pt_hairline_vote_pos-5,1):min(x_third_pt_hairline_vote_pos+5,size(third_pt_hairline_map,2));
                    y_third_pt_hairline_patch_coords = max(y_third_pt_hairline_vote_pos-5,1):min(y_third_pt_hairline_vote_pos+5,size(third_pt_hairline_map,1));
                    third_pt_hairline_map(y_third_pt_hairline_patch_coords,x_third_pt_hairline_patch_coords) = imgaussfilt(third_pt_hairline_map(y_third_pt_hairline_patch_coords,x_third_pt_hairline_patch_coords),0.5,'FilterSize',5);
                end
                
                %Vote for the first point on the face_boundary
                x_first_pt_face_boundary_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_first_pt_face_boundary));
                y_first_pt_face_boundary_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_first_pt_face_boundary));
                if(x_first_pt_face_boundary_vote_pos<=size(frame,2) && y_first_pt_face_boundary_vote_pos<=size(frame,1) && x_first_pt_face_boundary_vote_pos>=1 && y_first_pt_face_boundary_vote_pos>=1)
                    first_pt_face_boundary_map(y_first_pt_face_boundary_vote_pos,x_first_pt_face_boundary_vote_pos) = first_pt_face_boundary_map(y_first_pt_face_boundary_vote_pos,x_first_pt_face_boundary_vote_pos) + exp(-D(1,match_no));
                    %Blur the first point of face_boundary map at the voted position using a 5*5 gaussian filter
                    x_first_pt_face_boundary_patch_coords = max(x_first_pt_face_boundary_vote_pos-5,1):min(x_first_pt_face_boundary_vote_pos+5,size(first_pt_face_boundary_map,2));
                    y_first_pt_face_boundary_patch_coords = max(y_first_pt_face_boundary_vote_pos-5,1):min(y_first_pt_face_boundary_vote_pos+5,size(first_pt_face_boundary_map,1));
                    first_pt_face_boundary_map(y_first_pt_face_boundary_patch_coords,x_first_pt_face_boundary_patch_coords) = imgaussfilt(first_pt_face_boundary_map(y_first_pt_face_boundary_patch_coords,x_first_pt_face_boundary_patch_coords),0.5,'FilterSize',5);
                end
                
                %Vote for the second point on the face_boundary
                x_second_pt_face_boundary_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_second_pt_face_boundary));
                y_second_pt_face_boundary_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_second_pt_face_boundary));
                if(x_second_pt_face_boundary_vote_pos<=size(frame,2) && y_second_pt_face_boundary_vote_pos<=size(frame,1) && x_second_pt_face_boundary_vote_pos>=1 && y_second_pt_face_boundary_vote_pos>=1)
                    second_pt_face_boundary_map(y_second_pt_face_boundary_vote_pos,x_second_pt_face_boundary_vote_pos) = second_pt_face_boundary_map(y_second_pt_face_boundary_vote_pos,x_second_pt_face_boundary_vote_pos) + exp(-D(1,match_no));
                    %Blur the second point of face_boundary map at the voted position using a 5*5 gaussian filter
                    x_second_pt_face_boundary_patch_coords = max(x_second_pt_face_boundary_vote_pos-5,1):min(x_second_pt_face_boundary_vote_pos+5,size(second_pt_face_boundary_map,2));
                    y_second_pt_face_boundary_patch_coords = max(y_second_pt_face_boundary_vote_pos-5,1):min(y_second_pt_face_boundary_vote_pos+5,size(second_pt_face_boundary_map,1));
                    second_pt_face_boundary_map(y_second_pt_face_boundary_patch_coords,x_second_pt_face_boundary_patch_coords) = imgaussfilt(second_pt_face_boundary_map(y_second_pt_face_boundary_patch_coords,x_second_pt_face_boundary_patch_coords),0.5,'FilterSize',5);
                end
                
                %Vote for the third point on the face_boundary
                x_third_pt_face_boundary_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_third_pt_face_boundary));
                y_third_pt_face_boundary_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_third_pt_face_boundary));
                if(x_third_pt_face_boundary_vote_pos<=size(frame,2) && y_third_pt_face_boundary_vote_pos<=size(frame,1) && x_third_pt_face_boundary_vote_pos>=1 && y_third_pt_face_boundary_vote_pos>=1)
                    third_pt_face_boundary_map(y_third_pt_face_boundary_vote_pos,x_third_pt_face_boundary_vote_pos) = third_pt_face_boundary_map(y_third_pt_face_boundary_vote_pos,x_third_pt_face_boundary_vote_pos) + exp(-D(1,match_no));
                    %Blur the third point of face_boundary map at the voted position using a 5*5 gaussian filter
                    x_third_pt_face_boundary_patch_coords = max(x_third_pt_face_boundary_vote_pos-5,1):min(x_third_pt_face_boundary_vote_pos+5,size(third_pt_face_boundary_map,2));
                    y_third_pt_face_boundary_patch_coords = max(y_third_pt_face_boundary_vote_pos-5,1):min(y_third_pt_face_boundary_vote_pos+5,size(third_pt_face_boundary_map,1));
                    third_pt_face_boundary_map(y_third_pt_face_boundary_patch_coords,x_third_pt_face_boundary_patch_coords) = imgaussfilt(third_pt_face_boundary_map(y_third_pt_face_boundary_patch_coords,x_third_pt_face_boundary_patch_coords),0.5,'FilterSize',5);
                end

                %Vote for the fourth point on the face_boundary
                x_fourth_pt_face_boundary_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_fourth_pt_face_boundary));
                y_fourth_pt_face_boundary_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_fourth_pt_face_boundary));
                if(x_fourth_pt_face_boundary_vote_pos<=size(frame,2) && y_fourth_pt_face_boundary_vote_pos<=size(frame,1) && x_fourth_pt_face_boundary_vote_pos>=1 && y_fourth_pt_face_boundary_vote_pos>=1)
                    fourth_pt_face_boundary_map(y_fourth_pt_face_boundary_vote_pos,x_fourth_pt_face_boundary_vote_pos) = fourth_pt_face_boundary_map(y_fourth_pt_face_boundary_vote_pos,x_fourth_pt_face_boundary_vote_pos) + exp(-D(1,match_no));
                    %Blur the fourth point of face_boundary map at the voted position using a 5*5 gaussian filter
                    x_fourth_pt_face_boundary_patch_coords = max(x_fourth_pt_face_boundary_vote_pos-5,1):min(x_fourth_pt_face_boundary_vote_pos+5,size(fourth_pt_face_boundary_map,2));
                    y_fourth_pt_face_boundary_patch_coords = max(y_fourth_pt_face_boundary_vote_pos-5,1):min(y_fourth_pt_face_boundary_vote_pos+5,size(fourth_pt_face_boundary_map,1));
                    fourth_pt_face_boundary_map(y_fourth_pt_face_boundary_patch_coords,x_fourth_pt_face_boundary_patch_coords) = imgaussfilt(fourth_pt_face_boundary_map(y_fourth_pt_face_boundary_patch_coords,x_fourth_pt_face_boundary_patch_coords),0.5,'FilterSize',5);
                end
                
                %Vote for the fifth point on the face_boundary
                x_fifth_pt_face_boundary_vote_pos = round(x_center+(dist_width_ratio*x_dist_to_fifth_pt_face_boundary));
                y_fifth_pt_face_boundary_vote_pos = round(y_center+(dist_height_ratio*y_dist_to_fifth_pt_face_boundary));
                if(x_fifth_pt_face_boundary_vote_pos<=size(frame,2) && y_fifth_pt_face_boundary_vote_pos<=size(frame,1) && x_fifth_pt_face_boundary_vote_pos>=1 && y_fifth_pt_face_boundary_vote_pos>=1)
                    fifth_pt_face_boundary_map(y_fifth_pt_face_boundary_vote_pos,x_fifth_pt_face_boundary_vote_pos) = fifth_pt_face_boundary_map(y_fifth_pt_face_boundary_vote_pos,x_fifth_pt_face_boundary_vote_pos) + exp(-D(1,match_no));
                    %Blur the fifth point of face_boundary map at the voted position using a 5*5 gaussian filter
                    x_fifth_pt_face_boundary_patch_coords = max(x_fifth_pt_face_boundary_vote_pos-5,1):min(x_fifth_pt_face_boundary_vote_pos+5,size(fifth_pt_face_boundary_map,2));
                    y_fifth_pt_face_boundary_patch_coords = max(y_fifth_pt_face_boundary_vote_pos-5,1):min(y_fifth_pt_face_boundary_vote_pos+5,size(fifth_pt_face_boundary_map,1));
                    fifth_pt_face_boundary_map(y_fifth_pt_face_boundary_patch_coords,x_fifth_pt_face_boundary_patch_coords) = imgaussfilt(fifth_pt_face_boundary_map(y_fifth_pt_face_boundary_patch_coords,x_fifth_pt_face_boundary_patch_coords),0.5,'FilterSize',5);
                end
                
            end
            
         end
    end           
    
    %Find the positions of the control points
    new_nose_map = nose_map>(mean(mean(nose_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3)))));
    [y_nose,x_nose] = find(new_nose_map);
    new_left_eye_map = left_eye_map>mean(mean(left_eye_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_left_eye,x_left_eye] = find(new_left_eye_map);
    new_right_eye_map = right_eye_map>mean(mean(right_eye_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_right_eye,x_right_eye] = find(new_right_eye_map);
    new_mouth_map = mouth_map>mean(mean(mouth_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_mouth,x_mouth] = find(new_mouth_map);
    % 3 points on the hairline
    new_first_pt_hairline_map = first_pt_hairline_map>mean(mean(first_pt_hairline_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_first_pt_hairline,x_first_pt_hairline] = find(new_first_pt_hairline_map);
    new_second_pt_hairline_map = second_pt_hairline_map>mean(mean(second_pt_hairline_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_second_pt_hairline,x_second_pt_hairline] = find(new_second_pt_hairline_map);
    new_third_pt_hairline_map = third_pt_hairline_map>mean(mean(third_pt_hairline_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_third_pt_hairline,x_third_pt_hairline] = find(new_third_pt_hairline_map);
    % 5 points on the face boundary
    new_first_pt_face_boundary_map = first_pt_face_boundary_map>mean(mean(first_pt_face_boundary_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_first_pt_face_boundary,x_first_pt_face_boundary] = find(new_first_pt_face_boundary_map);
    new_second_pt_face_boundary_map = second_pt_face_boundary_map>mean(mean(second_pt_face_boundary_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_second_pt_face_boundary,x_second_pt_face_boundary] = find(new_second_pt_face_boundary_map);
    new_third_pt_face_boundary_map = third_pt_face_boundary_map>mean(mean(third_pt_face_boundary_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_third_pt_face_boundary,x_third_pt_face_boundary] = find(new_third_pt_face_boundary_map); 
    new_fourth_pt_face_boundary_map = fourth_pt_face_boundary_map>mean(mean(fourth_pt_face_boundary_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_fourth_pt_face_boundary,x_fourth_pt_face_boundary] = find(new_fourth_pt_face_boundary_map);
    new_fifth_pt_face_boundary_map = fifth_pt_face_boundary_map>mean(mean(fifth_pt_face_boundary_map(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3))));
    [y_fifth_pt_face_boundary,x_fifth_pt_face_boundary] = find(new_fifth_pt_face_boundary_map); 
    
    %Return the control points
    control_points = [  mean(x_first_pt_hairline),mean(y_first_pt_hairline);...
                        mean(x_second_pt_hairline),mean(y_second_pt_hairline);...
                        mean(x_third_pt_hairline),mean(y_third_pt_hairline);...
                        mean(x_first_pt_face_boundary),mean(y_first_pt_face_boundary);...
                        mean(x_second_pt_face_boundary),mean(y_second_pt_face_boundary);...
                        mean(x_third_pt_face_boundary),mean(y_third_pt_face_boundary);...
                        mean(x_fourth_pt_face_boundary),mean(y_fourth_pt_face_boundary);...
                        mean(x_fifth_pt_face_boundary),mean(y_fifth_pt_face_boundary);...
                        mean(x_left_eye),mean(y_left_eye);...
                        mean(x_right_eye),mean(y_right_eye);...
                        mean(x_nose),mean(y_nose);...
                        mean(x_mouth),mean(y_mouth);...
                     ];
                        
    %plot the control points
    %{
    hold on;
    plot(mean(x_first_pt_hairline),mean(y_first_pt_hairline),'r.');
    plot(mean(x_second_pt_hairline),mean(y_second_pt_hairline),'r.');
    plot(mean(x_third_pt_hairline),mean(y_third_pt_hairline),'r.');
    plot(mean(x_first_pt_face_boundary),mean(y_first_pt_face_boundary),'r.');
    plot(mean(x_second_pt_face_boundary),mean(y_second_pt_face_boundary),'r.');
    plot(mean(x_third_pt_face_boundary),mean(y_third_pt_face_boundary),'r.');
    plot(mean(x_fourth_pt_face_boundary),mean(y_fourth_pt_face_boundary),'r.');
    plot(mean(x_fifth_pt_face_boundary),mean(y_fifth_pt_face_boundary),'r.');
    plot(mean(x_left_eye),mean(y_left_eye),'r.');
    plot(mean(x_right_eye),mean(y_right_eye),'r.');
    plot(mean(x_nose),mean(y_nose),'r.');
    plot(mean(x_mouth),mean(y_mouth),'r.');
    hold off;
    %}
end

 

function csd = chi_squared_distance(X,Y)
    csd = [];
    for i=1:size(Y,1)
        csd = [csd;sum(((Y(i,:)-X).*(Y(i,:)-X))./(X+Y(i,:)))/2.0;];
    end
end
