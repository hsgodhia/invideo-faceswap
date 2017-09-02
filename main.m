function main(source_video_file,replacement_image,target_video_file,img_bbox_no,x_global,y_global,mask)

    %Load the replacement image
    replacement_image = imread(replacement_image);
    [orig_height,orig_width,no_channels] = size(replacement_image);
    
    %Read in the video file
    video = vision.VideoFileReader(source_video_file);
    
    %Create the video writer
    readerObj = VideoReader(source_video_file);
    writerObj = VideoWriter(target_video_file);
    writerObj.FrameRate = readerObj.FrameRate;
    
    %Create a cascade detector object for detecting the face in the video
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
    
    %Open the video writer
    open(writerObj);
    
    %Read in the frame
    frame = readFrame(readerObj);
    videoFrame = step(video);
    %Run the detector on the current frame and store the returned
    %bounding box
    bbox = step(faceDetector,frame);
    if(isempty(bbox))
        disp('No bounding box found');
        return
    else
        bbox_no = img_bbox_no;
        %Expand the bounding box by 10 pixels in all 4 directions
        old_bbox = bbox;
        bbox(bbox_no,1) = max(old_bbox(bbox_no,1) - 10,1);
        bbox(bbox_no,2) = max(old_bbox(bbox_no,2) - 10,1);
        new_bbox_right_x = min(old_bbox(bbox_no,1)+old_bbox(bbox_no,3)+10,size(frame,2));
        new_bbox_down_y = min(old_bbox(bbox_no,2)+old_bbox(bbox_no,4)+10,size(frame,1));
        bbox(bbox_no,3) = new_bbox_right_x - bbox(bbox_no,1);
        bbox(bbox_no,4) = new_bbox_down_y - bbox(bbox_no,2);
        bbox_width = bbox(bbox_no,3);
        bbox_height = bbox(bbox_no,4);
    end
    
    %Replace the face in the first frame
    [resultImg,control_points] = face_replacement(frame,bbox,bbox_no,bbox_width,bbox_height,replacement_image,mask,orig_width,orig_height,x_global,y_global,0);
    
    %new_frame = im2frame(uint8(resultImg));
    writeVideo(writerObj,resultImg);

    %%%%%%%%%%%%%% Face Tracking using KLT ALGORITHM %%%%%%%%%%%%%%
    
    % Convert the first box into a list of 4 points
    % This is needed to be able to visualize the rotation of the object.
    bboxPoints = bbox2points(bbox(bbox_no, :));

    % Detect feature points in the face region.
    points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(bbox_no,:));

    % Create a point tracker and enable the bidirectional error constraint to
    % make it more robust in the presence of noise and clutter.
    pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

    % Initialize the tracker with the initial point locations and the initial
    % video frame.
    points = points.Location;
    initialize(pointTracker, points, videoFrame);

    % Make a copy of the points to be used for computing the geometric
    % transformation between the points in the previous and the current frames
    oldPoints = points;
    i =0 ;
    while ~isDone(video)

        i= i +1;
        % get the next frame
        videoFrame = step(video);
        frame = readFrame(readerObj);

        % Track the points. Note that some points may be lost.
        [points, isFound] = step(pointTracker, videoFrame);
        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);

        if size(visiblePoints, 1) >= 2 % need at least 2 points

            % Estimate the geometric transformation between the old points
            % and the new points and eliminate outliers
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box points
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Insert a bounding box around the object being tracked
            bboxPolygon = reshape(bboxPoints', 1, []);
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
                'LineWidth', 2);

            bbox_no = 1;
            %bbox_width = round(max(bboxPoints(:,1)) - min(bboxPoints(:,1)));
            %bbox_height = round(max(bboxPoints(:,2)) - min(bboxPoints(:,2)));
            bbox = [round(min(bboxPoints(:,1))),round((bboxPoints(1,2)+bboxPoints(2,2))/2.0),bbox_width,bbox_height];
            
            resultImg = face_replacement(frame,bbox,bbox_no,bbox_width,bbox_height,replacement_image,mask,orig_width,orig_height,x_global,y_global,i);
   
            %if(i==1)
            %    break;
            %end
          
            % Reset the points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

        % Display the annotated video frame using the video player object
        writeVideo(writerObj,resultImg);
    end

    % Clean up
    release(video);
    %release(videoPlayer);
    release(pointTracker);
    %close the writer object
    close(writerObj);
    
end