function [resultImg,control_points] = face_replacement(frame,bbox,bbox_no,bbox_width,bbox_height,replacement_image,mask,orig_width,orig_height,x_global,y_global,i)

        disp(i);
        
        %Run facial feature detector and get the control points for the
        %face boundary, left eye, right eye, nose and the mouth in this order
        %Translate the points wrt the left corner of the bounding box as
        %the new origin
        control_points = bsxfun(@minus,find_facial_keypoints(bbox(bbox_no,:),frame),[bbox(bbox_no,1),bbox(bbox_no,2)]);
        frame_bbox_end_pts = bsxfun(@minus,...
                                   [bbox(bbox_no,1),bbox(bbox_no,2);...
                                    bbox(bbox_no,1)+bbox_width,bbox(bbox_no,2);...
                                    bbox(bbox_no,1),bbox(bbox_no,2)+bbox_height;...
                                    bbox(bbox_no,1)+bbox_width,bbox(bbox_no,2)+bbox_height],...
                                    [bbox(bbox_no,1),bbox(bbox_no,2)]);
    
        %Resize the image and the mask to the same size as that of the bounding box in
        %the frame
        replacement_image = imresize(replacement_image,[bbox_height bbox_width]);
        mask = imresize(mask,[bbox_height bbox_width]);
        
        
        %Scale the original control points to the new resized image
        replacement_image_control_points = [double(bbox_width).*x_global./orig_width,double(bbox_height).*y_global./orig_height];
        replacement_image_bbox_end_pts = frame_bbox_end_pts;
  

        %Morph the replacement image to the test image using tps
        morphed_im = morph_tps_wrapper(...
            replacement_image,...
            frame(bbox(bbox_no,2):bbox(bbox_no,2)+bbox_height,bbox(bbox_no,1):bbox(bbox_no,1)+bbox_width,:),...
            [replacement_image_bbox_end_pts;replacement_image_control_points],...
            [frame_bbox_end_pts;control_points],...
            [0.5],[0]);

        %Store the offsets for pasting the source image in the target image
        offsetX = bbox(bbox_no,1);
        offsetY = bbox(bbox_no,2);

        %Paste the source image at the above offsets in the target image and
        %perform Poisson Image Editing
        resultImg = seamlessCloningPoisson(morphed_im,frame,mask,offsetX,offsetY);

 
        %figure;
        %imshow(resultImg);
        %axis image;

end


