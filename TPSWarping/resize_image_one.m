function im1 = resize_image_one(im1,im2)
 
    %Fetch resolutions of both the images
    [initial_height,initial_width,initial_channels] = size(im1);
    [target_height,target_width,target_channels] = size(im2);

    %Resize image 1 to size of image 2 if they are not equal
    if(not(initial_height==target_height && initial_width==target_width))
        height_factor = 1;
        width_factor = 1;
        if(initial_height>target_height)
            height_factor = initial_height/double(target_height);
        end
        if(initial_width>target_width)
            width_factor = initial_width/double(target_width);
        end
        if(height_factor>width_factor)
            im1 = imresize(im1,[initial_height/height_factor initial_width/height_factor]);
        else
            im1 = imresize(im1,[initial_height/width_factor initial_width/width_factor ]);
        end    
        
        %If image 1 size is lesser than image 2, pad with zeros
        [new_height,new_width,new_channels] = size(im1);
        no_rows = 0;
        if(new_height<target_height)
            no_rows = target_height-new_height;
            im1 = [zeros(floor(no_rows/2),new_width,new_channels);im1];
            im1 = [im1;zeros(no_rows - floor(no_rows/2),new_width,new_channels)];
        end
        if(new_width<target_width)
            no_cols = target_width-new_width;
            im1 = [zeros(new_height+no_rows,floor(no_cols/2),new_channels) im1];
            im1 = [im1 zeros(new_height+no_rows,no_cols - floor(no_cols/2),new_channels)];
        end
    end
 end