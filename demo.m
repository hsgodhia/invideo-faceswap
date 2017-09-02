%{
    Usage: run demo.m
    
    details:

    The ./data folder contains the mask and control points
    saved for the replacement face please uncomment the code below to
    re-create them. We've generated and saved them for easy2, easy3 and
    medium2 videos
    
    note: 'videos/results' already contain results of above 3 test videos

    Specify the video file, the replacement image and the control points 
    Prompt user to select the control points for
    1) 3 for hairline
    2) 5 for face boundary
    3) 2 for eyes
    4) 1 for nose
    5) 1 for mouth
%}
function demo()
    addpath('poissonImageEditing')
    addpath('TPSWarping')
    
    %for easy\easy3.mp4
    %Uncomment to change the control points and replacement mask
    %{
    figure;
    imshow('replacement_image3.png');
    axis image;   
    [x_global,y_global] = ginput(12);
    close all;
    %Prompt the user to select the replacement mask
    %Return a logical mask for the region of the source image to  be cut and
    %pasted in the target image
    mask = maskImage(imread('replacement_image3.png'));
    close all;
    save('data\easy3.mat','x_global','y_global','mask');
    %}
    
    %Face Replacement for easy\easy3.mp4
    load('data\easy3.mat');
    main('videos\easy\easy3.mp4','replacement_image3.png','videos\results\new_easy3.mp4',1,x_global,y_global,mask);
    
    %for easy\easy2.mp4
    %Uncomment to change the control points and replacement mask
    %{
    figure;
    imshow('replacement_image2.jpeg');
    axis image;   
    [x_global,y_global] = ginput(12);
    close all;
    %Prompt the user to select the replacement mask
    %Return a logical mask for the region of the source image to  be cut and
    %pasted in the target image
    mask = maskImage(imread('replacement_image2.jpeg'));
    close all;
    save('data\easy2.mat','x_global','y_global','mask');
    %}
    
    %Face Replacement for easy\easy2.mp4
    %load('data\easy2.mat');
    %main('videos\easy\easy2.mp4','replacement_image2.jpeg','videos\results\new_easy2.mp4',7,x_global,y_global,mask);
    
    %for medium2.mp4
    %Uncomment to change the control points and replacement mask
    %{
    figure;
    imshow('replacement_image2.jpeg');
    axis image;   
    [x_global,y_global] = ginput(12);
    close all;
    %Prompt the user to select the replacement mask
    %Return a logical mask for the region of the source image to  be cut and
    %pasted in the target image
    mask = maskImage(imread('replacement_image2.jpeg'));
    close all;
    save('data\medium2.mat','x_global','y_global','mask');
    %}
    
    %Face Replacement for medium2.mp4
    %load('data\medium2.mat');
    %main('videos\medium\medium2.mp4','replacement_image2.jpeg','videos\results\new_medium2.mp4',1,x_global,y_global,mask);

end