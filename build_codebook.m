%{
Use this file to build a codebook using HOG features for :
1) 3 for hairline
2) 5 for face boundary
3) 2 for eyes
4) 1 for nose
5) 1 for mouth
%}
function build_codebook()
    array_filenames = {'trainingImages/1.jpg';...
                       'trainingImages/3.jpg';...
                       'trainingImages/4.jpg';...
                       'trainingImages/5.jpg';...
                       'trainingImages/6.jpeg';...
                       'trainingImages/7.jpg';...
                       'trainingImages/8.jpg';...
                       'trainingImages/9.jpg';...
                       'trainingImages/10.jpg';...
                       'trainingImages/11.jpeg';...
                       'trainingImages/12.jpeg';...
                       'trainingImages/13.jpeg';...
                       'trainingImages/14.jpg';...
                       'trainingImages/16.jpg';...
    };
    codebook = [];
    x_coord = [];
    y_coord = [];
    bounding_box = [];
    no_files = size(array_filenames,1);
    
    %For each training image
    for i=1:no_files
        I = imread(array_filenames{i});
        
        %Detect the bounding box for the face
        faceDetector = vision.CascadeObjectDetector();
        bbox = step(faceDetector,I);
        
        figure;
        imshow(I);
        axis image;
        
        %Select the control points in the order mentioned above
        %Extract the HOG features for each point and store them
        [x,y] = ginput(12);
        [featureVector,validPoints,hogVisualization] = extractHOGFeatures(I,[x y],'CellSize',[24 24]);
        codebook = [codebook; featureVector];
        %disp(size(codebook));
        x_coord = [x_coord;x];
        y_coord = [y_coord;y];
        bounding_box = [bounding_box;repmat(bbox(1,:),[12 1])];
        
        hold on;
        plot(hogVisualization,'Color','green');
        hold off;
    end
    %Uncomment to save the vectors into the codebook
    %save('codebook.mat','codebook','x_coord','y_coord','bounding_box');
end