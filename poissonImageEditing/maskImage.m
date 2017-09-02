function mask = maskImage(Img)

%Uncomment to use new source image region for pasting into target image

    %Display the source image
    figure;
    imagesc(Img);
    axis image;
    
    %Call imfreehand to draw the freehand region to cut and paste onto the
    %target image
    h = imfreehand();
    
    %Generate a binary mask for the freehand region
    mask = createMask(h);

    %Save the mask into maskImageVar.mat file
    save('maskImageVar.mat','mask');
    
    %Close all figures
    %close all;

    %Load the saved mask
    load('maskImageVar');
    
end

