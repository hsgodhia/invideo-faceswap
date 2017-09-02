function resultImg = seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY)

    %Retrieve the height and width of the target image
    [targetH,targetW,ndim] = size(targetImg);
    
    %Get the indices of the target image that need to be replaced
    indexes = getIndexes(mask,targetH,targetW,offsetX,offsetY);
    
    %Get the coefficient matrix
    coefM = getCoefMatrix(indexes);
    

    %Get solution vector for the red channel
    solVector_red = getSolutionVect(indexes,sourceImg(:,:,1),targetImg(:,:,1),offsetX,offsetY);
    %Solve for x for the red channel
    x_red = mldivide(double(coefM),double(solVector_red));
    
    %Get solution vector for the green channel
    solVector_green = getSolutionVect(indexes,sourceImg(:,:,2),targetImg(:,:,2),offsetX,offsetY);
    %Solve for x for the red channel
    x_green = mldivide(double(coefM),double(solVector_green));
    
    %Get solution vector for the blue channel
    solVector_blue = getSolutionVect(indexes,sourceImg(:,:,3),targetImg(:,:,3),offsetX,offsetY);
    %Solve for x for the blue channel
    x_blue = mldivide(double(coefM),double(solVector_blue));

    %Reconstruct the new image
    resultImg = reconstructImg(indexes,x_red,x_green,x_blue,targetImg);

end