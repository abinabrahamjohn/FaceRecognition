function P = HOG_SVM(I)
    %load classifier_hog_svm;
    load classifier_hog_svm_gray;
    %load classifier_hog_svm_v1;
    faceDetector = vision.CascadeObjectDetector('MinSize',[50 50]);
    %I=imrotate(I,-90); %determine if need to be rotated or not
    %preprocessing
    %I=imsharpen(I);
    bboxes = faceDetector(I);
    sizeBbox=size(bboxes);
    numberDetectedFaces=sizeBbox(1);
    detectedFacePositions=zeros(numberDetectedFaces,4); % matrix to store image positions
    predictedLabels=zeros(numberDetectedFaces,1);
    detectedFaceCentroid=zeros(numberDetectedFaces,2);
    P=zeros(numberDetectedFaces,3);
    if numberDetectedFaces==0
        fprintf("Face not detected");
    else
        for i=1:numberDetectedFaces
            x= bboxes(i,1);
            y= bboxes(i,2);
            w=bboxes(i,3);
            h=bboxes(i,4);
            detectedFacePositions(i,:,:,:,:)=[x y w h];
            detectedFaceCentroid(i,1)=round(x+w/2);
            detectedFaceCentroid(i,2)=round(y+h/2);
            detectedFaceImage = imcrop(I,detectedFacePositions(i,:,:,:,:));
            detectedFaceImageGray=histeq(rgb2gray(detectedFaceImage));
            detectedFaceImageResized=imresize(detectedFaceImageGray,[80 80]);
            %detectedFaceImageResized=imresize(detectedFaceImage,[80 80]);
            
            %Save temp file
            baseFileName = sprintf('%d_processed.png', i);
            fullFileName = fullfile('..\Temp\', baseFileName);
            imwrite(detectedFaceImageResized, fullFileName);
            
            %Hog Extraction and SVM prediction
            queryFeatures = extractHOGFeatures(detectedFaceImageResized);
            %[personLabel,PostProbs]  = predict(classifier_hog_svm,queryFeatures);
            [personLabel,PostProbs]  = predict(classifier_hog_svm_gray,queryFeatures);

            %[max_val max_ind]=max(personLabel)
            % personLabel=Actual_Labels(max_ind)
             predictedLabels(i)=double(string(personLabel));            
        end
    end
    IFaces = insertObjectAnnotation(I,'rectangle',detectedFacePositions,predictedLabels,...
    'TextBoxOpacity',0.9,'FontSize',18);
    figure;imshow(IFaces)   
    P=[predictedLabels(:) detectedFaceCentroid(:,1) detectedFaceCentroid(:,2)];
end