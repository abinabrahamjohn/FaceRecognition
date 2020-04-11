function P = SURF_SVM(I)
    load '.\Models\categoryClassifier';
    %effect=1;
    faceDetector = vision.CascadeObjectDetector('MinSize',[50 50],'MergeThreshold',8);
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
            
            %Preprocessing
            %detectedFaceImageGray=histeq(rgb2gray(detectedFaceImage));
            detectedFaceImageGray=rgb2gray(histeq(detectedFaceImage));
            detectedFaceImageResized=imresize(detectedFaceImageGray,[80 80]);
            %detectedFaceImageResized=imresize(detectedFaceImage,[80 80]);
            
            %Save temp file
            baseFileName = sprintf('%d_processed.png', i);
            fullFileName = fullfile('..\Temp\', baseFileName);
            imwrite(detectedFaceImageResized, fullFileName);
            
            %SURF Extraction and SVM prediction
            [labelIdx, ~] = predict(categoryClassifier, detectedFaceImageResized);
            % Display the string label
            personLabel=categoryClassifier.Labels(labelIdx);
            predictedLabels(i)=double(string(personLabel));   
            
        end
    end
    IFaces = insertObjectAnnotation(I,'rectangle',detectedFacePositions,predictedLabels,...
    'TextBoxOpacity',0.9,'FontSize',18);
    figure;imshow(IFaces)   
    P=[predictedLabels(:) detectedFaceCentroid(:,1) detectedFaceCentroid(:,2)];
end