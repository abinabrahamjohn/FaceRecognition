clc; clear all;
imds = imageDatastore('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\CombinedNAgumented Images\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8);

for i=1:length(imdsTrain.Labels)
    trainingLabels(i)=imdsTrain.Labels(i);
    img = readimage(imdsTrain,i);
    img=imresize(img,[80,80]);
    trainingFeatures(i,:)= extractHOGFeatures(img);
    fprintf('\nImage %g HOG Extracted\n)', i);
end
hiddenLayerSize = [100];
MLP_HOG = feedforwardnet(hiddenLayerSize,'trainscg');
%MLP_HOG = configure(MLP_HOG,trainingFeatures',double(trainingLabels));
MLP_HOG = configure(MLP_HOG,trainingFeatures',dummyvar(trainingLabels')');
[MLP_HOG,tr] = train(MLP_HOG, trainingFeatures', dummyvar(trainingLabels')');
save MLP_HOG;

for i=1:length(imdsTrain.Labels)
    trainingLabels(i)=imdsTrain.Labels(i);
    img = readimage(imdsTrain,i);
    img=imresize(img,[80,80]);
    img_gray=histeq(rgb2gray(img))
    trainingFeatures_gray(i,:)= extractHOGFeatures(img_gray);
    fprintf('\nImage %g HOG Extracted\n)', i);
end
hiddenLayerSize = [100];
MLP_HOG_Gray = feedforwardnet(hiddenLayerSize,'trainscg');
%MLP_HOG = configure(MLP_HOG,trainingFeatures',double(trainingLabels));
MLP_HOG_Gray = configure(MLP_HOG_Gray,trainingFeatures_gray',dummyvar(trainingLabels')');
[MLP_HOG_Gray,tr] = train(MLP_HOG_Gray, trainingFeatures_gray', dummyvar(trainingLabels')');
save MLP_HOG_Gray;