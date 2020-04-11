clc; clear all;
imds = imageDatastore('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Augmented Imagesv2\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

for i=1:length(imdsTrain.Labels)
    trainingLabels(i)=imdsTrain.Labels(i);
    img = readimage(imdsTrain,i);
    img=imresize(img,[80,80]);
    trainingFeatures(i,:)= extractHOGFeatures(img);
    fprintf('\nImage %g HOG Extracted\n)', i);
end
classifier_hog_svm_v1 = fitcecoc(trainingFeatures, trainingLabels);
save classifier_hog_svm_v1
%{

for i=1:length(imdsTrain.Labels)
    trainingLabels(i)=imdsTrain.Labels(i);
    img = readimage(imdsTrain,i);
    img=imresize(img,[80,80]);
    img_gray=histeq(rgb2gray(img));
    trainingFeatures_gray(i,:)= extractHOGFeatures(img_gray);
    fprintf('\nImage %g HOG Extracted\n)', i);
end
classifier_hog_svm_gray = fitcecoc(trainingFeatures_gray, trainingLabels);
%}
%save classifier_hog_svm_gray