clc;clear all;close all;
imds = imageDatastore('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomize');
bag= bagOfFeatures(imdsTrain);
save bag;
