%load AbiNet
load cnn_tl_alexnet_v1;
faceDetector = vision.CascadeObjectDetector;
I = imread('test_group1.jpg');
%figure;imshow(I);
%I=imsharpen(I);
%I=imlocalbrighten(I);
%I=histeq(I);
%I=localcontrast(I);
figure;imshow(I);
bboxes = faceDetector(I);
IFaces = insertObjectAnnotation(I,'rectangle',bboxes,'Face');   
%figure
%imshow(IFaces)
%hold on;
%title('Detected faces');
position=zeros(length(bboxes),4);
label=strings(length(bboxes),1);
for i=1:length(bboxes)
    x= bboxes(i,1);
    y= bboxes(i,2);
    w=bboxes(i,3);
    h=bboxes(i,4);
    img_cut = imcrop(I,[x y w h]);
    img_cut=histeq(img_cut);
    imshow(img_cut);
    position(i,:)=[x y w h];
    img_cut=imresize(img_cut,[227 227]);
    %[YPred,probs] = classify(AbiNet,img_cut);
    [YPred,probs] = classify(cnn_tl_alexnet_v1,img_cut);
    label(i)=YPred;
    baseFileName = sprintf('%s.png', YPred);
    fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Test\Identified\', baseFileName);
end

IFaces = insertObjectAnnotation(I,'rectangle',position,label,...
    'TextBoxOpacity',0.9,'FontSize',18);
imshow(IFaces)