I= imread('test_face.jpg');
faceDetector = vision.CascadeObjectDetector;
bboxes = faceDetector(I);
IFaces = insertObjectAnnotation(I,'rectangle',bboxes,'Face');   
figure
imshow(IFaces)
title('Detected faces');
position=zeros(length(bboxes),4);
label=strings(length(bboxes),1);
for i=1:length(bboxes)
    x= bboxes(i,1);
    y= bboxes(i,2);
    w=bboxes(i,3);
    h=bboxes(i,4);
    img_cut = imcrop(I,[x y w h]);
    %imgd = im2double(img_cut);
    H = fspecial('disk',40);
    blurred = imfilter(img_cut,H,'replicate'); 
    I(y:y+h,x:x+w,:)=blurred;
end
imshow(I)