I= imread('.\Images\test_face.jpg');
%load maskedImage;
smiley=imread('.\Images\smiley.jpg');
smiley_scaled=imresize(smiley,0.1);
%[centers,radii] = imfindcircles(smiley_scaled,[40 60],'Sensitivity',0.9)
%h = viscircles(centers,radii);
%imshow(h)
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
    effect=imresize(smiley_scaled,[length(img_cut) length(img_cut)]); 
    %effect=imresize(maskedImage,[length(img_cut) length(img_cut)]) 
    I(y:y+h,x:x+w,:)=effect;
end
imshow(I)