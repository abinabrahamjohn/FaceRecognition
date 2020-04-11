I= imread('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Code\Images\IMG_2627.jpg');
%faceDetector = vision.CascadeObjectDetector;
I=imrotate(I,-90);
eyeDetector = vision.CascadeObjectDetector('EyePairBig');
bboxes = eyeDetector(I);
IEyes = insertObjectAnnotation(I,'rectangle',bboxes,'Eyes');   
figure
imshow(IEyes)
title('Detected eyes');
sizeBbox=size(bboxes);
numberDetectedFaces=sizeBbox(1);
position=zeros(length(bboxes),4);
label=strings(length(bboxes),1);
for i=1:numberDetectedFaces
    x= bboxes(i,1);
    y= bboxes(i,2);
    w=bboxes(i,3);
    h=bboxes(i,4);
    img_cut = imcrop(I,[x y w h]);
    img_cut_grey=img_cut;
    img_mask=ones(size(img_cut_grey));
    img_mask(:,1:1+(2*w/5),:)=0;
    img_mask=flip(img_mask,2);
    img_mask(:,1:1+(2*w/5),:)=0;
    img_mask(1:5,:,:)=0;
    img_mask=uint8(img_mask);
    I(y:y+h,x:x+w,:)= img_mask .*I(y:y+h,x:x+w,:);
    
end
imshow(I)