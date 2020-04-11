%I= imread('test_face.jpg');
I= imread('IMG_2627.jpg');
figure;imshow(I)
%I=imrotate(I,-90);
%figure;imshow(I)
faceDetector = vision.CascadeObjectDetector;
bboxes = faceDetector(I);
IFaces = insertObjectAnnotation(I,'rectangle',bboxes,'Face'); 
imshow(IFaces)
sizes=size(bboxes);
if sizes(1)==0
    I=imrotate(I,-90);
    imshow(I)
    bboxes = faceDetector(I);
    sizes=size(bboxes);
end
IFaces = insertObjectAnnotation(I,'rectangle',bboxes,'Face');   
figure
imshow(IFaces)
title('Detected faces');

if sizes(1)==1
    x= bboxes(1);
    y= bboxes(2);
    w=bboxes(3);
    h=bboxes(4);
    img_cut = imcrop(I,[x y w h]);
    %imgd = im2double(img_cut);
    [L,C] = imsegkmeans(img_cut,3);
    cartoon = label2rgb(L,im2double(C));
    I(y:y+h,x:x+w,:)=cartoon;
else
    for i=1:length(bboxes)
        x= bboxes(i,1);
        y= bboxes(i,2);
        w=bboxes(i,3);
        h=bboxes(i,4);
        img_cut = imcrop(I,[x y w h]);
        %imgd = im2double(img_cut);
        [L,C] = imsegkmeans(img_cut,3);
        cartoon = label2rgb(L,im2double(C));
        I(y:y+h,x:x+w,:)=cartoon;
    end
end
imshow(I)