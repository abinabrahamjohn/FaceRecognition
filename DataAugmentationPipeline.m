%Source- https://uk.mathworks.com/help/deeplearning/examples/image-augmentation-using-image-processing-toolbox.html

for photo=1:78
    D = 'C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Combined Imagesv2\'+string(photo)+'\';
    S = dir(fullfile(D,'*.png')); % pattern to match filenames.
    if numel(S)~=0
        for k = 1:2:numel(S)
            F = fullfile(D,S(k).name);
            I = imread(F);
            I=histeq(I);
            %I=adapthisteq(I,'ClipLimit',0.0005);
            I=rgb2gray(I);
            
            path='C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo);
            if ~exist(path,'dir')
               mkdir(path);
            end
            
            %Histogram equalized image
            img=I;
            img=imresize(img,[80 80]);
            baseFileName = sprintf('%d_histeq1.png', k);
            fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            imwrite(img, fullFileName);
            
            %Histogram equalized image small images
            %img=imresize(I,[60 60]);
            %img=imresize(img,[80 80]);
            %baseFileName = sprintf('%d_histeq2.png', k);
            %fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            %imwrite(img, fullFileName);
                        
            %Augmentation rotate 10 degrees and -10 degrees
            
            img=imrotate(I,-10,'bilinear','crop');
            img=imresize(img,[80 80]);
            baseFileName = sprintf('%d_rotate-10.png', k);
            fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            imwrite(img, fullFileName);
            
            img=imrotate(I,10,'bilinear','crop');
            img=imresize(img,[80 80]);
            baseFileName = sprintf('%d_rotate+10.png', k);
            fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            imwrite(img, fullFileName);
            
            %flip horizontal
            img=flip(I,2);
            img=imresize(img,[80 80]);
            baseFileName = sprintf('%d_flip.png', k);
            fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            imwrite(img, fullFileName);
            
            %blur
            sigma = 0.1+5*rand;
            img = imgaussfilt(I,sigma); 
            img=imresize(img,[80 80]);
            baseFileName = sprintf('%d_blur.png', k);
            fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            imwrite(img, fullFileName);
            
            %noise
            %img=imnoise(I,'gaussian');
            %img=imresize(img,[80 80]);
            %baseFileName = sprintf('%d_noise.png', k);
            %fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            %imwrite(img, fullFileName);
            
            %darken image
            %img=histeq(I);
            %img=imresize(img,[80 80]);
            %baseFileName = sprintf('%d_n#oise.png', k);
            %fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Augmented Imagesv4\'+string(photo)+'\', baseFileName);
            %imwrite(img, fullFileName);
            
        end 
    end
end