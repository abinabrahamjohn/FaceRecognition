D = 'C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Class Extracted Images\';
S = dir(fullfile(D,'*.png')); % pattern to match filenames.
    for k = 1:numel(S)
        F = fullfile(D,S(k).name);
        I = imread(F);
        %I=histeq(I);
        I=adapthisteq(I,'ClipLimit',0.0005);
        I=rgb2gray(I);

        path='C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Class Extracted Images Edit2\';
        if ~exist(path,'dir')
           mkdir(path);
        end

        %Histogram equalized image
        img=I;
        img=imresize(img,[80 80]);
        baseFileName = sprintf('%d_histeq1.png', k);
        fullFileName = fullfile('C:\Users\aaj21\OneDrive - City, University of London\Computer Vision-LAPTOP-O8A4BGK3\Coursework\Processed Data\Class Extracted Images Edit2\', baseFileName);
        imwrite(img, fullFileName);          
    end 
