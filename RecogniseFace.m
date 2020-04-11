function P = RecogniseFace(I, featureType, classifierType, creativeMode)
%   I: Image
%   featureType : HOG, SURF
%   classifierType : SVM, MLP, CNN
%   creativeMode : 0,1
    I = ReOrientFace(I);
    if (strcmp(featureType,'HOG')==1 && strcmp(classifierType,'SVM')==1 && strcmp(creativeMode,'0')==1)
        P=HOG_SVM(I);
    elseif (strcmp(featureType,'HOG')==1 && strcmp(classifierType,'MLP')==1 && strcmp(creativeMode,'0')==1)
        P=HOG_MLP(I);
    elseif (strcmp(featureType,'SURF')==1 && strcmp(classifierType,'SVM')==1 && strcmp(creativeMode,'0')==1)
        P=SURF_SVM(I);
    elseif (strcmp(featureType,'SURF')==1 && strcmp(classifierType,'MLP')==1 && strcmp(creativeMode,'0')==1)
        P=SURF_MLP(I);
    elseif (strcmp(featureType,'NONE')==1 && strcmp(classifierType,'CNN')==1 && strcmp(creativeMode,'0')==1)
        P=CNN(I);
    else
        fprintf('\n**Invalid Choice of input parameter**\n')
        fprintf('RecogniseFace(I, featureType, classifierType, creativeMode) expects...\nI: Image as Matrix\nfeatureType : HOG, SURF')
        fprintf('\nclassifierType : SVM, MLP, CNN\ncreativeMode : 0,1\n');
    end
end