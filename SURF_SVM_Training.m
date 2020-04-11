%Extract bag and train SVM classifier
categoryClassifier  = trainImageCategoryClassifier(imdsTrain,bag);
%Evaluate SVM classifer on test dataset
evaluate(categoryClassifier, imdsValidation);
save categoryClassifier
