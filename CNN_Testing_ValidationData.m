load cnn_tl_alexnet_v1;
[YPred,probs] = classify(cnn_tl_alexnet_v1,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
idx = randperm(numel(imdsValidation.Files),10);
figure
for i = 1:10
    subplot(2,5,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end