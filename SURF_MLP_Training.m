%Extract encode method for counting the visual word occurrences in an image. 
%It produced a histogram that becomes a new and reduced representation of an image. 
%and train MLP

for i=1:length(imdsTrain.Labels)
    trainingLabels(i)=imdsTrain.Labels(i);
    img = readimage(imdsTrain,i);
    featureVector(i,:) = encode(bag, img);
    fprintf('\nImage %g SURF Extracted\n)', i);
end
hiddenLayerSize = [100];
%MLP_SURF = feedforwardnet(hiddenLayerSize,'trainscg');
MLP_SURF = patternnet(hiddenLayerSize,'trainscg');
MLP_SURF.divideParam.trainRatio = 0.90;
MLP_SURF.divideParam.valRatio   = 0.10;
MLP_SURF.divideParam.testRatio  = 0;
%MLP_SURF = configure(MLP_SURF,featureVector',dummyvar(trainingLabels')');
[MLP_SURF_v1,tr] = train(MLP_SURF, featureVector', dummyvar(trainingLabels')');
save MLP_SURF_v1;
