clc; clear all;
load MLP_SURF_v1;
load bag;

for i=1:length(imdsValidation.Labels)
    testingLabels(i)=imdsValidation.Labels(i);
    img = readimage(imdsValidation,i);
    featureVector_Test(i,:) = encode(bag, img);
    fprintf('\nImage %g SURF Extracted\n)', i);
end
x=MLP_SURF_v1(featureVector_Test');
[max_val max_ind]=max(personLabel);
personLabel=Labels(max_ind);
predictedLabels(i)=double(string(personLabel));   
