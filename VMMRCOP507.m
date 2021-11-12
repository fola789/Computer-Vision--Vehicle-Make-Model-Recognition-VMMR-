carDatabasePath = fullfile('Car Database');%% builds a full file specification from the specified folder and file names
carImageDatabase = imageDatastore(carDatabasePath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames'); %% Load the image collection using an imageDatastore

%%
tbl = countEachLabel(carImageDatabase);%% inspect the  car database to see number of images per category as well as category labels
%% 27 x 2 table created

%%
figure
montage(carImageDatabase.Files(1:64:end)) %% display some of the car images for visual aid of whats in the dataset.
[trainingSet, validationSet] = splitEachLabel(carImageDatabase, 0.8, 'randomize'); %% randomly assigns the car database dataset into a trainingSet and ValidationSet. with the standard ratio of 80/20 (Pareto Principle) from each label to the new datastores. Randomising the split to reduce bias.

%%
%%  create visual bag of features from the training data. 

bag = bagOfFeatures(trainingSet, 'VocabularySize', 4500, 'StrongestFeatures', 1, 'PointSelection', 'Detector','Upright',true); %% returns a bag of features object By default, the visual vocabulary is created from SURF features extracted from the images training data
%% Number of visual words set as 5000, Fraction of 1 for strongest features to use from each label, Surf Detector point selection method for picking point locations rather than the grid default, Upright orientation of SURF feature vector, needed due to CustomExtractor function not being called.  
%%
img = readimage(carImageDatabase, 1);
featureVector = encode(bag, img);
%% The Histogram is a reduced representation of the image is used to train a classifier as well as the classifiaction of the image.

figure %% Plot the histogram of visual word occurrences
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

vmmr_classifier = trainImageCategoryClassifier(trainingSet, bag);%% Train the image classifier by feeding the encoded training images from each category

confMatrix = evaluate(vmmr_classifier, trainingSet);%% Evaluate the trained categoryclassifier by testing against the training set and using a confustion matrix

confMatrix = evaluate(vmmr_classifier, validationSet)%% Evaluate the trained categoryclassifier by testing against the validation set and using a confustion matrix
%%

mean(diag(confMatrix)) %% Calculate the average accuracy. 
%%
trueP = diag(confMatrix)%%Diag produces the true positive
%%
falseP = []; %% Create array for false positives
falseN = [];%% Create array for false negatives
for i = 1:length(trueP)%% for loop to append results
    tP = confMatrix(i);
    fP = sum(confMatrix(:,i),1) - tP;%% false positive is an outcome where the model incorrectly predicts the positive class
    fN = sum(confMatrix(i,:),2) - tP;%% false negative is an outcome where the model incorrectly predicts the negative class.
    
    falseP = [falseP;fP];
    falseN = [falseN;fN];
end %% note true negative isnt needed
%%
precision = trueP ./ (trueP + falseP)%%Calculate the precision Scores
recall = trueP ./ (trueP + falseN)%%Calculate the recall Scores
f1Scores =  (2*(precision.*recall))./(precision+recall)%%Calculate the f1 Scores
meanF1 =  mean(f1Scores)%%Calculate the mean f1 Scores
%%