close all   % closes all open figure windows
clear       % clears all variables from workspace
clc         % clears the command window

%% 1. LOAD DATA
data = readtable('Dry_Bean_Dataset.xlsx');   

% Separate features and labels
X = data{:, 1:end-1};           % All 16 feature columns (numeric)
Y = data{:, end};                % Target column (bean type labels)

% If Y is numeric, convert to categorical
Y = categorical(Y);

fprintf('Dataset loaded: %d samples, %d features, %d classes\n', ...
    size(X,1), size(X,4), numel(categories(Y)));

%% 2. CHECK CLASS DISTRIBUTION
figure;
histogram(Y);
title('Class Distribution — Bean Types');
xlabel('Bean Type');
ylabel('Count');

%% 3. TRAIN / TEST SPLIT (70/30, stratified)
rng(42);    % Set seed for reproducibility

cv = cvpartition(Y, 'HoldOut', 0.30, 'Stratify', true);

X_train = X(cv.training, :);
Y_train = Y(cv.training);
X_test  = X(cv.test, :);
Y_test  = Y(cv.test);

fprintf('Training samples: %d | Test samples: %d\n', ...
    sum(cv.training), sum(cv.test));

%% 4. TRAIN RANDOM FOREST
numTrees = 100;   % Number of trees — can tune this later

forest = TreeBagger(50, X_train, Y_train, ...   % dropped to 50 based on OOB plot
    'Method',                   'classification', ...
    'OOBPrediction',            'on', ...
    'OOBPredictorImportance',   'on', ...
    'MinLeafSize',              2, ...           % already set correctly
    'NumPredictorsToSample',    16);              % try 3, 4, 6, 8 and compare

%% 5. OUT-OF-BAG ERROR PLOT
% OOB error tells you if 100 trees is enough
% (should level off — if still dropping, increase numTrees)
figure;
oobErrorBaggedEnsemble = oobError(forest);
plot(oobErrorBaggedEnsemble, 'LineWidth', 1.5);
title('Out-of-Bag Classification Error vs. Number of Trees');
xlabel('Number of Trees');
ylabel('OOB Classification Error');
grid on;

%% 6. PREDICT ON TEST SET
[Y_pred_cell, scores] = predict(forest, X_test);
Y_pred = categorical(Y_pred_cell);   % Convert cell array back to categorical

%% 7. ACCURACY
accuracy = sum(Y_pred == Y_test) / numel(Y_test) * 100;
fprintf('\nTest Set Accuracy: %.2f%%\n', accuracy);

%% 8. CONFUSION MATRIX
figure;
cm = confusionchart(Y_test, Y_pred, ...
    'Title',                'Random Forest — Bean Classification Confusion Matrix', ...
    'RowSummary',           'row-normalized', ...   % Shows recall per class
    'ColumnSummary',        'column-normalized');   % Shows precision per class

%% 9. PER-CLASS METRICS (Precision, Recall, F1)
classNames = categories(Y_test);
numClasses = numel(classNames);

fprintf('\n%-12s %10s %10s %10s\n', 'Bean Type', 'Precision', 'Recall', 'F1 Score');
fprintf('%s\n', repmat('-', 1, 46));

cmMatrix = confusionmat(Y_test, Y_pred, 'Order', classNames);

for i = 1:numClasses
    TP = cmMatrix(i,i);
    FP = sum(cmMatrix(:,i)) - TP;
    FN = sum(cmMatrix(i,:)) - TP;

    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2 * (precision * recall) / (precision + recall + eps);

    fprintf('%-12s %10.3f %10.3f %10.3f\n', ...
        classNames{i}, precision, recall, f1);
end

%% 10. FEATURE IMPORTANCE
importance = forest.OOBPermutedPredictorDeltaError;
featureNames = data.Properties.VariableNames(1:end-1);

[sortedImp, sortIdx] = sort(importance, 'descend');

figure;
barh(flipud(sortedImp));                          % <-- fixed: was fliplud
yticks(1:numel(featureNames));
yticklabels(flipud(featureNames(sortIdx)));       % <-- fixed: was fliplud
title('Feature Importance (OOB Permutation)');
xlabel('Mean Decrease in Accuracy');
grid on;
