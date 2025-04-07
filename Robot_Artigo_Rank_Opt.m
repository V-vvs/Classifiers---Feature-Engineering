% Classificação de dados de falha de execução de manipuladores robóticos com engenharia de features %%%
%% Vanessa Vieira de Sousa - PPGETI, UFC
%===========================================================================================
% Classificação dos dados com todos os features 
% Carregar os dados do banco de dados de classes agregadas, 'RobotDataset';
dados=RobotDataset; 
% % Dividir os dados em características (features) e rótulos (labels)
X = dados(:, 1:end-1); % Características
x = normalize(X);
Y = dados(:, end); % Rótulos
y = table2array(Y);
% Identificar as classes únicas
classes = unique(Y);
classess=table2cell(classes);
classesm=table2array(classes);



% Calcular a matriz de correlação
dados_cor= table2array(x);
corrMatrix = corr(dados_cor);
cormatrix=array2table(corrMatrix);
%
% Plotar a matriz de correlação
figure;
imagesc(corrMatrix);
colorbar;
nomes_caracteristicas = x.Properties.VariableNames;
%
% Criar o heatmap da matriz de correlação
h = heatmap(corrMatrix);
%
% Ajustar os rótulos dos eixos
h.XDisplayLabels = nomes_caracteristicas;
h.YDisplayLabels = nomes_caracteristicas;

% Definir o esquema de validação cruzada
rng(31) % For reproducibility
% training_ratio = 0.80;
% Generate indices for the training and test sets
% yS = grp2idx(y);  % transformação para estratificação
cv = cvpartition(y, 'KFold', 5,'Stratify',false); % ativar a estratificação
% Split the features and labels into training and test sets
X_train = x(training(cv,1),:);
Y_train = Y(training(cv,1),:);
X_test = x(test(cv,1),:);
Y_test = Y(test(cv,1),:);
%dados

% Definir os classificadores a serem avaliados
classifiers = {'KNN', 'NaiveBayes', 'NeuralNetwork','Ensemble'};

% Otimização de hiperparâmetros com validação cruzada
% Treinar o classificador com os hiperparâmetros otimizados
  value1 = 200; %iterações da otimização
% case 'KNN'  
  modelknn = fitcknn(X_train,Y_train,'NumNeighbors',2,'Distance','cosine','DistanceWeight','squaredinverse'); 
% case 'NaiveBayes'
  modelnb = fitcnb(X_train, Y_train,'DistributionNames','normal');
% % % case 'NeuralNetwork'
  modelnn=fitcnet(X_train,Y_train,'Activations','relu','Lambda',0.0085282,'LayerWeightsInitializer','glorot','LayerBiasesInitializer','ones','LayerSizes',[21,180]);
% % % case 'Ensemble'
  modelens = fitcensemble(X_train,Y_train,'Method','AdaBoostM2','NumLearningCycles',178,'LearnRate',0.95123,'Learner',templateTree('MinLeaf', 4,'MaxNumSplits',32,'SplitCriterion','twoing'));


  % Predições para cada classificador
predictions_knn = predict(modelknn,X_test);
predictions_nb = predict(modelnb,X_test);
predictions_nn = predict(modelnn,X_test);
predictions_ens = predict(modelens,X_test);



Ranqueamento de features realizado no treinamento do Adaboost M2 (método de seleção de atributos Embedded)
Inicialize o vetor de importâncias
numFeatures_ens = size(X_train, 2);
featureImportances = zeros(1, numFeatures_ens);

% Itere sobre os weak learners
learners = modelens.Trained; % Obtém os weak learners
weights_ens = modelens.TrainedWeights; % Obtém os pesos dos weak learners

for i = 1:length(learners)
    tree = learners{i};
    imp = predictorImportance(tree); % Importância dos atributos na árvore
    featureImportances = featureImportances + weights_ens(i) * imp';
end

% Normalize as importâncias
featureImportances = featureImportances / sum(featureImportances);

% Classifique as importâncias
[sortedImportances, featureRanking] = sort(featureImportances, 'descend');

% Exiba os resultados
disp('Feature Ranking:');
disp(featureRanking);
disp('Feature Importances:');
disp(sortedImportances);

caracteristicas = tree.ExpandedPredictorNames;
figure;
bar(sortedImportances);
xticks(1:numFeatures_ens);
xticklabels(caracteristicas(featureRanking)); % Mostra os nomes dos preditores na ordem do ranking
xtickangle(90); % Ajusta a inclinação dos rótulos para melhor visualização
xlabel('Features');
ylabel('Importance Scores');
title('Adaboost M2');



% Inicializar célula para plotar matriz de confusão
Ytest = table2array(Y_test);
    % Calcular métricas de avaliação
    % Para KNN
    confusionMat_knn = confusionmat(Ytest, predictions_knn);
    accuracy_knn = sum(diag(confusionMat_knn)) / sum(confusionMat_knn(:));
    recall_knn = diag(confusionMat_knn) ./ sum(confusionMat_knn, 2);
    precision_knn = diag(confusionMat_knn) ./ sum(confusionMat_knn, 1)';
    f1_score_knn = 2 * (precision_knn .* recall_knn) ./ (precision_knn + recall_knn); % F1 Score
% 
    % Para Naive Bayes
    confusionMat_nb = confusionmat(Ytest, predictions_nb);
    accuracy_nb = sum(diag(confusionMat_nb)) / sum(confusionMat_nb(:));
    recall_nb = diag(confusionMat_nb) ./ sum(confusionMat_nb, 2);
    precision_nb = diag(confusionMat_nb) ./ sum(confusionMat_nb, 1)';
    f1_score_nb = 2 * (precision_nb .* recall_nb) ./ (precision_nb + recall_nb); % F1 Score

%   % Para Neural Network
    confusionMat_nn = confusionmat(Ytest, predictions_nn);
    accuracy_nn = sum(diag(confusionMat_nn)) / sum(confusionMat_nn(:));
    recall_nn = diag(confusionMat_nn) ./ sum(confusionMat_nn, 2);
    precision_nn = diag(confusionMat_nn) ./ sum(confusionMat_nn, 1)';
    f1_score_nn = 2 * (precision_nn .* recall_nn) ./ (precision_nn + recall_nn); % F1 Score
% 
  % Para Ensemble
    confusionMat_ens = confusionmat(Ytest, predictions_ens);
    accuracy_ens = sum(diag(confusionMat_ens)) / sum(confusionMat_ens(:));
    recall_ens = diag(confusionMat_ens) ./ sum(confusionMat_ens, 2);
    precision_ens = diag(confusionMat_ens) ./ sum(confusionMat_ens, 1)';
    f1_score_ens = 2 * (precision_ens .* recall_ens) ./ (precision_ens + recall_ens); % F1 Score

% 
    % Exibir métricas de desempenho para cada classificador
disp(['Accuracy_KNN: ', num2str(accuracy_knn)]);
disp(['Accuracy_Naive Bayes: ', num2str(accuracy_nb)]);
disp(['Accuracy_Neural Network: ', num2str(accuracy_nn)]);
disp(['Accuracy_Ensemble: ', num2str(accuracy_ens)]);
% % disp(['Recall: ', num2str(recall_knn)]);
% % disp(['Precision: ', num2str(precision)]);
% % disp(['F1 Score: ', num2str(f1_score)]);
% % % 
% Plotar as matrizes de confusão
figure;
confusionchart(confusionMat_knn,classes,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - KNN - All Features');
figure;
confusionchart(confusionMat_nb,classes,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - Naive Bayes - All Features');
figure;
confusionchart(confusionMat_nn,classes,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - Neural Network - All Features');
figure;
confusionchart(confusionMat_ens,classes,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - Ensemble - All Features');
% 
figure Name 'Matrix Confusion - All Features'
subplot(2, 2, 1);
confusionchart(confusionMat_knn,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('KNN - All Features');
subplot(2, 2, 2);
confusionchart(confusionMat_nb,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Naive Bayes - All Features');
subplot(2, 2, 3);
confusionchart(confusionMat_nn,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Neural Network - All Features');
subplot(2, 2, 4);
confusionchart(confusionMat_ens,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Ensemble - All Features');


% % Calcular e plotar os ranqueamentos dos preditores com Relieff, Chi2 e MRMR (método de seleção de atributos Filtro)
Xtrain = table2array(X_train);
Ytrain = table2array(Y_train);
figure
% Executar o algoritmo relieff nas características numéricas
[ranking1, weights1] = relieff(Xtrain, Ytrain, 10); % Substitua 'labels' pelos rótulos da sua tabela
nomes_caracteristicas = x.Properties.VariableNames;
ranking1(1:90)
ranking_reliefF=weights1(ranking1);
subplot(3, 1, 1);
bar(weights1(ranking1));
xticklabels(nomes_caracteristicas(ranking1));
xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90])
xtickangle(90);
xlabel('Features');
ylabel('Importance Scores');
title('ReliefF');
% Executar o algoritmo MRMR nas características numéricas
[ranking2, weights2] = fscmrmr(Xtrain,Ytrain); % Substitua 'labels' pelos rótulos da sua tabela
nomes_caracteristicas = x.Properties.VariableNames;
ranking2(1:90)
subplot(3, 1, 2);
bar(weights2(ranking2));
xticklabels(nomes_caracteristicas(ranking2));
xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90])
xtickangle(90);
xlabel('Features');
ylabel('Importance Scores');
title('MRMR');
% Executar o algoritmo Chi-Square nas características numéricas
[ranking3, weights3] = fscchi2(Xtrain,Ytrain); % Substitua 'labels' pelos rótulos da sua tabela
nomes_caracteristicas = x.Properties.VariableNames;
ranking3(1:90)
subplot(3, 1, 3);
bar(weights3(ranking3));
xticklabels(nomes_caracteristicas(ranking3));
xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90])
xtickangle(90);
xlabel('Features');
ylabel('Importance Scores');
title('CHI2');
% % 
Realizar interpretabilidade dos dados com método Shap (condicional)
 Xtrain = table2array(X_train); % Transformação para interpretabilidade
explainer_knn = shapley(modelknn, Xtrain,QueryPoints=Xtrain,Method="conditional"); 
explainer_nb = shapley(modelnb, Xtrain,QueryPoints=Xtrain,Method="conditional");
explainer_nn = shapley(modelnn, Xtrain,QueryPoints=Xtrain,Method="conditional"); 
explainer_ens = shapley(modelens, Xtrain,QueryPoints=Xtrain,Method="conditional"); 
% Plotar gráficos Shap
% Plot visualization of mean(abs(shap)) bar plot, and swarmchart for each output class. Note that these multi-query-point plots require R2024a or higher
figure(1); tiledlayout(2,2); nexttile(1);
% % Plot the mean(abs(shap)) plot for this multi-query-point shapley object
 plot(explainer_knn,"NumImportantPredictors",90);
% Plot the shapley summary swarmchart for each output class
% KNN
for i=2:4
    nexttile(i);
    swarmchart(explainer_knn,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=90)
end
% Naive Bayes
figure(1); tiledlayout(2,2); nexttile(1);  
plot(explainer_nb,"NumImportantPredictors",20); 
for i=2:4
    nexttile(i);
    swarmchart(explainer_nb,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=20)
end
% Neural Network
figure(1); tiledlayout(2,2); nexttile(1);    
plot(explainer_nn,"NumImportantPredictors",90);
for i=2:4
    nexttile(i);
    swarmchart(explainer_nn,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=90)
end
% Ensemble
figure(1); tiledlayout(2,2); nexttile(1);    
plot(explainer_ens,"NumImportantPredictors",20); 
for i=2:4
    nexttile(i);
    swarmchart(explainer_ens,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=20)
end

Realizar interpretabilidade dos dados com método Shap (interventional)
explainer_knn_int = shapley(modelknn, Xtrain,QueryPoints=Xtrain,Method="interventional"); 
explainer_nb_int = shapley(modelnb, Xtrain,QueryPoints=Xtrain,Method="interventional");
explainer_nn_int = shapley(modelnn, Xtrain,QueryPoints=Xtrain,Method="interventional"); 
explainer_ens_int = shapley(modelens, Xtrain,QueryPoints=Xtrain,Method="interventional"); 
% Plotar gráficos Shap
% Plot visualization of mean(abs(shap)) bar plot, and swarmchart for each output class. Note that these multi-query-point plots require R2024a or higher
figure(1); tiledlayout(2,2); nexttile(1);
% Plot the mean(abs(shap)) plot for this multi-query-point shapley object
 plot(explainer_knn_int,"NumImportantPredictors",20);
% Plot the shapley summary swarmchart for each output class
% KNN
for i=2:4
    nexttile(i);
    swarmchart(explainer_knn_int,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=20)
end
% Naive Bayes
figure(1); tiledlayout(2,2); nexttile(1);  
plot(explainer_nb_int,"NumImportantPredictors",90); 
for i=2:4
    nexttile(i);
    swarmchart(explainer_nb_int,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=90)
end
% Neural Network
figure(1); tiledlayout(2,2); nexttile(1);    
plot(explainer_nn_int,"NumImportantPredictors",20);
for i=2:4
    nexttile(i);
    swarmchart(explainer_nn_int,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=20)
end
% Ensemble
figure(1); tiledlayout(2,2); nexttile(1);    
plot(explainer_ens_int,"NumImportantPredictors",90); 
for i=2:4
    nexttile(i);
    swarmchart(explainer_ens_int,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=90)
end






% Nova classificação dos dados com os features selecionados por filtros para limiares diferentes

% RefliefF
SF_train_relieff_10 = X_train(:, ranking1(1:10)); % dados de treino com melhores features
SF_train_relieff_20 = X_train(:, ranking1(1:20)); % dados de treino com melhores features
SF_train_relieff_30 = X_train(:, ranking1(1:30)); % dados de treino com melhores features
SF_train_relieff_40 = X_train(:, ranking1(1:40)); % dados de treino com melhores features
SF_train_relieff_50 = X_train(:, ranking1(1:50)); % dados de treino com melhores features
SF_train_relieff_60 = X_train(:, ranking1(1:60)); % dados de treino com melhores features
SF_train_relieff_70 = X_train(:, ranking1(1:70)); % dados de treino com melhores features
SF_train_relieff_80 = X_train(:, ranking1(1:80)); % dados de treino com melhores features
SF_train_relieff_90 = X_train(:, ranking1(1:90)); % dados de treino com melhores features
SF_test_relieff_10 = X_test(:, ranking1(1:10)); % dados de teste com melhores features
SF_test_relieff_20 = X_test(:, ranking1(1:20)); % dados de teste com melhores features
SF_test_relieff_30 = X_test(:, ranking1(1:30)); % dados de teste com melhores features
SF_test_relieff_40 = X_test(:, ranking1(1:40)); % dados de teste com melhores features
SF_test_relieff_50 = X_test(:, ranking1(1:50)); % dados de teste com melhores features
SF_test_relieff_60 = X_test(:, ranking1(1:60)); % dados de teste com melhores features
SF_test_relieff_70 = X_test(:, ranking1(1:70)); % dados de teste com melhores features
SF_test_relieff_80 = X_test(:, ranking1(1:80)); % dados de teste com melhores features
SF_test_relieff_90 = X_test(:, ranking1(1:90)); % dados de teste com melhores features

% Treinar o modelo novamente usando apenas os recursos selecionados
  modelknn_relieff_10 = fitcknn(SF_train_relieff_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_20 = fitcknn(SF_train_relieff_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_30 = fitcknn(SF_train_relieff_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_40 = fitcknn(SF_train_relieff_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_50 = fitcknn(SF_train_relieff_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_60 = fitcknn(SF_train_relieff_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_70 = fitcknn(SF_train_relieff_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_80 = fitcknn(SF_train_relieff_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_relieff_90 = fitcknn(SF_train_relieff_90,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnb_relieff_10 = fitcnb(SF_train_relieff_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_20 = fitcnb(SF_train_relieff_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_30 = fitcnb(SF_train_relieff_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_40 = fitcnb(SF_train_relieff_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_50 = fitcnb(SF_train_relieff_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_60 = fitcnb(SF_train_relieff_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_70 = fitcnb(SF_train_relieff_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_80 = fitcnb(SF_train_relieff_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_relieff_90 = fitcnb(SF_train_relieff_90,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnn_relieff_10 = fitcnet(SF_train_relieff_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_relieff_20 = fitcnet(SF_train_relieff_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_relieff_30 = fitcnet(SF_train_relieff_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_relieff_40 = fitcnet(SF_train_relieff_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_relieff_50 = fitcnet(SF_train_relieff_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_relieff_60 = fitcnet(SF_train_relieff_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_relieff_70 = fitcnet(SF_train_relieff_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_relieff_80 = fitcnet(SF_train_relieff_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelens_relieff_10 = fitcensemble(SF_train_relieff_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_20 = fitcensemble(SF_train_relieff_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_30 = fitcensemble(SF_train_relieff_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_40 = fitcensemble(SF_train_relieff_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_50 = fitcensemble(SF_train_relieff_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_60 = fitcensemble(SF_train_relieff_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_70 = fitcensemble(SF_train_relieff_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_80 = fitcensemble(SF_train_relieff_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_relieff_90 = fitcensemble(SF_train_relieff_90,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
% 
% % Testar o modelo novamente usando apenas os recursos selecionados
predictions_knn_relieff_10=predict(modelknn_relieff_10,SF_test_relieff_10);
predictions_knn_relieff_20=predict(modelknn_relieff_20,SF_test_relieff_20);
predictions_knn_relieff_30=predict(modelknn_relieff_30,SF_test_relieff_30);
predictions_knn_relieff_40=predict(modelknn_relieff_40,SF_test_relieff_40);
predictions_knn_relieff_50=predict(modelknn_relieff_50,SF_test_relieff_50);
predictions_knn_relieff_60=predict(modelknn_relieff_60,SF_test_relieff_60);
predictions_knn_relieff_70=predict(modelknn_relieff_70,SF_test_relieff_70);
predictions_knn_relieff_80=predict(modelknn_relieff_80,SF_test_relieff_80);
predictions_knn_relieff_90=predict(modelknn_relieff_90,SF_test_relieff_90);

predictions_nb_relieff_10=predict(modelnb_relieff_10,SF_test_relieff_10);
predictions_nb_relieff_20=predict(modelnb_relieff_20,SF_test_relieff_20);
predictions_nb_relieff_30=predict(modelnb_relieff_30,SF_test_relieff_30);
predictions_nb_relieff_40=predict(modelnb_relieff_40,SF_test_relieff_40);
predictions_nb_relieff_50=predict(modelnb_relieff_50,SF_test_relieff_50);
predictions_nb_relieff_60=predict(modelnb_relieff_60,SF_test_relieff_60);
predictions_nb_relieff_70=predict(modelnb_relieff_70,SF_test_relieff_70);
predictions_nb_relieff_80=predict(modelnb_relieff_80,SF_test_relieff_80);
predictions_nb_relieff_90=predict(modelnb_relieff_90,SF_test_relieff_90);

predictions_nn_relieff_10=predict(modelnn_relieff_10,SF_test_relieff_10);
predictions_nn_relieff_20=predict(modelnn_relieff_20,SF_test_relieff_20);
predictions_nn_relieff_30=predict(modelnn_relieff_30,SF_test_relieff_30);
predictions_nn_relieff_40=predict(modelnn_relieff_40,SF_test_relieff_40);
predictions_nn_relieff_50=predict(modelnn_relieff_50,SF_test_relieff_50);
predictions_nn_relieff_60=predict(modelnn_relieff_60,SF_test_relieff_60);
predictions_nn_relieff_70=predict(modelnn_relieff_70,SF_test_relieff_70);
predictions_nn_relieff_80=predict(modelnn_relieff_80,SF_test_relieff_80);

predictions_ens_relieff_10=predict(modelens_relieff_10,SF_test_relieff_10);
predictions_ens_relieff_20=predict(modelens_relieff_20,SF_test_relieff_20);
predictions_ens_relieff_30=predict(modelens_relieff_30,SF_test_relieff_30);
predictions_ens_relieff_40=predict(modelens_relieff_40,SF_test_relieff_40);
predictions_ens_relieff_50=predict(modelens_relieff_50,SF_test_relieff_50);
predictions_ens_relieff_60=predict(modelens_relieff_60,SF_test_relieff_60);
predictions_ens_relieff_70=predict(modelens_relieff_70,SF_test_relieff_70);
predictions_ens_relieff_80=predict(modelens_relieff_80,SF_test_relieff_80);
predictions_ens_relieff_90=predict(modelens_relieff_90,SF_test_relieff_90);

Avaliar a performance dos modelos nos dados de teste com os features selecionados
KNN
confusionMat_knn_relieff_10 = confusionmat(Ytest, predictions_knn_relieff_10);
confusionMat_knn_relieff_20 = confusionmat(Ytest, predictions_knn_relieff_20);
confusionMat_knn_relieff_30 = confusionmat(Ytest, predictions_knn_relieff_30);
confusionMat_knn_relieff_40 = confusionmat(Ytest, predictions_knn_relieff_40);
confusionMat_knn_relieff_50 = confusionmat(Ytest, predictions_knn_relieff_50);
confusionMat_knn_relieff_60 = confusionmat(Ytest, predictions_knn_relieff_60);
confusionMat_knn_relieff_70 = confusionmat(Ytest, predictions_knn_relieff_70);
confusionMat_knn_relieff_80 = confusionmat(Ytest, predictions_knn_relieff_80);
confusionMat_knn_relieff_90 = confusionmat(Ytest, predictions_knn_relieff_90);

accuracy_knn_relieff_10 = sum(diag(confusionMat_knn_relieff_10)) / sum(confusionMat_knn_relieff_10(:));
accuracy_knn_relieff_20 = sum(diag(confusionMat_knn_relieff_20)) / sum(confusionMat_knn_relieff_20(:));
accuracy_knn_relieff_30 = sum(diag(confusionMat_knn_relieff_30)) / sum(confusionMat_knn_relieff_30(:));
accuracy_knn_relieff_40 = sum(diag(confusionMat_knn_relieff_40)) / sum(confusionMat_knn_relieff_40(:));
accuracy_knn_relieff_50 = sum(diag(confusionMat_knn_relieff_50)) / sum(confusionMat_knn_relieff_50(:));
accuracy_knn_relieff_60 = sum(diag(confusionMat_knn_relieff_60)) / sum(confusionMat_knn_relieff_60(:));
accuracy_knn_relieff_70 = sum(diag(confusionMat_knn_relieff_70)) / sum(confusionMat_knn_relieff_70(:));
accuracy_knn_relieff_80 = sum(diag(confusionMat_knn_relieff_80)) / sum(confusionMat_knn_relieff_80(:));
accuracy_knn_relieff_90 = sum(diag(confusionMat_knn_relieff_90)) / sum(confusionMat_knn_relieff_90(:));

recall_knn_relieff_10 = diag(confusionMat_knn_relieff_10) ./ sum(confusionMat_knn_relieff_10, 2);
recall_knn_relieff_20 = diag(confusionMat_knn_relieff_20) ./ sum(confusionMat_knn_relieff_20, 2);
recall_knn_relieff_30 = diag(confusionMat_knn_relieff_30) ./ sum(confusionMat_knn_relieff_30, 2);
recall_knn_relieff_40 = diag(confusionMat_knn_relieff_40) ./ sum(confusionMat_knn_relieff_40, 2);
recall_knn_relieff_50 = diag(confusionMat_knn_relieff_50) ./ sum(confusionMat_knn_relieff_50, 2);
recall_knn_relieff_60 = diag(confusionMat_knn_relieff_60) ./ sum(confusionMat_knn_relieff_60, 2);
recall_knn_relieff_70 = diag(confusionMat_knn_relieff_70) ./ sum(confusionMat_knn_relieff_70, 2);
recall_knn_relieff_80 = diag(confusionMat_knn_relieff_80) ./ sum(confusionMat_knn_relieff_80, 2);
recall_knn_relieff_90 = diag(confusionMat_knn_relieff_90) ./ sum(confusionMat_knn_relieff_90, 2);

precision_knn_relieff_10 = diag(confusionMat_knn_relieff_10) ./ sum(confusionMat_knn_relieff_10, 1)';
precision_knn_relieff_20 = diag(confusionMat_knn_relieff_20) ./ sum(confusionMat_knn_relieff_20, 1)';
precision_knn_relieff_30 = diag(confusionMat_knn_relieff_30) ./ sum(confusionMat_knn_relieff_30, 1)';
precision_knn_relieff_40 = diag(confusionMat_knn_relieff_40) ./ sum(confusionMat_knn_relieff_40, 1)';
precision_knn_relieff_50 = diag(confusionMat_knn_relieff_50) ./ sum(confusionMat_knn_relieff_50, 1)';
precision_knn_relieff_60 = diag(confusionMat_knn_relieff_60) ./ sum(confusionMat_knn_relieff_60, 1)';
precision_knn_relieff_70 = diag(confusionMat_knn_relieff_70) ./ sum(confusionMat_knn_relieff_70, 1)';
precision_knn_relieff_80 = diag(confusionMat_knn_relieff_80) ./ sum(confusionMat_knn_relieff_80, 1)';
precision_knn_relieff_90 = diag(confusionMat_knn_relieff_90) ./ sum(confusionMat_knn_relieff_90, 1)';

f1_score_knn_relieff_10 = 2 * (precision_knn_relieff_10 .* recall_knn_relieff_10) ./ (precision_knn_relieff_10 + recall_knn_relieff_10); % F1 Score
f1_score_knn_relieff_20 = 2 * (precision_knn_relieff_20 .* recall_knn_relieff_20) ./ (precision_knn_relieff_20 + recall_knn_relieff_20); % F1 Score
f1_score_knn_relieff_30 = 2 * (precision_knn_relieff_30 .* recall_knn_relieff_30) ./ (precision_knn_relieff_30 + recall_knn_relieff_30); % F1 Score
f1_score_knn_relieff_40 = 2 * (precision_knn_relieff_40 .* recall_knn_relieff_40) ./ (precision_knn_relieff_40 + recall_knn_relieff_40); % F1 Score
f1_score_knn_relieff_50 = 2 * (precision_knn_relieff_50 .* recall_knn_relieff_50) ./ (precision_knn_relieff_50 + recall_knn_relieff_50); % F1 Score
f1_score_knn_relieff_60 = 2 * (precision_knn_relieff_60 .* recall_knn_relieff_60) ./ (precision_knn_relieff_60 + recall_knn_relieff_60); % F1 Score
f1_score_knn_relieff_70 = 2 * (precision_knn_relieff_70 .* recall_knn_relieff_70) ./ (precision_knn_relieff_70 + recall_knn_relieff_70); % F1 Score
f1_score_knn_relieff_80 = 2 * (precision_knn_relieff_80 .* recall_knn_relieff_80) ./ (precision_knn_relieff_80 + recall_knn_relieff_80); % F1 Score
f1_score_knn_relieff_90 = 2 * (precision_knn_relieff_90 .* recall_knn_relieff_90) ./ (precision_knn_relieff_90 + recall_knn_relieff_90); % F1 Score

% Naive Bayes
confusionMat_nb_relieff_10 = confusionmat(Ytest, predictions_nb_relieff_10);
confusionMat_nb_relieff_20 = confusionmat(Ytest, predictions_nb_relieff_20);
confusionMat_nb_relieff_30 = confusionmat(Ytest, predictions_nb_relieff_30);
confusionMat_nb_relieff_40 = confusionmat(Ytest, predictions_nb_relieff_40);
confusionMat_nb_relieff_50 = confusionmat(Ytest, predictions_nb_relieff_50);
confusionMat_nb_relieff_60 = confusionmat(Ytest, predictions_nb_relieff_60);
confusionMat_nb_relieff_70 = confusionmat(Ytest, predictions_nb_relieff_70);
confusionMat_nb_relieff_80 = confusionmat(Ytest, predictions_nb_relieff_80);
confusionMat_nb_relieff_90 = confusionmat(Ytest, predictions_nb_relieff_90);

accuracy_nb_relieff_10 = sum(diag(confusionMat_nb_relieff_10)) / sum(confusionMat_nb_relieff_10(:));
accuracy_nb_relieff_20 = sum(diag(confusionMat_nb_relieff_20)) / sum(confusionMat_nb_relieff_20(:));
accuracy_nb_relieff_30 = sum(diag(confusionMat_nb_relieff_30)) / sum(confusionMat_nb_relieff_30(:));
accuracy_nb_relieff_40 = sum(diag(confusionMat_nb_relieff_40)) / sum(confusionMat_nb_relieff_40(:));
accuracy_nb_relieff_50 = sum(diag(confusionMat_nb_relieff_50)) / sum(confusionMat_nb_relieff_50(:));
accuracy_nb_relieff_60 = sum(diag(confusionMat_nb_relieff_60)) / sum(confusionMat_nb_relieff_60(:));
accuracy_nb_relieff_70 = sum(diag(confusionMat_nb_relieff_70)) / sum(confusionMat_nb_relieff_70(:));
accuracy_nb_relieff_80 = sum(diag(confusionMat_nb_relieff_80)) / sum(confusionMat_nb_relieff_80(:));
accuracy_nb_relieff_90 = sum(diag(confusionMat_nb_relieff_90)) / sum(confusionMat_nb_relieff_90(:));

recall_nb_relieff_10 = diag(confusionMat_nb_relieff_10) ./ sum(confusionMat_nb_relieff_10, 2);
recall_nb_relieff_20 = diag(confusionMat_nb_relieff_20) ./ sum(confusionMat_nb_relieff_20, 2);
recall_nb_relieff_30 = diag(confusionMat_nb_relieff_30) ./ sum(confusionMat_nb_relieff_30, 2);
recall_nb_relieff_40 = diag(confusionMat_nb_relieff_40) ./ sum(confusionMat_nb_relieff_40, 2);
recall_nb_relieff_50 = diag(confusionMat_nb_relieff_50) ./ sum(confusionMat_nb_relieff_50, 2);
recall_nb_relieff_60 = diag(confusionMat_nb_relieff_60) ./ sum(confusionMat_nb_relieff_60, 2);
recall_nb_relieff_70 = diag(confusionMat_nb_relieff_70) ./ sum(confusionMat_nb_relieff_70, 2);
recall_nb_relieff_80 = diag(confusionMat_nb_relieff_80) ./ sum(confusionMat_nb_relieff_80, 2);
recall_nb_relieff_90 = diag(confusionMat_nb_relieff_90) ./ sum(confusionMat_nb_relieff_90, 2);

precision_nb_relieff_10 = diag(confusionMat_nb_relieff_10) ./ sum(confusionMat_nb_relieff_10, 1)';
precision_nb_relieff_20 = diag(confusionMat_nb_relieff_20) ./ sum(confusionMat_nb_relieff_20, 1)';
precision_nb_relieff_30 = diag(confusionMat_nb_relieff_30) ./ sum(confusionMat_nb_relieff_30, 1)';
precision_nb_relieff_40 = diag(confusionMat_nb_relieff_40) ./ sum(confusionMat_nb_relieff_40, 1)';
precision_nb_relieff_50 = diag(confusionMat_nb_relieff_50) ./ sum(confusionMat_nb_relieff_50, 1)';
precision_nb_relieff_60 = diag(confusionMat_nb_relieff_60) ./ sum(confusionMat_nb_relieff_60, 1)';
precision_nb_relieff_70 = diag(confusionMat_nb_relieff_70) ./ sum(confusionMat_nb_relieff_70, 1)';
precision_nb_relieff_80 = diag(confusionMat_nb_relieff_80) ./ sum(confusionMat_nb_relieff_80, 1)';
precision_nb_relieff_90 = diag(confusionMat_nb_relieff_90) ./ sum(confusionMat_nb_relieff_90, 1)';

f1_score_nb_relieff_10 = 2 * (precision_nb_relieff_10 .* recall_nb_relieff_10) ./ (precision_nb_relieff_10 + recall_nb_relieff_10); % F1 Score
f1_score_nb_relieff_20 = 2 * (precision_nb_relieff_20 .* recall_nb_relieff_20) ./ (precision_nb_relieff_20 + recall_nb_relieff_20); % F1 Score
f1_score_nb_relieff_30 = 2 * (precision_nb_relieff_30 .* recall_nb_relieff_30) ./ (precision_nb_relieff_30 + recall_nb_relieff_30); % F1 Score
f1_score_nb_relieff_40 = 2 * (precision_nb_relieff_40 .* recall_nb_relieff_40) ./ (precision_nb_relieff_40 + recall_nb_relieff_40); % F1 Score
f1_score_nb_relieff_50 = 2 * (precision_nb_relieff_50 .* recall_nb_relieff_50) ./ (precision_nb_relieff_50 + recall_nb_relieff_50); % F1 Score
f1_score_nb_relieff_60 = 2 * (precision_nb_relieff_60 .* recall_nb_relieff_60) ./ (precision_nb_relieff_60 + recall_nb_relieff_60); % F1 Score
f1_score_nb_relieff_70 = 2 * (precision_nb_relieff_70 .* recall_nb_relieff_70) ./ (precision_nb_relieff_70 + recall_nb_relieff_70); % F1 Score
f1_score_nb_relieff_80 = 2 * (precision_nb_relieff_80 .* recall_nb_relieff_80) ./ (precision_nb_relieff_80 + recall_nb_relieff_80); % F1 Score
f1_score_nb_relieff_90 = 2 * (precision_nb_relieff_90 .* recall_nb_relieff_90) ./ (precision_nb_relieff_90 + recall_nb_relieff_90); % F1 Score

% Neural Network
confusionMat_nn_relieff_10 = confusionmat(Ytest, predictions_nn_relieff_10);
confusionMat_nn_relieff_20 = confusionmat(Ytest, predictions_nn_relieff_20);
confusionMat_nn_relieff_30 = confusionmat(Ytest, predictions_nn_relieff_30);
confusionMat_nn_relieff_40 = confusionmat(Ytest, predictions_nn_relieff_40);
confusionMat_nn_relieff_50 = confusionmat(Ytest, predictions_nn_relieff_50);
confusionMat_nn_relieff_60 = confusionmat(Ytest, predictions_nn_relieff_60);
confusionMat_nn_relieff_70 = confusionmat(Ytest, predictions_nn_relieff_70);
confusionMat_nn_relieff_80 = confusionmat(Ytest, predictions_nn_relieff_80);

accuracy_nn_relieff_10 = sum(diag(confusionMat_nn_relieff_10)) / sum(confusionMat_nn_relieff_10(:));
accuracy_nn_relieff_20 = sum(diag(confusionMat_nn_relieff_20)) / sum(confusionMat_nn_relieff_20(:));
accuracy_nn_relieff_30 = sum(diag(confusionMat_nn_relieff_30)) / sum(confusionMat_nn_relieff_30(:));
accuracy_nn_relieff_40 = sum(diag(confusionMat_nn_relieff_40)) / sum(confusionMat_nn_relieff_40(:));
accuracy_nn_relieff_50 = sum(diag(confusionMat_nn_relieff_50)) / sum(confusionMat_nn_relieff_50(:));
accuracy_nn_relieff_60 = sum(diag(confusionMat_nn_relieff_60)) / sum(confusionMat_nn_relieff_60(:));
accuracy_nn_relieff_70 = sum(diag(confusionMat_nn_relieff_70)) / sum(confusionMat_nn_relieff_70(:));
accuracy_nn_relieff_80 = sum(diag(confusionMat_nn_relieff_80)) / sum(confusionMat_nn_relieff_80(:));

recall_nn_relieff_10 = diag(confusionMat_nn_relieff_10) ./ sum(confusionMat_nn_relieff_10, 2);
recall_nn_relieff_20 = diag(confusionMat_nn_relieff_20) ./ sum(confusionMat_nn_relieff_20, 2);
recall_nn_relieff_30 = diag(confusionMat_nn_relieff_30) ./ sum(confusionMat_nn_relieff_30, 2);
recall_nn_relieff_40 = diag(confusionMat_nn_relieff_40) ./ sum(confusionMat_nn_relieff_40, 2);
recall_nn_relieff_50 = diag(confusionMat_nn_relieff_50) ./ sum(confusionMat_nn_relieff_50, 2);
recall_nn_relieff_60 = diag(confusionMat_nn_relieff_60) ./ sum(confusionMat_nn_relieff_60, 2);
recall_nn_relieff_70 = diag(confusionMat_nn_relieff_70) ./ sum(confusionMat_nn_relieff_70, 2);
recall_nn_relieff_80 = diag(confusionMat_nn_relieff_80) ./ sum(confusionMat_nn_relieff_80, 2);

precision_nn_relieff_10 = diag(confusionMat_nn_relieff_10) ./ sum(confusionMat_nn_relieff_10, 1)';
precision_nn_relieff_20 = diag(confusionMat_nn_relieff_20) ./ sum(confusionMat_nn_relieff_20, 1)';
precision_nn_relieff_30 = diag(confusionMat_nn_relieff_30) ./ sum(confusionMat_nn_relieff_30, 1)';
precision_nn_relieff_40 = diag(confusionMat_nn_relieff_40) ./ sum(confusionMat_nn_relieff_40, 1)';
precision_nn_relieff_50 = diag(confusionMat_nn_relieff_50) ./ sum(confusionMat_nn_relieff_50, 1)';
precision_nn_relieff_60 = diag(confusionMat_nn_relieff_60) ./ sum(confusionMat_nn_relieff_60, 1)';
precision_nn_relieff_70 = diag(confusionMat_nn_relieff_70) ./ sum(confusionMat_nn_relieff_70, 1)';
precision_nn_relieff_80 = diag(confusionMat_nn_relieff_80) ./ sum(confusionMat_nn_relieff_80, 1)';

f1_score_nn_relieff_10 = 2 * (precision_nn_relieff_10 .* recall_nn_relieff_10) ./ (precision_nn_relieff_10 + recall_nn_relieff_10); % F1 Score
f1_score_nn_relieff_20 = 2 * (precision_nn_relieff_20 .* recall_nn_relieff_20) ./ (precision_nn_relieff_20 + recall_nn_relieff_20); % F1 Score
f1_score_nn_relieff_30 = 2 * (precision_nn_relieff_30 .* recall_nn_relieff_30) ./ (precision_nn_relieff_30 + recall_nn_relieff_30); % F1 Score
f1_score_nn_relieff_40 = 2 * (precision_nn_relieff_40 .* recall_nn_relieff_40) ./ (precision_nn_relieff_40 + recall_nn_relieff_40); % F1 Score
f1_score_nn_relieff_50 = 2 * (precision_nn_relieff_50 .* recall_nn_relieff_50) ./ (precision_nn_relieff_50 + recall_nn_relieff_50); % F1 Score
f1_score_nn_relieff_60 = 2 * (precision_nn_relieff_60 .* recall_nn_relieff_60) ./ (precision_nn_relieff_60 + recall_nn_relieff_60); % F1 Score
f1_score_nn_relieff_70 = 2 * (precision_nn_relieff_70 .* recall_nn_relieff_70) ./ (precision_nn_relieff_70 + recall_nn_relieff_70); % F1 Score
f1_score_nn_relieff_80 = 2 * (precision_nn_relieff_80 .* recall_nn_relieff_80) ./ (precision_nn_relieff_80 + recall_nn_relieff_80); % F1 Score

% Ensemble   
confusionMat_ens_relieff_10 = confusionmat(Ytest, predictions_ens_relieff_10);
confusionMat_ens_relieff_20 = confusionmat(Ytest, predictions_ens_relieff_20);
confusionMat_ens_relieff_30 = confusionmat(Ytest, predictions_ens_relieff_30);
confusionMat_ens_relieff_40 = confusionmat(Ytest, predictions_ens_relieff_40);
confusionMat_ens_relieff_50 = confusionmat(Ytest, predictions_ens_relieff_50);
confusionMat_ens_relieff_60 = confusionmat(Ytest, predictions_ens_relieff_60);
confusionMat_ens_relieff_70 = confusionmat(Ytest, predictions_ens_relieff_70);
confusionMat_ens_relieff_80 = confusionmat(Ytest, predictions_ens_relieff_80);
confusionMat_ens_relieff_90 = confusionmat(Ytest, predictions_ens_relieff_90);

accuracy_ens_relieff_10 = sum(diag(confusionMat_ens_relieff_10)) / sum(confusionMat_ens_relieff_10(:));
accuracy_ens_relieff_20 = sum(diag(confusionMat_ens_relieff_20)) / sum(confusionMat_ens_relieff_20(:));
accuracy_ens_relieff_30 = sum(diag(confusionMat_ens_relieff_30)) / sum(confusionMat_ens_relieff_30(:));
accuracy_ens_relieff_40 = sum(diag(confusionMat_ens_relieff_40)) / sum(confusionMat_ens_relieff_40(:));
accuracy_ens_relieff_50 = sum(diag(confusionMat_ens_relieff_50)) / sum(confusionMat_ens_relieff_50(:));
accuracy_ens_relieff_60 = sum(diag(confusionMat_ens_relieff_60)) / sum(confusionMat_ens_relieff_60(:));
accuracy_ens_relieff_70 = sum(diag(confusionMat_ens_relieff_70)) / sum(confusionMat_ens_relieff_70(:));
accuracy_ens_relieff_80 = sum(diag(confusionMat_ens_relieff_80)) / sum(confusionMat_ens_relieff_80(:));
accuracy_ens_relieff_90 = sum(diag(confusionMat_ens_relieff_90)) / sum(confusionMat_ens_relieff_90(:));

recall_ens_relieff_10 = diag(confusionMat_ens_relieff_10) ./ sum(confusionMat_ens_relieff_10, 2);
recall_ens_relieff_20 = diag(confusionMat_ens_relieff_20) ./ sum(confusionMat_ens_relieff_20, 2);
recall_ens_relieff_30 = diag(confusionMat_ens_relieff_30) ./ sum(confusionMat_ens_relieff_30, 2);
recall_ens_relieff_40 = diag(confusionMat_ens_relieff_40) ./ sum(confusionMat_ens_relieff_40, 2);
recall_ens_relieff_50 = diag(confusionMat_ens_relieff_50) ./ sum(confusionMat_ens_relieff_50, 2);
recall_ens_relieff_60 = diag(confusionMat_ens_relieff_60) ./ sum(confusionMat_ens_relieff_60, 2);
recall_ens_relieff_70 = diag(confusionMat_ens_relieff_70) ./ sum(confusionMat_ens_relieff_70, 2);
recall_ens_relieff_80 = diag(confusionMat_ens_relieff_80) ./ sum(confusionMat_ens_relieff_80, 2);
recall_ens_relieff_90 = diag(confusionMat_ens_relieff_90) ./ sum(confusionMat_ens_relieff_90, 2);

precision_ens_relieff_10 = diag(confusionMat_ens_relieff_10) ./ sum(confusionMat_ens_relieff_10, 1)';
precision_ens_relieff_20 = diag(confusionMat_ens_relieff_20) ./ sum(confusionMat_ens_relieff_20, 1)';
precision_ens_relieff_30 = diag(confusionMat_ens_relieff_30) ./ sum(confusionMat_ens_relieff_30, 1)';
precision_ens_relieff_40 = diag(confusionMat_ens_relieff_40) ./ sum(confusionMat_ens_relieff_40, 1)';
precision_ens_relieff_50 = diag(confusionMat_ens_relieff_50) ./ sum(confusionMat_ens_relieff_50, 1)';
precision_ens_relieff_60 = diag(confusionMat_ens_relieff_60) ./ sum(confusionMat_ens_relieff_60, 1)';
precision_ens_relieff_70 = diag(confusionMat_ens_relieff_70) ./ sum(confusionMat_ens_relieff_70, 1)';
precision_ens_relieff_80 = diag(confusionMat_ens_relieff_80) ./ sum(confusionMat_ens_relieff_80, 1)';
precision_ens_relieff_90 = diag(confusionMat_ens_relieff_90) ./ sum(confusionMat_ens_relieff_90, 1)';

f1_score_ens_relieff_10 = 2 * (precision_ens_relieff_10 .* recall_ens_relieff_10) ./ (precision_ens_relieff_10 + recall_ens_relieff_10); % F1 Score
f1_score_ens_relieff_20 = 2 * (precision_ens_relieff_20 .* recall_ens_relieff_20) ./ (precision_ens_relieff_20 + recall_ens_relieff_20); % F1 Score
f1_score_ens_relieff_30 = 2 * (precision_ens_relieff_30 .* recall_ens_relieff_30) ./ (precision_ens_relieff_30 + recall_ens_relieff_30); % F1 Score
f1_score_ens_relieff_40 = 2 * (precision_ens_relieff_40 .* recall_ens_relieff_40) ./ (precision_ens_relieff_40 + recall_ens_relieff_40); % F1 Score
f1_score_ens_relieff_50 = 2 * (precision_ens_relieff_50 .* recall_ens_relieff_50) ./ (precision_ens_relieff_50 + recall_ens_relieff_50); % F1 Score
f1_score_ens_relieff_60 = 2 * (precision_ens_relieff_60 .* recall_ens_relieff_60) ./ (precision_ens_relieff_60 + recall_ens_relieff_60); % F1 Score
f1_score_ens_relieff_70 = 2 * (precision_ens_relieff_70 .* recall_ens_relieff_70) ./ (precision_ens_relieff_70 + recall_ens_relieff_70); % F1 Score
f1_score_ens_relieff_80 = 2 * (precision_ens_relieff_80 .* recall_ens_relieff_80) ./ (precision_ens_relieff_80 + recall_ens_relieff_80); % F1 Score
f1_score_ens_relieff_90 = 2 * (precision_ens_relieff_90 .* recall_ens_relieff_90) ./ (precision_ens_relieff_90 + recall_ens_relieff_90); % F1 Score

%     % Exibir a acurácia e a matriz de confusão
% disp(['Acurácia_KNN_ReliefF: ', num2str(accuracy_knn_relieff)]);
% disp(['Accuracy_Naive Bayes_ReliefF: ', num2str(accuracy_nb_relieff)]);
% disp(['Accuracy_Neural Network_ReliefF: ', num2str(accuracy_nn_relieff)]);
% disp(['Accuracy_Ensemble_ReliefF: ', num2str(accuracy_ens_relieff)]);
% % disp('Matriz de Confusão_KNN_ReliefF:');
% % disp(confusionMat_knn_relieff);
% % disp('Matriz de Confusão_Naive Bayes_ReliefF:');
% % disp(confusionMat_nb_relieff);
% % disp('Matriz de Confusão_Neural Network_ReliefF:');
% % disp(confusionMat_nn_relieff);
% % disp('Matriz de Confusão_Ensemble_ReliefF:');
% % disp(confusionMat_ens_relieff);

% % % Plotar as matrizes de confusão
% figure;
% confusionchart(confusionMat_knn_relieff_10,classes,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('Confusion Matrix - KNN - Top 10 Features Selected by ReliefF');
% figure;
% confusionchart(confusionMat_nb_relieff_60,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('Confusion Matrix - Naive Bayes - Top 60 Features Selected by ReliefF');
% figure;
% confusionchart(confusionMat_nn_relieff_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('Confusion Matrix - Neural Network - Top 20 Features Selected by ReliefF');
% figure;
% confusionchart(confusionMat_ens_relieff_80,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('Confusion Matrix - Ensemble - Top 80 Features Selected by ReliefF');
% figure Name 'Matrix Confusion - Features selected by ReliefF - Features thresholds with best accuracy and F1-score'
% subplot(2, 2, 1);
% confusionchart(confusionMat_knn_relieff_10,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('KNN - Top 10 Features Selected by ReliefF');
% subplot(2, 2, 2);
% confusionchart(confusionMat_nb_relieff_60,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('Naive Bayes - Top 60 Features Selected by ReliefF');
% subplot(2, 2, 3);
% confusionchart(confusionMat_nn_relieff_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('Neural Network - Top 20 Features Selected by ReliefF');
% subplot(2, 2, 4);
% confusionchart(confusionMat_ens_relieff_80,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% title('Ensemble - Top 80 Features Selected by ReliefF');
% % % % 



% Dados de exemplo
numAttributes = [10, 20, 30, 40, 50, 60, 70, 80, 90];
% Precisões para ReliefF e Naive Bayes
accuracy_nb_relieff = [
    accuracy_nb_relieff_10, 
    accuracy_nb_relieff_20, 
    accuracy_nb_relieff_30, 
    accuracy_nb_relieff_40, 
    accuracy_nb_relieff_50, 
    accuracy_nb_relieff_60, 
    accuracy_nb_relieff_70, 
    accuracy_nb_relieff_80, 
    accuracy_nb_relieff_90
];
% Criar gráfico de linha
figure;
hold on;
plot(numAttributes, accuracy_nb_relieff, '-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Naive Bayes com ReliefF');
hold off;
% Adicionar rótulos aos eixos e título
xlabel('Número de Atributos Selecionados');
ylabel('Precisão');
title('Comparação de Precisão vs. Número de Atributos Selecionados');
% Adicionar grade
grid on;

% Adicionar legenda
legend('Location', 'best');
% Acurácias para ReliefF e Neural Network
accuracy_nn_relieff = [
    accuracy_nn_relieff_10, 
    accuracy_nn_relieff_20, 
    accuracy_nn_relieff_30, 
    accuracy_nn_relieff_40, 
    accuracy_nn_relieff_50, 
    accuracy_nn_relieff_60, 
    accuracy_nn_relieff_70, 
    accuracy_nn_relieff_80, 
    accuracy_nn_relieff_90
];
% Criar gráfico de linha
figure;
hold on;
plot(numAttributes, accuracy_nn_relieff, '-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Neural Network com ReliefF');
hold off;
% Adicionar rótulos aos eixos e título
xlabel('Número de Atributos Selecionados');
ylabel('Precisão');
title('Comparação de Precisão vs. Número de Atributos Selecionados');
% Adicionar grade
grid on;
% Adicionar legenda
legend('Location', 'best');

% Dados de exemplo para recall
numAttributes = [10, 20, 30, 40, 50, 60, 70, 80, 90];
% Recall para ReliefF e Neural Network
recall_nn_relieff = [
    mean(recall_nn_relieff_10), 
    mean(recall_nn_relieff_20), 
    mean(recall_nn_relieff_30), 
    mean(recall_nn_relieff_40), 
    mean(recall_nn_relieff_50), 
    mean(recall_nn_relieff_60), 
    mean(recall_nn_relieff_70), 
    mean(recall_nn_relieff_80), 
    mean(recall_nn_relieff_90)
];
% Criar gráfico de linha
figure;
plot(numAttributes, recall_nn_relieff, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Neural Network com ReliefF');
% Adicionar rótulos aos eixos e título
xlabel('Número de Atributos Selecionados');
ylabel('Recall');
title('Recall vs. Número de Atributos Selecionados (Neural Network com ReliefF)');
% Adicionar grade
grid on;
% Adicionar legenda
legend('Location', 'best');




% MRMR
SF_train_mrmr_10 = X_train(:, ranking2(1:10)); % dados de treino com melhores features
SF_test_mrmr_10 = X_test(:, ranking2(1:10)); % dados de teste com melhores features
SF_train_mrmr_20 = X_train(:, ranking2(1:20)); % dados de treino com melhores features
SF_test_mrmr_20 = X_test(:, ranking2(1:20)); % dados de teste com melhores features
SF_train_mrmr_30 = X_train(:, ranking2(1:30)); % dados de treino com melhores features
SF_test_mrmr_30 = X_test(:, ranking2(1:30)); % dados de teste com melhores features
SF_train_mrmr_40 = X_train(:, ranking2(1:40)); % dados de treino com melhores features
SF_test_mrmr_40 = X_test(:, ranking2(1:40)); % dados de teste com melhores features
SF_train_mrmr_50 = X_train(:, ranking2(1:50)); % dados de treino com melhores features
SF_test_mrmr_50 = X_test(:, ranking2(1:50)); % dados de teste com melhores features
SF_train_mrmr_60 = X_train(:, ranking2(1:60)); % dados de treino com melhores features
SF_test_mrmr_60 = X_test(:, ranking2(1:60)); % dados de teste com melhores features
SF_train_mrmr_70 = X_train(:, ranking2(1:70)); % dados de treino com melhores features
SF_test_mrmr_70 = X_test(:, ranking2(1:70)); % dados de teste com melhores features
SF_train_mrmr_80 = X_train(:, ranking2(1:80)); % dados de treino com melhores features
SF_test_mrmr_80 = X_test(:, ranking2(1:80)); % dados de teste com melhores features
SF_train_mrmr_90 = X_train(:, ranking2(1:90)); % dados de treino com melhores features
SF_test_mrmr_90 = X_test(:, ranking2(1:90)); % dados de teste com melhores features

% Treinar o modelo novamente usando apenas os recursos selecionados
  modelknn_mrmr_10 = fitcknn(SF_train_mrmr_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_mrmr_20 = fitcknn(SF_train_mrmr_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_mrmr_30 = fitcknn(SF_train_mrmr_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_mrmr_40 = fitcknn(SF_train_mrmr_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_mrmr_50 = fitcknn(SF_train_mrmr_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_mrmr_60 = fitcknn(SF_train_mrmr_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_mrmr_70 = fitcknn(SF_train_mrmr_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_mrmr_80 = fitcknn(SF_train_mrmr_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));

  % modelnb_mrmr_10 = fitcnb(SF_train_mrmr_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  % modelnb_mrmr_20 = fitcnb(SF_train_mrmr_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  % modelnb_mrmr_30 = fitcnb(SF_train_mrmr_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  % modelnb_mrmr_40 = fitcnb(SF_train_mrmr_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  % modelnb_mrmr_50 = fitcnb(SF_train_mrmr_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  % modelnb_mrmr_60 = fitcnb(SF_train_mrmr_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  % modelnb_mrmr_70 = fitcnb(SF_train_mrmr_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  % modelnb_mrmr_80 = fitcnb(SF_train_mrmr_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  %
  modelnn_mrmr_10 = fitcnet(SF_train_mrmr_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_mrmr_20 = fitcnet(SF_train_mrmr_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_mrmr_30 = fitcnet(SF_train_mrmr_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_mrmr_40 = fitcnet(SF_train_mrmr_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_mrmr_50 = fitcnet(SF_train_mrmr_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_mrmr_60 = fitcnet(SF_train_mrmr_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_mrmr_70 = fitcnet(SF_train_mrmr_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_mrmr_80 = fitcnet(SF_train_mrmr_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));


  modelens_mrmr_10 = fitcensemble(SF_train_mrmr_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_mrmr_20 = fitcensemble(SF_train_mrmr_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_mrmr_30 = fitcensemble(SF_train_mrmr_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_mrmr_40 = fitcensemble(SF_train_mrmr_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_mrmr_50 = fitcensemble(SF_train_mrmr_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_mrmr_60 = fitcensemble(SF_train_mrmr_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_mrmr_70 = fitcensemble(SF_train_mrmr_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_mrmr_80 = fitcensemble(SF_train_mrmr_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));


% Testar o modelo novamente usando apenas os recursos selecionados
predictions_knn_mrmr_10=predict(modelknn_mrmr_10,SF_test_mrmr_10);
predictions_knn_mrmr_20=predict(modelknn_mrmr_20,SF_test_mrmr_20);
predictions_knn_mrmr_30=predict(modelknn_mrmr_30,SF_test_mrmr_30);
predictions_knn_mrmr_40=predict(modelknn_mrmr_40,SF_test_mrmr_40);
predictions_knn_mrmr_50=predict(modelknn_mrmr_50,SF_test_mrmr_50);
predictions_knn_mrmr_60=predict(modelknn_mrmr_60,SF_test_mrmr_60);
predictions_knn_mrmr_70=predict(modelknn_mrmr_70,SF_test_mrmr_70);
predictions_knn_mrmr_80=predict(modelknn_mrmr_80,SF_test_mrmr_80);

predictions_nb_mrmr_10=predict(modelnb_mrmr_10,SF_test_mrmr_10);
predictions_nb_mrmr_20=predict(modelnb_mrmr_20,SF_test_mrmr_20);
predictions_nb_mrmr_30=predict(modelnb_mrmr_30,SF_test_mrmr_30);
predictions_nb_mrmr_40=predict(modelnb_mrmr_40,SF_test_mrmr_40);
predictions_nb_mrmr_50=predict(modelnb_mrmr_50,SF_test_mrmr_50);
predictions_nb_mrmr_60=predict(modelnb_mrmr_60,SF_test_mrmr_60);
predictions_nb_mrmr_70=predict(modelnb_mrmr_70,SF_test_mrmr_70);
predictions_nb_mrmr_80=predict(modelnb_mrmr_80,SF_test_mrmr_80);

predictions_nn_mrmr_10=predict(modelnn_mrmr_10,SF_test_mrmr_10);
predictions_nn_mrmr_20=predict(modelnn_mrmr_20,SF_test_mrmr_20);
predictions_nn_mrmr_30=predict(modelnn_mrmr_30,SF_test_mrmr_30);
predictions_nn_mrmr_40=predict(modelnn_mrmr_40,SF_test_mrmr_40);
predictions_nn_mrmr_50=predict(modelnn_mrmr_50,SF_test_mrmr_50);
predictions_nn_mrmr_60=predict(modelnn_mrmr_60,SF_test_mrmr_60);
predictions_nn_mrmr_70=predict(modelnn_mrmr_70,SF_test_mrmr_70);
predictions_nn_mrmr_80=predict(modelnn_mrmr_80,SF_test_mrmr_80);

predictions_ens_mrmr_10=predict(modelens_mrmr_10,SF_test_mrmr_10);
predictions_ens_mrmr_20=predict(modelens_mrmr_20,SF_test_mrmr_20);
predictions_ens_mrmr_30=predict(modelens_mrmr_30,SF_test_mrmr_30);
predictions_ens_mrmr_40=predict(modelens_mrmr_40,SF_test_mrmr_40);
predictions_ens_mrmr_50=predict(modelens_mrmr_50,SF_test_mrmr_50);
predictions_ens_mrmr_60=predict(modelens_mrmr_60,SF_test_mrmr_60);
predictions_ens_mrmr_70=predict(modelens_mrmr_70,SF_test_mrmr_70);
predictions_ens_mrmr_80=predict(modelens_mrmr_80,SF_test_mrmr_80);

% Avaliar a performance dos modelos nos dados de teste com os features selecionados
% KNN
confusionMat_knn_mrmr_10 = confusionmat(Ytest, predictions_knn_mrmr_10);
confusionMat_knn_mrmr_20 = confusionmat(Ytest, predictions_knn_mrmr_20);
confusionMat_knn_mrmr_30 = confusionmat(Ytest, predictions_knn_mrmr_30);
confusionMat_knn_mrmr_40 = confusionmat(Ytest, predictions_knn_mrmr_40);
confusionMat_knn_mrmr_50 = confusionmat(Ytest, predictions_knn_mrmr_50);
confusionMat_knn_mrmr_60 = confusionmat(Ytest, predictions_knn_mrmr_60);
confusionMat_knn_mrmr_70 = confusionmat(Ytest, predictions_knn_mrmr_70);
confusionMat_knn_mrmr_80 = confusionmat(Ytest, predictions_knn_mrmr_80);

accuracy_knn_mrmr_10 = sum(diag(confusionMat_knn_mrmr_10)) / sum(confusionMat_knn_mrmr_10(:));
accuracy_knn_mrmr_20 = sum(diag(confusionMat_knn_mrmr_20)) / sum(confusionMat_knn_mrmr_20(:));
accuracy_knn_mrmr_30 = sum(diag(confusionMat_knn_mrmr_30)) / sum(confusionMat_knn_mrmr_30(:));
accuracy_knn_mrmr_40 = sum(diag(confusionMat_knn_mrmr_40)) / sum(confusionMat_knn_mrmr_40(:));
accuracy_knn_mrmr_50 = sum(diag(confusionMat_knn_mrmr_50)) / sum(confusionMat_knn_mrmr_50(:));
accuracy_knn_mrmr_60 = sum(diag(confusionMat_knn_mrmr_60)) / sum(confusionMat_knn_mrmr_60(:));
accuracy_knn_mrmr_70 = sum(diag(confusionMat_knn_mrmr_70)) / sum(confusionMat_knn_mrmr_70(:));
accuracy_knn_mrmr_80 = sum(diag(confusionMat_knn_mrmr_80)) / sum(confusionMat_knn_mrmr_80(:));

recall_knn_mrmr_10 = diag(confusionMat_knn_mrmr_10) ./ sum(confusionMat_knn_mrmr_10, 2);
recall_knn_mrmr_20 = diag(confusionMat_knn_mrmr_20) ./ sum(confusionMat_knn_mrmr_20, 2);
recall_knn_mrmr_30 = diag(confusionMat_knn_mrmr_30) ./ sum(confusionMat_knn_mrmr_30, 2);
recall_knn_mrmr_40 = diag(confusionMat_knn_mrmr_40) ./ sum(confusionMat_knn_mrmr_40, 2);
recall_knn_mrmr_50 = diag(confusionMat_knn_mrmr_50) ./ sum(confusionMat_knn_mrmr_50, 2);
recall_knn_mrmr_60 = diag(confusionMat_knn_mrmr_60) ./ sum(confusionMat_knn_mrmr_60, 2);
recall_knn_mrmr_70 = diag(confusionMat_knn_mrmr_70) ./ sum(confusionMat_knn_mrmr_70, 2);
recall_knn_mrmr_80 = diag(confusionMat_knn_mrmr_80) ./ sum(confusionMat_knn_mrmr_80, 2);

precision_knn_mrmr_10 = diag(confusionMat_knn_mrmr_10) ./ sum(confusionMat_knn_mrmr_10, 1)';
precision_knn_mrmr_20 = diag(confusionMat_knn_mrmr_20) ./ sum(confusionMat_knn_mrmr_20, 1)';
precision_knn_mrmr_30 = diag(confusionMat_knn_mrmr_30) ./ sum(confusionMat_knn_mrmr_30, 1)';
precision_knn_mrmr_40 = diag(confusionMat_knn_mrmr_40) ./ sum(confusionMat_knn_mrmr_40, 1)';
precision_knn_mrmr_50 = diag(confusionMat_knn_mrmr_50) ./ sum(confusionMat_knn_mrmr_50, 1)';
precision_knn_mrmr_60 = diag(confusionMat_knn_mrmr_60) ./ sum(confusionMat_knn_mrmr_60, 1)';
precision_knn_mrmr_70 = diag(confusionMat_knn_mrmr_70) ./ sum(confusionMat_knn_mrmr_70, 1)';
precision_knn_mrmr_80 = diag(confusionMat_knn_mrmr_80) ./ sum(confusionMat_knn_mrmr_80, 1)';

f1_score_knn_mrmr_10 = 2 * (precision_knn_mrmr_10 .* recall_knn_mrmr_10) ./ (precision_knn_mrmr_10 + recall_knn_mrmr_10); % F1 Score
f1_score_knn_mrmr_20 = 2 * (precision_knn_mrmr_20 .* recall_knn_mrmr_20) ./ (precision_knn_mrmr_20 + recall_knn_mrmr_20); % F1 Score
f1_score_knn_mrmr_30 = 2 * (precision_knn_mrmr_30 .* recall_knn_mrmr_30) ./ (precision_knn_mrmr_30 + recall_knn_mrmr_30); % F1 Score
f1_score_knn_mrmr_40 = 2 * (precision_knn_mrmr_40 .* recall_knn_mrmr_40) ./ (precision_knn_mrmr_40 + recall_knn_mrmr_40); % F1 Score
f1_score_knn_mrmr_50 = 2 * (precision_knn_mrmr_50 .* recall_knn_mrmr_50) ./ (precision_knn_mrmr_50 + recall_knn_mrmr_50); % F1 Score
f1_score_knn_mrmr_60 = 2 * (precision_knn_mrmr_60 .* recall_knn_mrmr_60) ./ (precision_knn_mrmr_60 + recall_knn_mrmr_60); % F1 Score
f1_score_knn_mrmr_70 = 2 * (precision_knn_mrmr_70 .* recall_knn_mrmr_70) ./ (precision_knn_mrmr_70 + recall_knn_mrmr_70); % F1 Score
f1_score_knn_mrmr_80 = 2 * (precision_knn_mrmr_80 .* recall_knn_mrmr_80) ./ (precision_knn_mrmr_80 + recall_knn_mrmr_80); % F1 Score

% Naive Bayes
confusionMat_nb_mrmr_10 = confusionmat(Ytest, predictions_nb_mrmr_10);
confusionMat_nb_mrmr_20 = confusionmat(Ytest, predictions_nb_mrmr_20);
confusionMat_nb_mrmr_30 = confusionmat(Ytest, predictions_nb_mrmr_30);
confusionMat_nb_mrmr_40 = confusionmat(Ytest, predictions_nb_mrmr_40);
confusionMat_nb_mrmr_50 = confusionmat(Ytest, predictions_nb_mrmr_50);
confusionMat_nb_mrmr_60 = confusionmat(Ytest, predictions_nb_mrmr_60);
confusionMat_nb_mrmr_70 = confusionmat(Ytest, predictions_nb_mrmr_70);
confusionMat_nb_mrmr_80 = confusionmat(Ytest, predictions_nb_mrmr_80);

accuracy_nb_mrmr_10 = sum(diag(confusionMat_nb_mrmr_10)) / sum(confusionMat_nb_mrmr_10(:));
accuracy_nb_mrmr_20 = sum(diag(confusionMat_nb_mrmr_20)) / sum(confusionMat_nb_mrmr_20(:));
accuracy_nb_mrmr_30 = sum(diag(confusionMat_nb_mrmr_30)) / sum(confusionMat_nb_mrmr_30(:));
accuracy_nb_mrmr_40 = sum(diag(confusionMat_nb_mrmr_40)) / sum(confusionMat_nb_mrmr_40(:));
accuracy_nb_mrmr_50 = sum(diag(confusionMat_nb_mrmr_50)) / sum(confusionMat_nb_mrmr_50(:));
accuracy_nb_mrmr_60 = sum(diag(confusionMat_nb_mrmr_60)) / sum(confusionMat_nb_mrmr_60(:));
accuracy_nb_mrmr_70 = sum(diag(confusionMat_nb_mrmr_70)) / sum(confusionMat_nb_mrmr_70(:));
accuracy_nb_mrmr_80 = sum(diag(confusionMat_nb_mrmr_80)) / sum(confusionMat_nb_mrmr_80(:));

recall_nb_mrmr_10 = diag(confusionMat_nb_mrmr_10) ./ sum(confusionMat_nb_mrmr_10, 2);
recall_nb_mrmr_20 = diag(confusionMat_nb_mrmr_20) ./ sum(confusionMat_nb_mrmr_20, 2);
recall_nb_mrmr_30 = diag(confusionMat_nb_mrmr_30) ./ sum(confusionMat_nb_mrmr_30, 2);
recall_nb_mrmr_40 = diag(confusionMat_nb_mrmr_40) ./ sum(confusionMat_nb_mrmr_40, 2);
recall_nb_mrmr_50 = diag(confusionMat_nb_mrmr_50) ./ sum(confusionMat_nb_mrmr_50, 2);
recall_nb_mrmr_60 = diag(confusionMat_nb_mrmr_60) ./ sum(confusionMat_nb_mrmr_60, 2);
recall_nb_mrmr_70 = diag(confusionMat_nb_mrmr_70) ./ sum(confusionMat_nb_mrmr_70, 2);
recall_nb_mrmr_80 = diag(confusionMat_nb_mrmr_80) ./ sum(confusionMat_nb_mrmr_80, 2);

precision_nb_mrmr_10 = diag(confusionMat_nb_mrmr_10) ./ sum(confusionMat_nb_mrmr_10, 1)';
precision_nb_mrmr_20 = diag(confusionMat_nb_mrmr_20) ./ sum(confusionMat_nb_mrmr_20, 1)';
precision_nb_mrmr_30 = diag(confusionMat_nb_mrmr_30) ./ sum(confusionMat_nb_mrmr_30, 1)';
precision_nb_mrmr_40 = diag(confusionMat_nb_mrmr_40) ./ sum(confusionMat_nb_mrmr_40, 1)';
precision_nb_mrmr_50 = diag(confusionMat_nb_mrmr_50) ./ sum(confusionMat_nb_mrmr_50, 1)';
precision_nb_mrmr_60 = diag(confusionMat_nb_mrmr_60) ./ sum(confusionMat_nb_mrmr_60, 1)';
precision_nb_mrmr_70 = diag(confusionMat_nb_mrmr_70) ./ sum(confusionMat_nb_mrmr_70, 1)';
precision_nb_mrmr_80 = diag(confusionMat_nb_mrmr_80) ./ sum(confusionMat_nb_mrmr_80, 1)';

f1_score_nb_mrmr_10 = 2 * (precision_nb_mrmr_10 .* recall_nb_mrmr_10) ./ (precision_nb_mrmr_10 + recall_nb_mrmr_10); % F1 Score
f1_score_nb_mrmr_20 = 2 * (precision_nb_mrmr_20 .* recall_nb_mrmr_20) ./ (precision_nb_mrmr_20 + recall_nb_mrmr_20); % F1 Score
f1_score_nb_mrmr_30 = 2 * (precision_nb_mrmr_30 .* recall_nb_mrmr_30) ./ (precision_nb_mrmr_30 + recall_nb_mrmr_30); % F1 Score
f1_score_nb_mrmr_40 = 2 * (precision_nb_mrmr_40 .* recall_nb_mrmr_40) ./ (precision_nb_mrmr_40 + recall_nb_mrmr_40); % F1 Score
f1_score_nb_mrmr_50 = 2 * (precision_nb_mrmr_50 .* recall_nb_mrmr_50) ./ (precision_nb_mrmr_50 + recall_nb_mrmr_50); % F1 Score
f1_score_nb_mrmr_60 = 2 * (precision_nb_mrmr_60 .* recall_nb_mrmr_60) ./ (precision_nb_mrmr_60 + recall_nb_mrmr_60); % F1 Score
f1_score_nb_mrmr_70 = 2 * (precision_nb_mrmr_70 .* recall_nb_mrmr_70) ./ (precision_nb_mrmr_70 + recall_nb_mrmr_70); % F1 Score
f1_score_nb_mrmr_80 = 2 * (precision_nb_mrmr_80 .* recall_nb_mrmr_80) ./ (precision_nb_mrmr_80 + recall_nb_mrmr_80); % F1 Score

% Neural Network
confusionMat_nn_mrmr_10 = confusionmat(Ytest, predictions_nn_mrmr_10);
confusionMat_nn_mrmr_20 = confusionmat(Ytest, predictions_nn_mrmr_20);
confusionMat_nn_mrmr_30 = confusionmat(Ytest, predictions_nn_mrmr_30);
confusionMat_nn_mrmr_40 = confusionmat(Ytest, predictions_nn_mrmr_40);
confusionMat_nn_mrmr_50 = confusionmat(Ytest, predictions_nn_mrmr_50);
confusionMat_nn_mrmr_60 = confusionmat(Ytest, predictions_nn_mrmr_60);
confusionMat_nn_mrmr_70 = confusionmat(Ytest, predictions_nn_mrmr_70);
confusionMat_nn_mrmr_80 = confusionmat(Ytest, predictions_nn_mrmr_80);

accuracy_nn_mrmr_10 = sum(diag(confusionMat_nn_mrmr_10)) / sum(confusionMat_nn_mrmr_10(:));
accuracy_nn_mrmr_20 = sum(diag(confusionMat_nn_mrmr_20)) / sum(confusionMat_nn_mrmr_20(:));
accuracy_nn_mrmr_30 = sum(diag(confusionMat_nn_mrmr_30)) / sum(confusionMat_nn_mrmr_30(:));
accuracy_nn_mrmr_40 = sum(diag(confusionMat_nn_mrmr_40)) / sum(confusionMat_nn_mrmr_40(:));
accuracy_nn_mrmr_50 = sum(diag(confusionMat_nn_mrmr_50)) / sum(confusionMat_nn_mrmr_50(:));
accuracy_nn_mrmr_60 = sum(diag(confusionMat_nn_mrmr_60)) / sum(confusionMat_nn_mrmr_60(:));
accuracy_nn_mrmr_70 = sum(diag(confusionMat_nn_mrmr_70)) / sum(confusionMat_nn_mrmr_70(:));
accuracy_nn_mrmr_80 = sum(diag(confusionMat_nn_mrmr_80)) / sum(confusionMat_nn_mrmr_80(:));

recall_nn_mrmr_10 = diag(confusionMat_nn_mrmr_10) ./ sum(confusionMat_nn_mrmr_10, 2);
recall_nn_mrmr_20 = diag(confusionMat_nn_mrmr_20) ./ sum(confusionMat_nn_mrmr_20, 2);
recall_nn_mrmr_30 = diag(confusionMat_nn_mrmr_30) ./ sum(confusionMat_nn_mrmr_30, 2);
recall_nn_mrmr_40 = diag(confusionMat_nn_mrmr_40) ./ sum(confusionMat_nn_mrmr_40, 2);
recall_nn_mrmr_50 = diag(confusionMat_nn_mrmr_50) ./ sum(confusionMat_nn_mrmr_50, 2);
recall_nn_mrmr_60 = diag(confusionMat_nn_mrmr_60) ./ sum(confusionMat_nn_mrmr_60, 2);
recall_nn_mrmr_70 = diag(confusionMat_nn_mrmr_70) ./ sum(confusionMat_nn_mrmr_70, 2);
recall_nn_mrmr_80 = diag(confusionMat_nn_mrmr_80) ./ sum(confusionMat_nn_mrmr_80, 2);

precision_nn_mrmr_10 = diag(confusionMat_nn_mrmr_10) ./ sum(confusionMat_nn_mrmr_10, 1)';
precision_nn_mrmr_20 = diag(confusionMat_nn_mrmr_20) ./ sum(confusionMat_nn_mrmr_20, 1)';
precision_nn_mrmr_30 = diag(confusionMat_nn_mrmr_30) ./ sum(confusionMat_nn_mrmr_30, 1)';
precision_nn_mrmr_40 = diag(confusionMat_nn_mrmr_40) ./ sum(confusionMat_nn_mrmr_40, 1)';
precision_nn_mrmr_50 = diag(confusionMat_nn_mrmr_50) ./ sum(confusionMat_nn_mrmr_50, 1)';
precision_nn_mrmr_60 = diag(confusionMat_nn_mrmr_60) ./ sum(confusionMat_nn_mrmr_60, 1)';
precision_nn_mrmr_70 = diag(confusionMat_nn_mrmr_70) ./ sum(confusionMat_nn_mrmr_70, 1)';
precision_nn_mrmr_80 = diag(confusionMat_nn_mrmr_80) ./ sum(confusionMat_nn_mrmr_80, 1)';

f1_score_nn_mrmr_10 = 2 * (precision_nn_mrmr_10 .* recall_nn_mrmr_10) ./ (precision_nn_mrmr_10 + recall_nn_mrmr_10); % F1 Score
f1_score_nn_mrmr_20 = 2 * (precision_nn_mrmr_20 .* recall_nn_mrmr_20) ./ (precision_nn_mrmr_20 + recall_nn_mrmr_20); % F1 Score
f1_score_nn_mrmr_30 = 2 * (precision_nn_mrmr_30 .* recall_nn_mrmr_30) ./ (precision_nn_mrmr_30 + recall_nn_mrmr_30); % F1 Score
f1_score_nn_mrmr_40 = 2 * (precision_nn_mrmr_40 .* recall_nn_mrmr_40) ./ (precision_nn_mrmr_40 + recall_nn_mrmr_40); % F1 Score
f1_score_nn_mrmr_50 = 2 * (precision_nn_mrmr_50 .* recall_nn_mrmr_50) ./ (precision_nn_mrmr_50 + recall_nn_mrmr_50); % F1 Score
f1_score_nn_mrmr_60 = 2 * (precision_nn_mrmr_60 .* recall_nn_mrmr_60) ./ (precision_nn_mrmr_60 + recall_nn_mrmr_60); % F1 Score
f1_score_nn_mrmr_70 = 2 * (precision_nn_mrmr_70 .* recall_nn_mrmr_70) ./ (precision_nn_mrmr_70 + recall_nn_mrmr_70); % F1 Score
f1_score_nn_mrmr_80 = 2 * (precision_nn_mrmr_80 .* recall_nn_mrmr_80) ./ (precision_nn_mrmr_80 + recall_nn_mrmr_80); % F1 Score

% Ensemble   
confusionMat_ens_mrmr_10 = confusionmat(Ytest, predictions_ens_mrmr_10);
confusionMat_ens_mrmr_20 = confusionmat(Ytest, predictions_ens_mrmr_20);
confusionMat_ens_mrmr_30 = confusionmat(Ytest, predictions_ens_mrmr_30);
confusionMat_ens_mrmr_40 = confusionmat(Ytest, predictions_ens_mrmr_40);
confusionMat_ens_mrmr_50 = confusionmat(Ytest, predictions_ens_mrmr_50);
confusionMat_ens_mrmr_60 = confusionmat(Ytest, predictions_ens_mrmr_60);
confusionMat_ens_mrmr_70 = confusionmat(Ytest, predictions_ens_mrmr_70);
confusionMat_ens_mrmr_80 = confusionmat(Ytest, predictions_ens_mrmr_80);
confusionMat_ens_mrmr_90 = confusionmat(Ytest, predictions_ens_mrmr_90);

accuracy_ens_mrmr_10 = sum(diag(confusionMat_ens_mrmr_10)) / sum(confusionMat_ens_mrmr_10(:));
accuracy_ens_mrmr_20 = sum(diag(confusionMat_ens_mrmr_20)) / sum(confusionMat_ens_mrmr_20(:));
accuracy_ens_mrmr_30 = sum(diag(confusionMat_ens_mrmr_30)) / sum(confusionMat_ens_mrmr_30(:));
accuracy_ens_mrmr_40 = sum(diag(confusionMat_ens_mrmr_40)) / sum(confusionMat_ens_mrmr_40(:));
accuracy_ens_mrmr_50 = sum(diag(confusionMat_ens_mrmr_50)) / sum(confusionMat_ens_mrmr_50(:));
accuracy_ens_mrmr_60 = sum(diag(confusionMat_ens_mrmr_60)) / sum(confusionMat_ens_mrmr_60(:));
accuracy_ens_mrmr_70 = sum(diag(confusionMat_ens_mrmr_70)) / sum(confusionMat_ens_mrmr_70(:));
accuracy_ens_mrmr_80 = sum(diag(confusionMat_ens_mrmr_80)) / sum(confusionMat_ens_mrmr_80(:));
accuracy_ens_mrmr_90 = sum(diag(confusionMat_ens_mrmr_90)) / sum(confusionMat_ens_mrmr_90(:));

recall_ens_mrmr_10 = diag(confusionMat_ens_mrmr_10) ./ sum(confusionMat_ens_mrmr_10, 2);
recall_ens_mrmr_20 = diag(confusionMat_ens_mrmr_20) ./ sum(confusionMat_ens_mrmr_20, 2);
recall_ens_mrmr_30 = diag(confusionMat_ens_mrmr_30) ./ sum(confusionMat_ens_mrmr_30, 2);
recall_ens_mrmr_40 = diag(confusionMat_ens_mrmr_40) ./ sum(confusionMat_ens_mrmr_40, 2);
recall_ens_mrmr_50 = diag(confusionMat_ens_mrmr_50) ./ sum(confusionMat_ens_mrmr_50, 2);
recall_ens_mrmr_60 = diag(confusionMat_ens_mrmr_60) ./ sum(confusionMat_ens_mrmr_60, 2);
recall_ens_mrmr_70 = diag(confusionMat_ens_mrmr_70) ./ sum(confusionMat_ens_mrmr_70, 2);
recall_ens_mrmr_80 = diag(confusionMat_ens_mrmr_80) ./ sum(confusionMat_ens_mrmr_80, 2);
recall_ens_mrmr_90 = diag(confusionMat_ens_mrmr_90) ./ sum(confusionMat_ens_mrmr_90, 2);

precision_ens_mrmr_10 = diag(confusionMat_ens_mrmr_10) ./ sum(confusionMat_ens_mrmr_10, 1)';
precision_ens_mrmr_20 = diag(confusionMat_ens_mrmr_20) ./ sum(confusionMat_ens_mrmr_20, 1)';
precision_ens_mrmr_30 = diag(confusionMat_ens_mrmr_30) ./ sum(confusionMat_ens_mrmr_30, 1)';
precision_ens_mrmr_40 = diag(confusionMat_ens_mrmr_40) ./ sum(confusionMat_ens_mrmr_40, 1)';
precision_ens_mrmr_50 = diag(confusionMat_ens_mrmr_50) ./ sum(confusionMat_ens_mrmr_50, 1)';
precision_ens_mrmr_60 = diag(confusionMat_ens_mrmr_60) ./ sum(confusionMat_ens_mrmr_60, 1)';
precision_ens_mrmr_70 = diag(confusionMat_ens_mrmr_70) ./ sum(confusionMat_ens_mrmr_70, 1)';
precision_ens_mrmr_80 = diag(confusionMat_ens_mrmr_80) ./ sum(confusionMat_ens_mrmr_80, 1)';
precision_ens_mrmr_90 = diag(confusionMat_ens_mrmr_90) ./ sum(confusionMat_ens_mrmr_90, 1)';

f1_score_ens_mrmr_10 = 2 * (precision_ens_mrmr_10 .* recall_ens_mrmr_10) ./ (precision_ens_mrmr_10 + recall_ens_mrmr_10); % F1 Score
f1_score_ens_mrmr_20 = 2 * (precision_ens_mrmr_20 .* recall_ens_mrmr_20) ./ (precision_ens_mrmr_20 + recall_ens_mrmr_20); % F1 Score
f1_score_ens_mrmr_30 = 2 * (precision_ens_mrmr_30 .* recall_ens_mrmr_30) ./ (precision_ens_mrmr_30 + recall_ens_mrmr_30); % F1 Score
f1_score_ens_mrmr_40 = 2 * (precision_ens_mrmr_40 .* recall_ens_mrmr_40) ./ (precision_ens_mrmr_40 + recall_ens_mrmr_40); % F1 Score
f1_score_ens_mrmr_50 = 2 * (precision_ens_mrmr_50 .* recall_ens_mrmr_50) ./ (precision_ens_mrmr_50 + recall_ens_mrmr_50); % F1 Score
f1_score_ens_mrmr_60 = 2 * (precision_ens_mrmr_60 .* recall_ens_mrmr_60) ./ (precision_ens_mrmr_60 + recall_ens_mrmr_60); % F1 Score
f1_score_ens_mrmr_70 = 2 * (precision_ens_mrmr_70 .* recall_ens_mrmr_70) ./ (precision_ens_mrmr_70 + recall_ens_mrmr_70); % F1 Score
f1_score_ens_mrmr_80 = 2 * (precision_ens_mrmr_80 .* recall_ens_mrmr_80) ./ (precision_ens_mrmr_80 + recall_ens_mrmr_80); % F1 Score
f1_score_ens_mrmr_90 = 2 * (precision_ens_mrmr_90 .* recall_ens_mrmr_90) ./ (precision_ens_mrmr_90 + recall_ens_mrmr_90); % F1 Score

%     % Exibir a acurácia e a matriz de confusão
% disp(['Acurácia_KNN_MRMR: ', num2str(accuracy_knn_mrmr)]);
% disp(['Accuracy_Naive Bayes_MRMR: ', num2str(accuracy_nb_mrmr)]);
% disp(['Accuracy_Neural Network_MRMR: ', num2str(accuracy_nn_mrmr)]);
% disp(['Accuracy_Ensemble_MRMR: ', num2str(accuracy_ens_mrmr)]);
% % disp('Matriz de Confusão_KNN_MRMR:');
% % disp(confusionMat_knn_mrmr);
% % disp('Matriz de Confusão_Naive Bayes_MRMR:');
% % disp(confusionMat_nb_mrmr);
% % disp('Matriz de Confusão_Neural Network_MRMR:');
% % disp(confusionMat_nn_mrmr);
% % disp('Matriz de Confusão_Ensemble_MRMR:');
% % disp(confusionMat_ens_mrmr);
% % % Plotar as matrizes de confusão
figure;
confusionchart(confusionMat_knn_mrmr_80,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - KNN - Top 80 Features Selected by MRMR');
figure;
confusionchart(confusionMat_nb_mrmr_50,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - Naive Bayes - Top 50 Features Selected by MRMR');
figure;
confusionchart(confusionMat_nn_mrmr_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - Neural Network - Top 20 Features Selected by MRMR');
figure;
confusionchart(confusionMat_ens_mrmr_40,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion Matrix - Ensemble - Top 40 Features Selected by MRMR');
figure Name 'Matrix Confusion - Features selected by MRMR - Features thresholds with best accuracy and F1-score'
subplot(2, 2, 1);
confusionchart(confusionMat_knn_mrmr_80,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('KNN - Top 80 Features Selected by MRMR');
subplot(2, 2, 2);
confusionchart(confusionMat_nb_mrmr_50,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Naive Bayes - Top 50 Features Selected by MRMR');
subplot(2, 2, 3);
confusionchart(confusionMat_nn_mrmr_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Neural Network - Top 20 Features Selected by MRMR');
subplot(2, 2, 4);
confusionchart(confusionMat_ens_mrmr_40,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Ensemble - Top 40 Features Selected by MRMR');
% % 
% Chi2
SF_train_chi2_10 = X_train(:, ranking3(1:10)); % dados de treino com melhores features
SF_test_chi2_10 = X_test(:, ranking3(1:10)); % dados de teste com melhores features
SF_train_chi2_20 = X_train(:, ranking3(1:20)); % dados de treino com melhores features
SF_test_chi2_20 = X_test(:, ranking3(1:20)); % dados de teste com melhores features
SF_train_chi2_30 = X_train(:, ranking3(1:30)); % dados de treino com melhores features
SF_test_chi2_30 = X_test(:, ranking3(1:30)); % dados de teste com melhores features
SF_train_chi2_40 = X_train(:, ranking3(1:40)); % dados de treino com melhores features
SF_test_chi2_40 = X_test(:, ranking3(1:40)); % dados de teste com melhores features
SF_train_chi2_50 = X_train(:, ranking3(1:50)); % dados de treino com melhores features
SF_test_chi2_50 = X_test(:, ranking3(1:50)); % dados de teste com melhores features
SF_train_chi2_60 = X_train(:, ranking3(1:60)); % dados de treino com melhores features
SF_test_chi2_60 = X_test(:, ranking3(1:60)); % dados de teste com melhores features
SF_train_chi2_70 = X_train(:, ranking3(1:70)); % dados de treino com melhores features
SF_test_chi2_70 = X_test(:, ranking3(1:70)); % dados de teste com melhores features
SF_train_chi2_80 = X_train(:, ranking3(1:80)); % dados de treino com melhores features
SF_test_chi2_80 = X_test(:, ranking3(1:80)); % dados de teste com melhores features
SF_train_chi2_90 = X_train(:, ranking3(1:90)); % dados de treino com melhores features
SF_test_chi2_90 = X_test(:, ranking3(1:90)); % dados de teste com melhores features

% Treinar o modelo novamente usando apenas os recursos selecionados
  modelknn_chi2_10 = fitcknn(SF_train_chi2_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_chi2_20 = fitcknn(SF_train_chi2_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_chi2_30 = fitcknn(SF_train_chi2_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_chi2_40 = fitcknn(SF_train_chi2_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_chi2_50 = fitcknn(SF_train_chi2_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_chi2_60 = fitcknn(SF_train_chi2_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_chi2_70 = fitcknn(SF_train_chi2_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_chi2_80 = fitcknn(SF_train_chi2_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnb_chi2_10 = fitcnb(SF_train_chi2_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_chi2_20 = fitcnb(SF_train_chi2_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_chi2_30 = fitcnb(SF_train_chi2_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_chi2_40 = fitcnb(SF_train_chi2_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_chi2_50 = fitcnb(SF_train_chi2_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_chi2_60 = fitcnb(SF_train_chi2_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_chi2_70 = fitcnb(SF_train_chi2_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_chi2_80 = fitcnb(SF_train_chi2_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnn_chi2_10 = fitcnet(SF_train_chi2_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_chi2_20 = fitcnet(SF_train_chi2_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_chi2_30 = fitcnet(SF_train_chi2_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_chi2_40 = fitcnet(SF_train_chi2_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_chi2_50 = fitcnet(SF_train_chi2_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_chi2_60 = fitcnet(SF_train_chi2_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_chi2_70 = fitcnet(SF_train_chi2_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_chi2_80 = fitcnet(SF_train_chi2_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelens_chi2_10 = fitcensemble(SF_train_chi2_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_chi2_20 = fitcensemble(SF_train_chi2_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_chi2_30 = fitcensemble(SF_train_chi2_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_chi2_40 = fitcensemble(SF_train_chi2_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_chi2_50 = fitcensemble(SF_train_chi2_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_chi2_60 = fitcensemble(SF_train_chi2_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_chi2_70 = fitcensemble(SF_train_chi2_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_chi2_80 = fitcensemble(SF_train_chi2_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));

% Testar o modelo novamente usando apenas os recursos selecionados
predictions_knn_chi2_10=predict(modelknn_chi2_10,SF_test_chi2_10);
predictions_knn_chi2_20=predict(modelknn_chi2_20,SF_test_chi2_20);
predictions_knn_chi2_30=predict(modelknn_chi2_30,SF_test_chi2_30);
predictions_knn_chi2_40=predict(modelknn_chi2_40,SF_test_chi2_40);
predictions_knn_chi2_50=predict(modelknn_chi2_50,SF_test_chi2_50);
predictions_knn_chi2_60=predict(modelknn_chi2_60,SF_test_chi2_60);
predictions_knn_chi2_70=predict(modelknn_chi2_70,SF_test_chi2_70);
predictions_knn_chi2_80=predict(modelknn_chi2_80,SF_test_chi2_80);

predictions_nb_chi2_10=predict(modelnb_chi2_10,SF_test_chi2_10);
predictions_nb_chi2_20=predict(modelnb_chi2_20,SF_test_chi2_20);
predictions_nb_chi2_30=predict(modelnb_chi2_30,SF_test_chi2_30);
predictions_nb_chi2_40=predict(modelnb_chi2_40,SF_test_chi2_40);
predictions_nb_chi2_50=predict(modelnb_chi2_50,SF_test_chi2_50);
predictions_nb_chi2_60=predict(modelnb_chi2_60,SF_test_chi2_60);
predictions_nb_chi2_70=predict(modelnb_chi2_70,SF_test_chi2_70);
predictions_nb_chi2_80=predict(modelnb_chi2_80,SF_test_chi2_80);

predictions_nn_chi2_10=predict(modelnn_chi2_10,SF_test_chi2_10);
predictions_nn_chi2_20=predict(modelnn_chi2_20,SF_test_chi2_20);
predictions_nn_chi2_30=predict(modelnn_chi2_30,SF_test_chi2_30);
predictions_nn_chi2_40=predict(modelnn_chi2_40,SF_test_chi2_40);
predictions_nn_chi2_50=predict(modelnn_chi2_50,SF_test_chi2_50);
predictions_nn_chi2_60=predict(modelnn_chi2_60,SF_test_chi2_60);
predictions_nn_chi2_70=predict(modelnn_chi2_70,SF_test_chi2_70);
predictions_nn_chi2_80=predict(modelnn_chi2_80,SF_test_chi2_80);

predictions_ens_chi2_10=predict(modelens_chi2_10,SF_test_chi2_10);
predictions_ens_chi2_20=predict(modelens_chi2_20,SF_test_chi2_20);
predictions_ens_chi2_30=predict(modelens_chi2_30,SF_test_chi2_30);
predictions_ens_chi2_40=predict(modelens_chi2_40,SF_test_chi2_40);
predictions_ens_chi2_50=predict(modelens_chi2_50,SF_test_chi2_50);
predictions_ens_chi2_60=predict(modelens_chi2_60,SF_test_chi2_60);
predictions_ens_chi2_70=predict(modelens_chi2_70,SF_test_chi2_70);
predictions_ens_chi2_80=predict(modelens_chi2_80,SF_test_chi2_80);

% Avaliar a performance dos modelos nos dados de teste com os features selecionados
% KNN
confusionMat_knn_chi2_10 = confusionmat(Ytest, predictions_knn_chi2_10);
confusionMat_knn_chi2_20 = confusionmat(Ytest, predictions_knn_chi2_20);
confusionMat_knn_chi2_30 = confusionmat(Ytest, predictions_knn_chi2_30);
confusionMat_knn_chi2_40 = confusionmat(Ytest, predictions_knn_chi2_40);
confusionMat_knn_chi2_50 = confusionmat(Ytest, predictions_knn_chi2_50);
confusionMat_knn_chi2_60 = confusionmat(Ytest, predictions_knn_chi2_60);
confusionMat_knn_chi2_70 = confusionmat(Ytest, predictions_knn_chi2_70);
confusionMat_knn_chi2_80 = confusionmat(Ytest, predictions_knn_chi2_80);
confusionMat_knn_chi2_90 = confusionmat(Ytest, predictions_knn_chi2_90);

accuracy_knn_chi2_10 = sum(diag(confusionMat_knn_chi2_10)) / sum(confusionMat_knn_chi2_10(:));
accuracy_knn_chi2_20 = sum(diag(confusionMat_knn_chi2_20)) / sum(confusionMat_knn_chi2_20(:));
accuracy_knn_chi2_30 = sum(diag(confusionMat_knn_chi2_30)) / sum(confusionMat_knn_chi2_30(:));
accuracy_knn_chi2_40 = sum(diag(confusionMat_knn_chi2_40)) / sum(confusionMat_knn_chi2_40(:));
accuracy_knn_chi2_50 = sum(diag(confusionMat_knn_chi2_50)) / sum(confusionMat_knn_chi2_50(:));
accuracy_knn_chi2_60 = sum(diag(confusionMat_knn_chi2_60)) / sum(confusionMat_knn_chi2_60(:));
accuracy_knn_chi2_70 = sum(diag(confusionMat_knn_chi2_70)) / sum(confusionMat_knn_chi2_70(:));
accuracy_knn_chi2_80 = sum(diag(confusionMat_knn_chi2_80)) / sum(confusionMat_knn_chi2_80(:));
accuracy_knn_chi2_90 = sum(diag(confusionMat_knn_chi2_90)) / sum(confusionMat_knn_chi2_90(:));

recall_knn_chi2_10 = diag(confusionMat_knn_chi2_10) ./ sum(confusionMat_knn_chi2_10, 2);
recall_knn_chi2_20 = diag(confusionMat_knn_chi2_20) ./ sum(confusionMat_knn_chi2_20, 2);
recall_knn_chi2_30 = diag(confusionMat_knn_chi2_30) ./ sum(confusionMat_knn_chi2_30, 2);
recall_knn_chi2_40 = diag(confusionMat_knn_chi2_40) ./ sum(confusionMat_knn_chi2_40, 2);
recall_knn_chi2_50 = diag(confusionMat_knn_chi2_50) ./ sum(confusionMat_knn_chi2_50, 2);
recall_knn_chi2_60 = diag(confusionMat_knn_chi2_60) ./ sum(confusionMat_knn_chi2_60, 2);
recall_knn_chi2_70 = diag(confusionMat_knn_chi2_70) ./ sum(confusionMat_knn_chi2_70, 2);
recall_knn_chi2_80 = diag(confusionMat_knn_chi2_80) ./ sum(confusionMat_knn_chi2_80, 2);
recall_knn_chi2_90 = diag(confusionMat_knn_chi2_90) ./ sum(confusionMat_knn_chi2_90, 2);

precision_knn_chi2_10 = diag(confusionMat_knn_chi2_10) ./ sum(confusionMat_knn_chi2_10, 1)';
precision_knn_chi2_20 = diag(confusionMat_knn_chi2_20) ./ sum(confusionMat_knn_chi2_20, 1)';
precision_knn_chi2_30 = diag(confusionMat_knn_chi2_30) ./ sum(confusionMat_knn_chi2_30, 1)';
precision_knn_chi2_40 = diag(confusionMat_knn_chi2_40) ./ sum(confusionMat_knn_chi2_40, 1)';
precision_knn_chi2_50 = diag(confusionMat_knn_chi2_50) ./ sum(confusionMat_knn_chi2_50, 1)';
precision_knn_chi2_60 = diag(confusionMat_knn_chi2_60) ./ sum(confusionMat_knn_chi2_60, 1)';
precision_knn_chi2_70 = diag(confusionMat_knn_chi2_70) ./ sum(confusionMat_knn_chi2_70, 1)';
precision_knn_chi2_80 = diag(confusionMat_knn_chi2_80) ./ sum(confusionMat_knn_chi2_80, 1)';
precision_knn_chi2_90 = diag(confusionMat_knn_chi2_90) ./ sum(confusionMat_knn_chi2_90, 1)';

f1_score_knn_chi2_10 = 2 * (precision_knn_chi2_10 .* recall_knn_chi2_10) ./ (precision_knn_chi2_10 + recall_knn_chi2_10); % F1 Score
f1_score_knn_chi2_20 = 2 * (precision_knn_chi2_20 .* recall_knn_chi2_20) ./ (precision_knn_chi2_20 + recall_knn_chi2_20); % F1 Score
f1_score_knn_chi2_30 = 2 * (precision_knn_chi2_30 .* recall_knn_chi2_30) ./ (precision_knn_chi2_30 + recall_knn_chi2_30); % F1 Score
f1_score_knn_chi2_40 = 2 * (precision_knn_chi2_40 .* recall_knn_chi2_40) ./ (precision_knn_chi2_40 + recall_knn_chi2_40); % F1 Score
f1_score_knn_chi2_50 = 2 * (precision_knn_chi2_50 .* recall_knn_chi2_50) ./ (precision_knn_chi2_50 + recall_knn_chi2_50); % F1 Score
f1_score_knn_chi2_60 = 2 * (precision_knn_chi2_60 .* recall_knn_chi2_60) ./ (precision_knn_chi2_60 + recall_knn_chi2_60); % F1 Score
f1_score_knn_chi2_70 = 2 * (precision_knn_chi2_70 .* recall_knn_chi2_70) ./ (precision_knn_chi2_70 + recall_knn_chi2_70); % F1 Score
f1_score_knn_chi2_80 = 2 * (precision_knn_chi2_80 .* recall_knn_chi2_80) ./ (precision_knn_chi2_80 + recall_knn_chi2_80); % F1 Score
f1_score_knn_chi2_90 = 2 * (precision_knn_chi2_90 .* recall_knn_chi2_90) ./ (precision_knn_chi2_90 + recall_knn_chi2_90); % F1 Score

% Naive Bayes
confusionMat_nb_chi2_10 = confusionmat(Ytest, predictions_nb_chi2_10);
confusionMat_nb_chi2_20 = confusionmat(Ytest, predictions_nb_chi2_20);
confusionMat_nb_chi2_30 = confusionmat(Ytest, predictions_nb_chi2_30);
confusionMat_nb_chi2_40 = confusionmat(Ytest, predictions_nb_chi2_40);
confusionMat_nb_chi2_50 = confusionmat(Ytest, predictions_nb_chi2_50);
confusionMat_nb_chi2_60 = confusionmat(Ytest, predictions_nb_chi2_60);
confusionMat_nb_chi2_70 = confusionmat(Ytest, predictions_nb_chi2_70);
confusionMat_nb_chi2_80 = confusionmat(Ytest, predictions_nb_chi2_80);
confusionMat_nb_chi2_90 = confusionmat(Ytest, predictions_nb_chi2_90);

accuracy_nb_chi2_10 = sum(diag(confusionMat_nb_chi2_10)) / sum(confusionMat_nb_chi2_10(:));
accuracy_nb_chi2_20 = sum(diag(confusionMat_nb_chi2_20)) / sum(confusionMat_nb_chi2_20(:));
accuracy_nb_chi2_30 = sum(diag(confusionMat_nb_chi2_30)) / sum(confusionMat_nb_chi2_30(:));
accuracy_nb_chi2_40 = sum(diag(confusionMat_nb_chi2_40)) / sum(confusionMat_nb_chi2_40(:));
accuracy_nb_chi2_50 = sum(diag(confusionMat_nb_chi2_50)) / sum(confusionMat_nb_chi2_50(:));
accuracy_nb_chi2_60 = sum(diag(confusionMat_nb_chi2_60)) / sum(confusionMat_nb_chi2_60(:));
accuracy_nb_chi2_70 = sum(diag(confusionMat_nb_chi2_70)) / sum(confusionMat_nb_chi2_70(:));
accuracy_nb_chi2_80 = sum(diag(confusionMat_nb_chi2_80)) / sum(confusionMat_nb_chi2_80(:));
accuracy_nb_chi2_90 = sum(diag(confusionMat_nb_chi2_90)) / sum(confusionMat_nb_chi2_90(:));

recall_nb_chi2_10 = diag(confusionMat_nb_chi2_10) ./ sum(confusionMat_nb_chi2_10, 2);
recall_nb_chi2_20 = diag(confusionMat_nb_chi2_20) ./ sum(confusionMat_nb_chi2_20, 2);
recall_nb_chi2_30 = diag(confusionMat_nb_chi2_30) ./ sum(confusionMat_nb_chi2_30, 2);
recall_nb_chi2_40 = diag(confusionMat_nb_chi2_40) ./ sum(confusionMat_nb_chi2_40, 2);
recall_nb_chi2_50 = diag(confusionMat_nb_chi2_50) ./ sum(confusionMat_nb_chi2_50, 2);
recall_nb_chi2_60 = diag(confusionMat_nb_chi2_60) ./ sum(confusionMat_nb_chi2_60, 2);
recall_nb_chi2_70 = diag(confusionMat_nb_chi2_70) ./ sum(confusionMat_nb_chi2_70, 2);
recall_nb_chi2_80 = diag(confusionMat_nb_chi2_80) ./ sum(confusionMat_nb_chi2_80, 2);
recall_nb_chi2_90 = diag(confusionMat_nb_chi2_90) ./ sum(confusionMat_nb_chi2_90, 2);

precision_nb_chi2_10 = diag(confusionMat_nb_chi2_10) ./ sum(confusionMat_nb_chi2_10, 1)';
precision_nb_chi2_20 = diag(confusionMat_nb_chi2_20) ./ sum(confusionMat_nb_chi2_20, 1)';
precision_nb_chi2_30 = diag(confusionMat_nb_chi2_30) ./ sum(confusionMat_nb_chi2_30, 1)';
precision_nb_chi2_40 = diag(confusionMat_nb_chi2_40) ./ sum(confusionMat_nb_chi2_40, 1)';
precision_nb_chi2_50 = diag(confusionMat_nb_chi2_50) ./ sum(confusionMat_nb_chi2_50, 1)';
precision_nb_chi2_60 = diag(confusionMat_nb_chi2_60) ./ sum(confusionMat_nb_chi2_60, 1)';
precision_nb_chi2_70 = diag(confusionMat_nb_chi2_70) ./ sum(confusionMat_nb_chi2_70, 1)';
precision_nb_chi2_80 = diag(confusionMat_nb_chi2_80) ./ sum(confusionMat_nb_chi2_80, 1)';
precision_nb_chi2_90 = diag(confusionMat_nb_chi2_90) ./ sum(confusionMat_nb_chi2_90, 1)';

f1_score_nb_chi2_10 = 2 * (precision_nb_chi2_10 .* recall_nb_chi2_10) ./ (precision_nb_chi2_10 + recall_nb_chi2_10); % F1 Score
f1_score_nb_chi2_20 = 2 * (precision_nb_chi2_20 .* recall_nb_chi2_20) ./ (precision_nb_chi2_20 + recall_nb_chi2_20); % F1 Score
f1_score_nb_chi2_30 = 2 * (precision_nb_chi2_30 .* recall_nb_chi2_30) ./ (precision_nb_chi2_30 + recall_nb_chi2_30); % F1 Score
f1_score_nb_chi2_40 = 2 * (precision_nb_chi2_40 .* recall_nb_chi2_40) ./ (precision_nb_chi2_40 + recall_nb_chi2_40); % F1 Score
f1_score_nb_chi2_50 = 2 * (precision_nb_chi2_50 .* recall_nb_chi2_50) ./ (precision_nb_chi2_50 + recall_nb_chi2_50); % F1 Score
f1_score_nb_chi2_60 = 2 * (precision_nb_chi2_60 .* recall_nb_chi2_60) ./ (precision_nb_chi2_60 + recall_nb_chi2_60); % F1 Score
f1_score_nb_chi2_70 = 2 * (precision_nb_chi2_70 .* recall_nb_chi2_70) ./ (precision_nb_chi2_70 + recall_nb_chi2_70); % F1 Score
f1_score_nb_chi2_80 = 2 * (precision_nb_chi2_80 .* recall_nb_chi2_80) ./ (precision_nb_chi2_80 + recall_nb_chi2_80); % F1 Score
f1_score_nb_chi2_90 = 2 * (precision_nb_chi2_90 .* recall_nb_chi2_90) ./ (precision_nb_chi2_90 + recall_nb_chi2_90); % F1 Score

% Neural Network
confusionMat_nn_chi2_10 = confusionmat(Ytest, predictions_nn_chi2_10);
confusionMat_nn_chi2_20 = confusionmat(Ytest, predictions_nn_chi2_20);
confusionMat_nn_chi2_30 = confusionmat(Ytest, predictions_nn_chi2_30);
confusionMat_nn_chi2_40 = confusionmat(Ytest, predictions_nn_chi2_40);
confusionMat_nn_chi2_50 = confusionmat(Ytest, predictions_nn_chi2_50);
confusionMat_nn_chi2_60 = confusionmat(Ytest, predictions_nn_chi2_60);
confusionMat_nn_chi2_70 = confusionmat(Ytest, predictions_nn_chi2_70);
confusionMat_nn_chi2_80 = confusionmat(Ytest, predictions_nn_chi2_80);

accuracy_nn_chi2_10 = sum(diag(confusionMat_nn_chi2_10)) / sum(confusionMat_nn_chi2_10(:));
accuracy_nn_chi2_20 = sum(diag(confusionMat_nn_chi2_20)) / sum(confusionMat_nn_chi2_20(:));
accuracy_nn_chi2_30 = sum(diag(confusionMat_nn_chi2_30)) / sum(confusionMat_nn_chi2_30(:));
accuracy_nn_chi2_40 = sum(diag(confusionMat_nn_chi2_40)) / sum(confusionMat_nn_chi2_40(:));
accuracy_nn_chi2_50 = sum(diag(confusionMat_nn_chi2_50)) / sum(confusionMat_nn_chi2_50(:));
accuracy_nn_chi2_60 = sum(diag(confusionMat_nn_chi2_60)) / sum(confusionMat_nn_chi2_60(:));
accuracy_nn_chi2_70 = sum(diag(confusionMat_nn_chi2_70)) / sum(confusionMat_nn_chi2_70(:));
accuracy_nn_chi2_80 = sum(diag(confusionMat_nn_chi2_80)) / sum(confusionMat_nn_chi2_80(:));

recall_nn_chi2_10 = diag(confusionMat_nn_chi2_10) ./ sum(confusionMat_nn_chi2_10, 2);
recall_nn_chi2_20 = diag(confusionMat_nn_chi2_20) ./ sum(confusionMat_nn_chi2_20, 2);
recall_nn_chi2_30 = diag(confusionMat_nn_chi2_30) ./ sum(confusionMat_nn_chi2_30, 2);
recall_nn_chi2_40 = diag(confusionMat_nn_chi2_40) ./ sum(confusionMat_nn_chi2_40, 2);
recall_nn_chi2_50 = diag(confusionMat_nn_chi2_50) ./ sum(confusionMat_nn_chi2_50, 2);
recall_nn_chi2_60 = diag(confusionMat_nn_chi2_60) ./ sum(confusionMat_nn_chi2_60, 2);
recall_nn_chi2_70 = diag(confusionMat_nn_chi2_70) ./ sum(confusionMat_nn_chi2_70, 2);
recall_nn_chi2_80 = diag(confusionMat_nn_chi2_80) ./ sum(confusionMat_nn_chi2_80, 2);

precision_nn_chi2_10 = diag(confusionMat_nn_chi2_10) ./ sum(confusionMat_nn_chi2_10, 1)';
precision_nn_chi2_20 = diag(confusionMat_nn_chi2_20) ./ sum(confusionMat_nn_chi2_20, 1)';
precision_nn_chi2_30 = diag(confusionMat_nn_chi2_30) ./ sum(confusionMat_nn_chi2_30, 1)';
precision_nn_chi2_40 = diag(confusionMat_nn_chi2_40) ./ sum(confusionMat_nn_chi2_40, 1)';
precision_nn_chi2_50 = diag(confusionMat_nn_chi2_50) ./ sum(confusionMat_nn_chi2_50, 1)';
precision_nn_chi2_60 = diag(confusionMat_nn_chi2_60) ./ sum(confusionMat_nn_chi2_60, 1)';
precision_nn_chi2_70 = diag(confusionMat_nn_chi2_70) ./ sum(confusionMat_nn_chi2_70, 1)';
precision_nn_chi2_80 = diag(confusionMat_nn_chi2_80) ./ sum(confusionMat_nn_chi2_80, 1)';

f1_score_nn_chi2_10 = 2 * (precision_nn_chi2_10 .* recall_nn_chi2_10) ./ (precision_nn_chi2_10 + recall_nn_chi2_10); % F1 Score
f1_score_nn_chi2_20 = 2 * (precision_nn_chi2_20 .* recall_nn_chi2_20) ./ (precision_nn_chi2_20 + recall_nn_chi2_20); % F1 Score
f1_score_nn_chi2_30 = 2 * (precision_nn_chi2_30 .* recall_nn_chi2_30) ./ (precision_nn_chi2_30 + recall_nn_chi2_30); % F1 Score
f1_score_nn_chi2_40 = 2 * (precision_nn_chi2_40 .* recall_nn_chi2_40) ./ (precision_nn_chi2_40 + recall_nn_chi2_40); % F1 Score
f1_score_nn_chi2_50 = 2 * (precision_nn_chi2_50 .* recall_nn_chi2_50) ./ (precision_nn_chi2_50 + recall_nn_chi2_50); % F1 Score
f1_score_nn_chi2_60 = 2 * (precision_nn_chi2_60 .* recall_nn_chi2_60) ./ (precision_nn_chi2_60 + recall_nn_chi2_60); % F1 Score
f1_score_nn_chi2_70 = 2 * (precision_nn_chi2_70 .* recall_nn_chi2_70) ./ (precision_nn_chi2_70 + recall_nn_chi2_70); % F1 Score
f1_score_nn_chi2_80 = 2 * (precision_nn_chi2_80 .* recall_nn_chi2_80) ./ (precision_nn_chi2_80 + recall_nn_chi2_80); % F1 Score

% Ensemble   
confusionMat_ens_chi2_10 = confusionmat(Ytest, predictions_ens_chi2_10);
confusionMat_ens_chi2_20 = confusionmat(Ytest, predictions_ens_chi2_20);
confusionMat_ens_chi2_30 = confusionmat(Ytest, predictions_ens_chi2_30);
confusionMat_ens_chi2_40 = confusionmat(Ytest, predictions_ens_chi2_40);
confusionMat_ens_chi2_50 = confusionmat(Ytest, predictions_ens_chi2_50);
confusionMat_ens_chi2_60 = confusionmat(Ytest, predictions_ens_chi2_60);
confusionMat_ens_chi2_70 = confusionmat(Ytest, predictions_ens_chi2_70);
confusionMat_ens_chi2_80 = confusionmat(Ytest, predictions_ens_chi2_80);

accuracy_ens_chi2_10 = sum(diag(confusionMat_ens_chi2_10)) / sum(confusionMat_ens_chi2_10(:));
accuracy_ens_chi2_20 = sum(diag(confusionMat_ens_chi2_20)) / sum(confusionMat_ens_chi2_20(:));
accuracy_ens_chi2_30 = sum(diag(confusionMat_ens_chi2_30)) / sum(confusionMat_ens_chi2_30(:));
accuracy_ens_chi2_40 = sum(diag(confusionMat_ens_chi2_40)) / sum(confusionMat_ens_chi2_40(:));
accuracy_ens_chi2_50 = sum(diag(confusionMat_ens_chi2_50)) / sum(confusionMat_ens_chi2_50(:));
accuracy_ens_chi2_60 = sum(diag(confusionMat_ens_chi2_60)) / sum(confusionMat_ens_chi2_60(:));
accuracy_ens_chi2_70 = sum(diag(confusionMat_ens_chi2_70)) / sum(confusionMat_ens_chi2_70(:));
accuracy_ens_chi2_80 = sum(diag(confusionMat_ens_chi2_80)) / sum(confusionMat_ens_chi2_80(:));

recall_ens_chi2_10 = diag(confusionMat_ens_chi2_10) ./ sum(confusionMat_ens_chi2_10, 2);
recall_ens_chi2_20 = diag(confusionMat_ens_chi2_20) ./ sum(confusionMat_ens_chi2_20, 2);
recall_ens_chi2_30 = diag(confusionMat_ens_chi2_30) ./ sum(confusionMat_ens_chi2_30, 2);
recall_ens_chi2_40 = diag(confusionMat_ens_chi2_40) ./ sum(confusionMat_ens_chi2_40, 2);
recall_ens_chi2_50 = diag(confusionMat_ens_chi2_50) ./ sum(confusionMat_ens_chi2_50, 2);
recall_ens_chi2_60 = diag(confusionMat_ens_chi2_60) ./ sum(confusionMat_ens_chi2_60, 2);
recall_ens_chi2_70 = diag(confusionMat_ens_chi2_70) ./ sum(confusionMat_ens_chi2_70, 2);
recall_ens_chi2_80 = diag(confusionMat_ens_chi2_80) ./ sum(confusionMat_ens_chi2_80, 2);

precision_ens_chi2_10 = diag(confusionMat_ens_chi2_10) ./ sum(confusionMat_ens_chi2_10, 1)';
precision_ens_chi2_20 = diag(confusionMat_ens_chi2_20) ./ sum(confusionMat_ens_chi2_20, 1)';
precision_ens_chi2_30 = diag(confusionMat_ens_chi2_30) ./ sum(confusionMat_ens_chi2_30, 1)';
precision_ens_chi2_40 = diag(confusionMat_ens_chi2_40) ./ sum(confusionMat_ens_chi2_40, 1)';
precision_ens_chi2_50 = diag(confusionMat_ens_chi2_50) ./ sum(confusionMat_ens_chi2_50, 1)';
precision_ens_chi2_60 = diag(confusionMat_ens_chi2_60) ./ sum(confusionMat_ens_chi2_60, 1)';
precision_ens_chi2_70 = diag(confusionMat_ens_chi2_70) ./ sum(confusionMat_ens_chi2_70, 1)';
precision_ens_chi2_80 = diag(confusionMat_ens_chi2_80) ./ sum(confusionMat_ens_chi2_80, 1)';

f1_score_ens_chi2_10 = 2 * (precision_ens_chi2_10 .* recall_ens_chi2_10) ./ (precision_ens_chi2_10 + recall_ens_chi2_10); % F1 Score
f1_score_ens_chi2_20 = 2 * (precision_ens_chi2_20 .* recall_ens_chi2_20) ./ (precision_ens_chi2_20 + recall_ens_chi2_20); % F1 Score
f1_score_ens_chi2_30 = 2 * (precision_ens_chi2_30 .* recall_ens_chi2_30) ./ (precision_ens_chi2_30 + recall_ens_chi2_30); % F1 Score
f1_score_ens_chi2_40 = 2 * (precision_ens_chi2_40 .* recall_ens_chi2_40) ./ (precision_ens_chi2_40 + recall_ens_chi2_40); % F1 Score
f1_score_ens_chi2_50 = 2 * (precision_ens_chi2_50 .* recall_ens_chi2_50) ./ (precision_ens_chi2_50 + recall_ens_chi2_50); % F1 Score
f1_score_ens_chi2_60 = 2 * (precision_ens_chi2_60 .* recall_ens_chi2_60) ./ (precision_ens_chi2_60 + recall_ens_chi2_60); % F1 Score
f1_score_ens_chi2_70 = 2 * (precision_ens_chi2_70 .* recall_ens_chi2_70) ./ (precision_ens_chi2_70 + recall_ens_chi2_70); % F1 Score
f1_score_ens_chi2_80 = 2 * (precision_ens_chi2_80 .* recall_ens_chi2_80) ./ (precision_ens_chi2_80 + recall_ens_chi2_80); % F1 Score

%     % Exibir a acurácia e a matriz de confusão
% disp(['Acurácia_KNN_Chi-Square: ', num2str(accuracy_knn_chi2)]);
% disp(['Accuracy_Naive Bayes_Chi-Square: ', num2str(accuracy_nb_chi2)]);
% disp(['Accuracy_Neural Network_Chi-Square: ', num2str(accuracy_nn_chi2)]);
% disp(['Accuracy_Ensemble_Chi-Square: ', num2str(accuracy_ens_chi2)]);
% % disp('Matriz de Confusão_Chi-Square:');
% % disp(confusionMat_knn_chi2);
% % disp('Matriz de Confusão_Naive Bayes_Chi-Square:');
% % disp(confusionMat_nb_chi2);
% % disp('Matriz de Confusão_Neural Network_Chi-Square:');
% % disp(confusionMat_nn_chi2);
% % disp('Matriz de Confusão_Ensemble_Chi-Square:');
% % disp(confusionMat_ens_chi2);
% % % Plotar as matrizes de confusão
% % figure;
% % confusionchart(confusionMat_knn_chi2,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% % title('Confusion Matrix - KNN - Features Selected by Chi-Square');
% % figure;
% % confusionchart(confusionMat_nb_chi2,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% % title('Confusion Matrix - Naive Bayes - Features Selected by Chi-Square');
% % figure;
% % confusionchart(confusionMat_nn_chi2,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% % title('Confusion Matrix - Neural Network - Features Selected by Chi-Square');
% % figure;
% % confusionchart(confusionMat_ens_chi2,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% % title('Confusion Matrix - Ensemble - Features Selected by Chi-Square');
figure Name 'Matrix Confusion - Features selected by Chi-Square - Features thresholds with best accuracy and F1-score'
subplot(2, 2, 1);
confusionchart(confusionMat_knn_chi2_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('KNN - Top 20 Features Selected by Chi-Square');
subplot(2, 2, 2);
confusionchart(confusionMat_nb_chi2_50,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Naive Bayes - Top 50 Features Selected by Chi-Square');
subplot(2, 2, 3);
confusionchart(confusionMat_nn_chi2_60,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Neural Network - Top 60 Features Selected by Chi-Square');
subplot(2, 2, 4);
confusionchart(confusionMat_ens_chi2_40,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Ensemble - Top 40 Features Selected by Chi-Square');
% % 
% 
% 
% 


% Nova classificação dos dados com os features selecionados por Adaboost M2 (método de seleção de atributos Embedded)
n_ens = 90;
ens_train_10 = X_train(:, featureRanking(1:10)); % dados de treino com melhores features
ens_test_10 = X_test(:, featureRanking(1:10)); % dados de teste com melhores features
ens_train_20 = X_train(:, featureRanking(1:20)); % dados de treino com melhores features
ens_test_20 = X_test(:, featureRanking(1:20)); % dados de teste com melhores features
ens_train_30 = X_train(:, featureRanking(1:30)); % dados de treino com melhores features
ens_test_30 = X_test(:, featureRanking(1:30)); % dados de teste com melhores features
ens_train_40 = X_train(:, featureRanking(1:40)); % dados de treino com melhores features
ens_test_40 = X_test(:, featureRanking(1:40)); % dados de teste com melhores features
ens_train_50 = X_train(:, featureRanking(1:50)); % dados de treino com melhores features
ens_test_50 = X_test(:, featureRanking(1:50)); % dados de teste com melhores features
ens_train_60 = X_train(:, featureRanking(1:60)); % dados de treino com melhores features
ens_test_60 = X_test(:, featureRanking(1:60)); % dados de teste com melhores features
ens_train_70 = X_train(:, featureRanking(1:70)); % dados de treino com melhores features
ens_test_70 = X_test(:, featureRanking(1:70)); % dados de teste com melhores features
ens_train_80 = X_train(:, featureRanking(1:80)); % dados de treino com melhores features
ens_test_80 = X_test(:, featureRanking(1:80)); % dados de teste com melhores features


% % Treinar o modelo novamente usando apenas os recursos selecionados

  modelens_rank_10 = fitcensemble(ens_train_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_rank_20 = fitcensemble(ens_train_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_rank_30 = fitcensemble(ens_train_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_rank_40 = fitcensemble(ens_train_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_rank_50 = fitcensemble(ens_train_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_rank_60 = fitcensemble(ens_train_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_rank_70 = fitcensemble(ens_train_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_rank_80 = fitcensemble(ens_train_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));


% Testar o modelo novamente usando apenas os recursos selecionados

predictions_ens_rank_10=predict(modelens_rank_10,ens_test_10);
predictions_ens_rank_20=predict(modelens_rank_20,ens_test_20);
predictions_ens_rank_30=predict(modelens_rank_30,ens_test_30);
predictions_ens_rank_40=predict(modelens_rank_40,ens_test_40);
predictions_ens_rank_50=predict(modelens_rank_50,ens_test_50);
predictions_ens_rank_60=predict(modelens_rank_60,ens_test_60);
predictions_ens_rank_70=predict(modelens_rank_70,ens_test_70);
predictions_ens_rank_80=predict(modelens_rank_80,ens_test_80);


% Avaliar a performance dos modelos nos dados de teste com os features selecionados

% Ensemble   
confusionMat_ens_rank_10 = confusionmat(Ytest, predictions_ens_rank_10);
confusionMat_ens_rank_20 = confusionmat(Ytest, predictions_ens_rank_20);
confusionMat_ens_rank_30 = confusionmat(Ytest, predictions_ens_rank_30);
confusionMat_ens_rank_40 = confusionmat(Ytest, predictions_ens_rank_40);
confusionMat_ens_rank_50 = confusionmat(Ytest, predictions_ens_rank_50);
confusionMat_ens_rank_60 = confusionmat(Ytest, predictions_ens_rank_60);
confusionMat_ens_rank_70 = confusionmat(Ytest, predictions_ens_rank_70);
confusionMat_ens_rank_80 = confusionmat(Ytest, predictions_ens_rank_80);


%
accuracy_ens_rank_10 = sum(diag(confusionMat_ens_rank_10)) / sum(confusionMat_ens_rank_10(:));
accuracy_ens_rank_20 = sum(diag(confusionMat_ens_rank_20)) / sum(confusionMat_ens_rank_20(:));
accuracy_ens_rank_30 = sum(diag(confusionMat_ens_rank_30)) / sum(confusionMat_ens_rank_30(:));
accuracy_ens_rank_40 = sum(diag(confusionMat_ens_rank_40)) / sum(confusionMat_ens_rank_40(:));
accuracy_ens_rank_50 = sum(diag(confusionMat_ens_rank_50)) / sum(confusionMat_ens_rank_50(:));
accuracy_ens_rank_60 = sum(diag(confusionMat_ens_rank_60)) / sum(confusionMat_ens_rank_60(:));
accuracy_ens_rank_70 = sum(diag(confusionMat_ens_rank_70)) / sum(confusionMat_ens_rank_70(:));
accuracy_ens_rank_80 = sum(diag(confusionMat_ens_rank_80)) / sum(confusionMat_ens_rank_80(:));


recall_ens_rank_10 = diag(confusionMat_ens_rank_10) ./ sum(confusionMat_ens_rank_10, 2);
recall_ens_rank_20 = diag(confusionMat_ens_rank_20) ./ sum(confusionMat_ens_rank_20, 2);
recall_ens_rank_30 = diag(confusionMat_ens_rank_30) ./ sum(confusionMat_ens_rank_30, 2);
recall_ens_rank_40 = diag(confusionMat_ens_rank_40) ./ sum(confusionMat_ens_rank_40, 2);
recall_ens_rank_50 = diag(confusionMat_ens_rank_50) ./ sum(confusionMat_ens_rank_50, 2);
recall_ens_rank_60 = diag(confusionMat_ens_rank_60) ./ sum(confusionMat_ens_rank_60, 2);
recall_ens_rank_70 = diag(confusionMat_ens_rank_70) ./ sum(confusionMat_ens_rank_70, 2);
recall_ens_rank_80 = diag(confusionMat_ens_rank_80) ./ sum(confusionMat_ens_rank_80, 2);

% 
precision_ens_rank_10 = diag(confusionMat_ens_rank_10) ./ sum(confusionMat_ens_rank_10, 1)';
precision_ens_rank_20 = diag(confusionMat_ens_rank_20) ./ sum(confusionMat_ens_rank_20, 1)';
precision_ens_rank_30 = diag(confusionMat_ens_rank_30) ./ sum(confusionMat_ens_rank_30, 1)';
precision_ens_rank_40 = diag(confusionMat_ens_rank_40) ./ sum(confusionMat_ens_rank_40, 1)';
precision_ens_rank_50 = diag(confusionMat_ens_rank_50) ./ sum(confusionMat_ens_rank_50, 1)';
precision_ens_rank_60 = diag(confusionMat_ens_rank_60) ./ sum(confusionMat_ens_rank_60, 1)';
precision_ens_rank_70 = diag(confusionMat_ens_rank_70) ./ sum(confusionMat_ens_rank_70, 1)';
precision_ens_rank_80 = diag(confusionMat_ens_rank_80) ./ sum(confusionMat_ens_rank_80, 1)';

%
f1_score_ens_rank_10 = 2 * (precision_ens_rank_10 .* recall_ens_rank_10) ./ (precision_ens_rank_10 + recall_ens_rank_10); % F1 Score
f1_score_ens_rank_20 = 2 * (precision_ens_rank_20 .* recall_ens_rank_20) ./ (precision_ens_rank_20 + recall_ens_rank_20); % F1 Score
f1_score_ens_rank_30 = 2 * (precision_ens_rank_30 .* recall_ens_rank_30) ./ (precision_ens_rank_30 + recall_ens_rank_30); % F1 Score
f1_score_ens_rank_40 = 2 * (precision_ens_rank_40 .* recall_ens_rank_40) ./ (precision_ens_rank_40 + recall_ens_rank_40); % F1 Score
f1_score_ens_rank_50 = 2 * (precision_ens_rank_50 .* recall_ens_rank_50) ./ (precision_ens_rank_50 + recall_ens_rank_50); % F1 Score
f1_score_ens_rank_60 = 2 * (precision_ens_rank_60 .* recall_ens_rank_60) ./ (precision_ens_rank_60 + recall_ens_rank_60); % F1 Score
f1_score_ens_rank_70 = 2 * (precision_ens_rank_70 .* recall_ens_rank_70) ./ (precision_ens_rank_70 + recall_ens_rank_70); % F1 Score
f1_score_ens_rank_80 = 2 * (precision_ens_rank_80 .* recall_ens_rank_80) ./ (precision_ens_rank_80 + recall_ens_rank_80); % F1 Score

% Plote Matriz
figure Name 'Matrix Confusion - Features selected by Embedded Method - Features thresholds with best accuracy and F1-score'
confusionchart(confusionMat_ens_rank_40,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Ensemble - Top 40 Features Selected by Embedded Method');



% Nova classificação dos dados com os features selecionados pelo método Shap
% Ranqueando os preditores mais importantes por shap (conditional), para cada classificador
% KNN
% n_SHAP = 90;
MeanSHAP_knn = explainer_knn.MeanAbsoluteShapley.collision + explainer_knn.MeanAbsoluteShapley.normal + explainer_knn.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_knn] = sort(MeanSHAP_knn, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_train_knn_10 = X_train(:, sorted_indices_knn(1:10)); % dados de treino com melhores features
SHAP_test_knn_10 = X_test(:, sorted_indices_knn(1:10)); % dados de teste com melhores features
SHAP_train_knn_20 = X_train(:, sorted_indices_knn(1:20)); % dados de treino com melhores features
SHAP_test_knn_20 = X_test(:, sorted_indices_knn(1:20)); % dados de teste com melhores features
SHAP_train_knn_30 = X_train(:, sorted_indices_knn(1:30)); % dados de treino com melhores features
SHAP_test_knn_30 = X_test(:, sorted_indices_knn(1:30)); % dados de teste com melhores features
SHAP_train_knn_40 = X_train(:, sorted_indices_knn(1:40)); % dados de treino com melhores features
SHAP_test_knn_40 = X_test(:, sorted_indices_knn(1:40)); % dados de teste com melhores features
SHAP_train_knn_50 = X_train(:, sorted_indices_knn(1:50)); % dados de treino com melhores features
SHAP_test_knn_50 = X_test(:, sorted_indices_knn(1:50)); % dados de teste com melhores features
SHAP_train_knn_60 = X_train(:, sorted_indices_knn(1:60)); % dados de treino com melhores features
SHAP_test_knn_60 = X_test(:, sorted_indices_knn(1:60)); % dados de teste com melhores features
SHAP_train_knn_70 = X_train(:, sorted_indices_knn(1:70)); % dados de treino com melhores features
SHAP_test_knn_70 = X_test(:, sorted_indices_knn(1:70)); % dados de teste com melhores features
SHAP_train_knn_80 = X_train(:, sorted_indices_knn(1:80)); % dados de treino com melhores features
SHAP_test_knn_80 = X_test(:, sorted_indices_knn(1:80)); % dados de teste com melhores features
SHAP_train_knn_90 = X_train(:, sorted_indices_knn(1:90)); % dados de treino com melhores features
SHAP_test_knn_90 = X_test(:, sorted_indices_knn(1:90)); % dados de teste com melhores features
% Naive Bayes
MeanSHAP_nb = explainer_nb.MeanAbsoluteShapley.collision + explainer_nb.MeanAbsoluteShapley.normal + explainer_nb.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_nb] = sort(MeanSHAP_nb, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_train_nb_10 = X_train(:, sorted_indices_nb(1:10)); % dados de treino com melhores features
SHAP_test_nb_10 = X_test(:, sorted_indices_nb(1:10)); % dados de teste com melhores features
SHAP_train_nb_20 = X_train(:, sorted_indices_nb(1:20)); % dados de treino com melhores features
SHAP_test_nb_20 = X_test(:, sorted_indices_nb(1:20)); % dados de teste com melhores features
SHAP_train_nb_30 = X_train(:, sorted_indices_nb(1:30)); % dados de treino com melhores features
SHAP_test_nb_30 = X_test(:, sorted_indices_nb(1:30)); % dados de teste com melhores features
SHAP_train_nb_40 = X_train(:, sorted_indices_nb(1:40)); % dados de treino com melhores features
SHAP_test_nb_40 = X_test(:, sorted_indices_nb(1:40)); % dados de teste com melhores features
SHAP_train_nb_50 = X_train(:, sorted_indices_nb(1:50)); % dados de treino com melhores features
SHAP_test_nb_50 = X_test(:, sorted_indices_nb(1:50)); % dados de teste com melhores features
SHAP_train_nb_60 = X_train(:, sorted_indices_nb(1:60)); % dados de treino com melhores features
SHAP_test_nb_60 = X_test(:, sorted_indices_nb(1:60)); % dados de teste com melhores features
SHAP_train_nb_70 = X_train(:, sorted_indices_nb(1:70)); % dados de treino com melhores features
SHAP_test_nb_70 = X_test(:, sorted_indices_nb(1:70)); % dados de teste com melhores features
SHAP_train_nb_80 = X_train(:, sorted_indices_nb(1:80)); % dados de treino com melhores features
SHAP_test_nb_80 = X_test(:, sorted_indices_nb(1:80)); % dados de teste com melhores features
SHAP_train_nb_90 = X_train(:, sorted_indices_nb(1:90)); % dados de treino com melhores features
SHAP_test_nb_90 = X_test(:, sorted_indices_nb(1:90)); % dados de teste com melhores features
% Neural Network
MeanSHAP_nn = explainer_nn.MeanAbsoluteShapley.collision + explainer_nn.MeanAbsoluteShapley.normal + explainer_nn.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_nn] = sort(MeanSHAP_nn, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_train_nn_10 = X_train(:, sorted_indices_nn(1:10)); % dados de treino com melhores features
SHAP_test_nn_10 = X_test(:, sorted_indices_nn(1:10)); % dados de teste com melhores features
SHAP_train_nn_20 = X_train(:, sorted_indices_nn(1:20)); % dados de treino com melhores features
SHAP_test_nn_20 = X_test(:, sorted_indices_nn(1:20)); % dados de teste com melhores features
SHAP_train_nn_30 = X_train(:, sorted_indices_nn(1:30)); % dados de treino com melhores features
SHAP_test_nn_30 = X_test(:, sorted_indices_nn(1:30)); % dados de teste com melhores features
SHAP_train_nn_40 = X_train(:, sorted_indices_nn(1:40)); % dados de treino com melhores features
SHAP_test_nn_40 = X_test(:, sorted_indices_nn(1:40)); % dados de teste com melhores features
SHAP_train_nn_50 = X_train(:, sorted_indices_nn(1:50)); % dados de treino com melhores features
SHAP_test_nn_50 = X_test(:, sorted_indices_nn(1:50)); % dados de teste com melhores features
SHAP_train_nn_60 = X_train(:, sorted_indices_nn(1:60)); % dados de treino com melhores features
SHAP_test_nn_60 = X_test(:, sorted_indices_nn(1:60)); % dados de teste com melhores features
SHAP_train_nn_70 = X_train(:, sorted_indices_nn(1:70)); % dados de treino com melhores features
SHAP_test_nn_70 = X_test(:, sorted_indices_nn(1:70)); % dados de teste com melhores features
SHAP_train_nn_80 = X_train(:, sorted_indices_nn(1:80)); % dados de treino com melhores features
SHAP_test_nn_80 = X_test(:, sorted_indices_nn(1:80)); % dados de teste com melhores features
SHAP_train_nn_90 = X_train(:, sorted_indices_nn(1:90)); % dados de treino com melhores features
SHAP_test_nn_90 = X_test(:, sorted_indices_nn(1:90)); % dados de teste com melhores features
% Ensemble
MeanSHAP_ens = explainer_ens.MeanAbsoluteShapley.collision + explainer_ens.MeanAbsoluteShapley.normal + explainer_ens.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_ens] = sort(MeanSHAP_ens, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_train_ens_10 = X_train(:, sorted_indices_ens(1:10)); % dados de treino com melhores features
SHAP_test_ens_10 = X_test(:, sorted_indices_ens(1:10)); % dados de teste com melhores features
SHAP_train_ens_20 = X_train(:, sorted_indices_ens(1:20)); % dados de treino com melhores features
SHAP_test_ens_20 = X_test(:, sorted_indices_ens(1:20)); % dados de teste com melhores features
SHAP_train_ens_30 = X_train(:, sorted_indices_ens(1:30)); % dados de treino com melhores features
SHAP_test_ens_30 = X_test(:, sorted_indices_ens(1:30)); % dados de teste com melhores features
SHAP_train_ens_40 = X_train(:, sorted_indices_ens(1:40)); % dados de treino com melhores features
SHAP_test_ens_40 = X_test(:, sorted_indices_ens(1:40)); % dados de teste com melhores features
SHAP_train_ens_50 = X_train(:, sorted_indices_ens(1:50)); % dados de treino com melhores features
SHAP_test_ens_50 = X_test(:, sorted_indices_ens(1:50)); % dados de teste com melhores features
SHAP_train_ens_60 = X_train(:, sorted_indices_ens(1:60)); % dados de treino com melhores features
SHAP_test_ens_60 = X_test(:, sorted_indices_ens(1:60)); % dados de teste com melhores features
SHAP_train_ens_70 = X_train(:, sorted_indices_ens(1:70)); % dados de treino com melhores features
SHAP_test_ens_70 = X_test(:, sorted_indices_ens(1:70)); % dados de teste com melhores features
SHAP_train_ens_80 = X_train(:, sorted_indices_ens(1:80)); % dados de treino com melhores features
SHAP_test_ens_80 = X_test(:, sorted_indices_ens(1:80)); % dados de teste com melhores features
SHAP_train_ens_90 = X_train(:, sorted_indices_ens(1:90)); % dados de treino com melhores features
SHAP_test_ens_90 = X_test(:, sorted_indices_ens(1:90)); % dados de teste com melhores features

% % Treinar o modelo novamente usando apenas os recursos selecionados
  modelknn_SHAP_10 = fitcknn(SHAP_train_knn_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_20 = fitcknn(SHAP_train_knn_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_30 = fitcknn(SHAP_train_knn_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_40 = fitcknn(SHAP_train_knn_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_50 = fitcknn(SHAP_train_knn_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_60 = fitcknn(SHAP_train_knn_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_70 = fitcknn(SHAP_train_knn_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_80 = fitcknn(SHAP_train_knn_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnb_SHAP_10 = fitcnb(SHAP_train_nb_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_20 = fitcnb(SHAP_train_nb_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_30 = fitcnb(SHAP_train_nb_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_40 = fitcnb(SHAP_train_nb_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_50 = fitcnb(SHAP_train_nb_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_60 = fitcnb(SHAP_train_nb_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_70 = fitcnb(SHAP_train_nb_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_80 = fitcnb(SHAP_train_nb_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnn_SHAP_10 = fitcnet(SHAP_train_nn_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_20 = fitcnet(SHAP_train_nn_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_30 = fitcnet(SHAP_train_nn_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_40 = fitcnet(SHAP_train_nn_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_50 = fitcnet(SHAP_train_nn_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_60 = fitcnet(SHAP_train_nn_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_70 = fitcnet(SHAP_train_nn_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_80 = fitcnet(SHAP_train_nn_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelens_SHAP_10 = fitcensemble(SHAP_train_ens_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_20 = fitcensemble(SHAP_train_ens_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_30 = fitcensemble(SHAP_train_ens_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_40 = fitcensemble(SHAP_train_ens_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_50 = fitcensemble(SHAP_train_ens_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_60 = fitcensemble(SHAP_train_ens_60,Y_train,'Method','AdaBoostM2','NumLearningCycles',178,'LearnRate',0.92123,'Learner',templateTree('MinLeaf', 4,'MaxNumSplits',25,'SplitCriterion','twoing'));
  modelens_SHAP_70 = fitcensemble(SHAP_train_ens_70,Y_train,'Method','AdaBoostM2','NumLearningCycles',178,'LearnRate',0.95123,'Learner',templateTree('MinLeaf', 3,'MaxNumSplits',32,'SplitCriterion','twoing'));
  modelens_SHAP_80 = fitcensemble(SHAP_train_ens_80,Y_train,'Method','AdaBoostM2','NumLearningCycles',178,'LearnRate',0.95123,'Learner',templateTree('MinLeaf', 3,'MaxNumSplits',32,'SplitCriterion','twoing'));


% Testar o modelo novamente usando apenas os recursos selecionados
predictions_knn_SHAP_10=predict(modelknn_SHAP_10,SHAP_test_knn_10);
predictions_knn_SHAP_20=predict(modelknn_SHAP_20,SHAP_test_knn_20);
predictions_knn_SHAP_30=predict(modelknn_SHAP_30,SHAP_test_knn_30);
predictions_knn_SHAP_40=predict(modelknn_SHAP_40,SHAP_test_knn_40);
predictions_knn_SHAP_50=predict(modelknn_SHAP_50,SHAP_test_knn_50);
predictions_knn_SHAP_60=predict(modelknn_SHAP_60,SHAP_test_knn_60);
predictions_knn_SHAP_70=predict(modelknn_SHAP_70,SHAP_test_knn_70);
predictions_knn_SHAP_80=predict(modelknn_SHAP_80,SHAP_test_knn_80);
%
predictions_nb_SHAP_10=predict(modelnb_SHAP_10,SHAP_test_nb_10);
predictions_nb_SHAP_20=predict(modelnb_SHAP_20,SHAP_test_nb_20);
predictions_nb_SHAP_30=predict(modelnb_SHAP_30,SHAP_test_nb_30);
predictions_nb_SHAP_40=predict(modelnb_SHAP_40,SHAP_test_nb_40);
predictions_nb_SHAP_50=predict(modelnb_SHAP_50,SHAP_test_nb_50);
predictions_nb_SHAP_60=predict(modelnb_SHAP_60,SHAP_test_nb_60);
predictions_nb_SHAP_70=predict(modelnb_SHAP_70,SHAP_test_nb_70);
predictions_nb_SHAP_80=predict(modelnb_SHAP_80,SHAP_test_nb_80);
%
predictions_nn_SHAP_10=predict(modelnn_SHAP_10,SHAP_test_nn_10);
predictions_nn_SHAP_20=predict(modelnn_SHAP_20,SHAP_test_nn_20);
predictions_nn_SHAP_30=predict(modelnn_SHAP_30,SHAP_test_nn_30);
predictions_nn_SHAP_40=predict(modelnn_SHAP_40,SHAP_test_nn_40);
predictions_nn_SHAP_50=predict(modelnn_SHAP_50,SHAP_test_nn_50);
predictions_nn_SHAP_60=predict(modelnn_SHAP_60,SHAP_test_nn_60);
predictions_nn_SHAP_70=predict(modelnn_SHAP_70,SHAP_test_nn_70);
predictions_nn_SHAP_80=predict(modelnn_SHAP_80,SHAP_test_nn_80);

predictions_ens_SHAP_10=predict(modelens_SHAP_10,SHAP_test_ens_10);
predictions_ens_SHAP_20=predict(modelens_SHAP_20,SHAP_test_ens_20);
predictions_ens_SHAP_30=predict(modelens_SHAP_30,SHAP_test_ens_30);
predictions_ens_SHAP_40=predict(modelens_SHAP_40,SHAP_test_ens_40);
predictions_ens_SHAP_50=predict(modelens_SHAP_50,SHAP_test_ens_50);
predictions_ens_SHAP_60=predict(modelens_SHAP_60,SHAP_test_ens_60);
predictions_ens_SHAP_70=predict(modelens_SHAP_70,SHAP_test_ens_70);
predictions_ens_SHAP_80=predict(modelens_SHAP_80,SHAP_test_ens_80);

Avaliar a performance dos modelos nos dados de teste com os features selecionados
KNN
confusionMat_knn_SHAP_10 = confusionmat(Ytest, predictions_knn_SHAP_10);
confusionMat_knn_SHAP_20 = confusionmat(Ytest, predictions_knn_SHAP_20);
confusionMat_knn_SHAP_30 = confusionmat(Ytest, predictions_knn_SHAP_30);
confusionMat_knn_SHAP_40 = confusionmat(Ytest, predictions_knn_SHAP_40);
confusionMat_knn_SHAP_50 = confusionmat(Ytest, predictions_knn_SHAP_50);
confusionMat_knn_SHAP_60 = confusionmat(Ytest, predictions_knn_SHAP_60);
confusionMat_knn_SHAP_70 = confusionmat(Ytest, predictions_knn_SHAP_70);
confusionMat_knn_SHAP_80 = confusionmat(Ytest, predictions_knn_SHAP_80);
confusionMat_knn_SHAP_90 = confusionmat(Ytest, predictions_knn_SHAP_90);
%
accuracy_knn_SHAP_10 = sum(diag(confusionMat_knn_SHAP_10)) / sum(confusionMat_knn_SHAP_10(:));
accuracy_knn_SHAP_20 = sum(diag(confusionMat_knn_SHAP_20)) / sum(confusionMat_knn_SHAP_20(:));
accuracy_knn_SHAP_30 = sum(diag(confusionMat_knn_SHAP_30)) / sum(confusionMat_knn_SHAP_30(:));
accuracy_knn_SHAP_40 = sum(diag(confusionMat_knn_SHAP_40)) / sum(confusionMat_knn_SHAP_40(:));
accuracy_knn_SHAP_50 = sum(diag(confusionMat_knn_SHAP_50)) / sum(confusionMat_knn_SHAP_50(:));
accuracy_knn_SHAP_60 = sum(diag(confusionMat_knn_SHAP_60)) / sum(confusionMat_knn_SHAP_60(:));
accuracy_knn_SHAP_70 = sum(diag(confusionMat_knn_SHAP_70)) / sum(confusionMat_knn_SHAP_70(:));
accuracy_knn_SHAP_80 = sum(diag(confusionMat_knn_SHAP_80)) / sum(confusionMat_knn_SHAP_80(:));
accuracy_knn_SHAP_90 = sum(diag(confusionMat_knn_SHAP_90)) / sum(confusionMat_knn_SHAP_90(:));
%
recall_knn_SHAP_10 = diag(confusionMat_knn_SHAP_10) ./ sum(confusionMat_knn_SHAP_10, 2);
recall_knn_SHAP_20 = diag(confusionMat_knn_SHAP_20) ./ sum(confusionMat_knn_SHAP_20, 2);
recall_knn_SHAP_30 = diag(confusionMat_knn_SHAP_30) ./ sum(confusionMat_knn_SHAP_30, 2);
recall_knn_SHAP_40 = diag(confusionMat_knn_SHAP_40) ./ sum(confusionMat_knn_SHAP_40, 2);
recall_knn_SHAP_50 = diag(confusionMat_knn_SHAP_50) ./ sum(confusionMat_knn_SHAP_50, 2);
recall_knn_SHAP_60 = diag(confusionMat_knn_SHAP_60) ./ sum(confusionMat_knn_SHAP_60, 2);
recall_knn_SHAP_70 = diag(confusionMat_knn_SHAP_70) ./ sum(confusionMat_knn_SHAP_70, 2);
recall_knn_SHAP_80 = diag(confusionMat_knn_SHAP_80) ./ sum(confusionMat_knn_SHAP_80, 2);
recall_knn_SHAP_90 = diag(confusionMat_knn_SHAP_90) ./ sum(confusionMat_knn_SHAP_90, 2);
%
precision_knn_SHAP_10 = diag(confusionMat_knn_SHAP_10) ./ sum(confusionMat_knn_SHAP_10, 1)';
precision_knn_SHAP_20 = diag(confusionMat_knn_SHAP_20) ./ sum(confusionMat_knn_SHAP_20, 1)';
precision_knn_SHAP_30 = diag(confusionMat_knn_SHAP_30) ./ sum(confusionMat_knn_SHAP_30, 1)';
precision_knn_SHAP_40 = diag(confusionMat_knn_SHAP_40) ./ sum(confusionMat_knn_SHAP_40, 1)';
precision_knn_SHAP_50 = diag(confusionMat_knn_SHAP_50) ./ sum(confusionMat_knn_SHAP_50, 1)';
precision_knn_SHAP_60 = diag(confusionMat_knn_SHAP_60) ./ sum(confusionMat_knn_SHAP_60, 1)';
precision_knn_SHAP_70 = diag(confusionMat_knn_SHAP_70) ./ sum(confusionMat_knn_SHAP_70, 1)';
precision_knn_SHAP_80 = diag(confusionMat_knn_SHAP_80) ./ sum(confusionMat_knn_SHAP_80, 1)';
precision_knn_SHAP_90 = diag(confusionMat_knn_SHAP_90) ./ sum(confusionMat_knn_SHAP_90, 1)';
%
f1_score_knn_SHAP_10 = 2 * (precision_knn_SHAP_10 .* recall_knn_SHAP_10) ./ (precision_knn_SHAP_10 + recall_knn_SHAP_10); % F1 Score
f1_score_knn_SHAP_20 = 2 * (precision_knn_SHAP_20 .* recall_knn_SHAP_20) ./ (precision_knn_SHAP_20 + recall_knn_SHAP_20); % F1 Score
f1_score_knn_SHAP_30 = 2 * (precision_knn_SHAP_30 .* recall_knn_SHAP_30) ./ (precision_knn_SHAP_30 + recall_knn_SHAP_30); % F1 Score
f1_score_knn_SHAP_40 = 2 * (precision_knn_SHAP_40 .* recall_knn_SHAP_40) ./ (precision_knn_SHAP_40 + recall_knn_SHAP_40); % F1 Score
f1_score_knn_SHAP_50 = 2 * (precision_knn_SHAP_50 .* recall_knn_SHAP_50) ./ (precision_knn_SHAP_50 + recall_knn_SHAP_50); % F1 Score
f1_score_knn_SHAP_60 = 2 * (precision_knn_SHAP_60 .* recall_knn_SHAP_60) ./ (precision_knn_SHAP_60 + recall_knn_SHAP_60); % F1 Score
f1_score_knn_SHAP_70 = 2 * (precision_knn_SHAP_70 .* recall_knn_SHAP_70) ./ (precision_knn_SHAP_70 + recall_knn_SHAP_70); % F1 Score
f1_score_knn_SHAP_80 = 2 * (precision_knn_SHAP_80 .* recall_knn_SHAP_80) ./ (precision_knn_SHAP_80 + recall_knn_SHAP_80); % F1 Score
f1_score_knn_SHAP_90 = 2 * (precision_knn_SHAP_90 .* recall_knn_SHAP_90) ./ (precision_knn_SHAP_90 + recall_knn_SHAP_90); % F1 Score

% Naive Bayes
confusionMat_nb_SHAP_10 = confusionmat(Ytest, predictions_nb_SHAP_10);
confusionMat_nb_SHAP_20 = confusionmat(Ytest, predictions_nb_SHAP_20);
confusionMat_nb_SHAP_30 = confusionmat(Ytest, predictions_nb_SHAP_30);
confusionMat_nb_SHAP_40 = confusionmat(Ytest, predictions_nb_SHAP_40);
confusionMat_nb_SHAP_50 = confusionmat(Ytest, predictions_nb_SHAP_50);
confusionMat_nb_SHAP_60 = confusionmat(Ytest, predictions_nb_SHAP_60);
confusionMat_nb_SHAP_70 = confusionmat(Ytest, predictions_nb_SHAP_70);
confusionMat_nb_SHAP_80 = confusionmat(Ytest, predictions_nb_SHAP_80);
confusionMat_nb_SHAP_90 = confusionmat(Ytest, predictions_nb_SHAP_90);
%
accuracy_nb_SHAP_10 = sum(diag(confusionMat_nb_SHAP_10)) / sum(confusionMat_nb_SHAP_10(:));
accuracy_nb_SHAP_20 = sum(diag(confusionMat_nb_SHAP_20)) / sum(confusionMat_nb_SHAP_20(:));
accuracy_nb_SHAP_30 = sum(diag(confusionMat_nb_SHAP_30)) / sum(confusionMat_nb_SHAP_30(:));
accuracy_nb_SHAP_40 = sum(diag(confusionMat_nb_SHAP_40)) / sum(confusionMat_nb_SHAP_40(:));
accuracy_nb_SHAP_50 = sum(diag(confusionMat_nb_SHAP_50)) / sum(confusionMat_nb_SHAP_50(:));
accuracy_nb_SHAP_60 = sum(diag(confusionMat_nb_SHAP_60)) / sum(confusionMat_nb_SHAP_60(:));
accuracy_nb_SHAP_70 = sum(diag(confusionMat_nb_SHAP_70)) / sum(confusionMat_nb_SHAP_70(:));
accuracy_nb_SHAP_80 = sum(diag(confusionMat_nb_SHAP_80)) / sum(confusionMat_nb_SHAP_80(:));
accuracy_nb_SHAP_90 = sum(diag(confusionMat_nb_SHAP_90)) / sum(confusionMat_nb_SHAP_90(:));
%
recall_nb_SHAP_10 = diag(confusionMat_nb_SHAP_10) ./ sum(confusionMat_nb_SHAP_10, 2);
recall_nb_SHAP_20 = diag(confusionMat_nb_SHAP_20) ./ sum(confusionMat_nb_SHAP_20, 2);
recall_nb_SHAP_30 = diag(confusionMat_nb_SHAP_30) ./ sum(confusionMat_nb_SHAP_30, 2);
recall_nb_SHAP_40 = diag(confusionMat_nb_SHAP_40) ./ sum(confusionMat_nb_SHAP_40, 2);
recall_nb_SHAP_50 = diag(confusionMat_nb_SHAP_50) ./ sum(confusionMat_nb_SHAP_50, 2);
recall_nb_SHAP_60 = diag(confusionMat_nb_SHAP_60) ./ sum(confusionMat_nb_SHAP_60, 2);
recall_nb_SHAP_70 = diag(confusionMat_nb_SHAP_70) ./ sum(confusionMat_nb_SHAP_70, 2);
recall_nb_SHAP_80 = diag(confusionMat_nb_SHAP_80) ./ sum(confusionMat_nb_SHAP_80, 2);
recall_nb_SHAP_90 = diag(confusionMat_nb_SHAP_90) ./ sum(confusionMat_nb_SHAP_90, 2);
%
precision_nb_SHAP_10 = diag(confusionMat_nb_SHAP_10) ./ sum(confusionMat_nb_SHAP_10, 1)';
precision_nb_SHAP_20 = diag(confusionMat_nb_SHAP_20) ./ sum(confusionMat_nb_SHAP_20, 1)';
precision_nb_SHAP_30 = diag(confusionMat_nb_SHAP_30) ./ sum(confusionMat_nb_SHAP_30, 1)';
precision_nb_SHAP_40 = diag(confusionMat_nb_SHAP_40) ./ sum(confusionMat_nb_SHAP_40, 1)';
precision_nb_SHAP_50 = diag(confusionMat_nb_SHAP_50) ./ sum(confusionMat_nb_SHAP_50, 1)';
precision_nb_SHAP_60 = diag(confusionMat_nb_SHAP_60) ./ sum(confusionMat_nb_SHAP_60, 1)';
precision_nb_SHAP_70 = diag(confusionMat_nb_SHAP_70) ./ sum(confusionMat_nb_SHAP_70, 1)';
precision_nb_SHAP_80 = diag(confusionMat_nb_SHAP_80) ./ sum(confusionMat_nb_SHAP_80, 1)';
precision_nb_SHAP_90 = diag(confusionMat_nb_SHAP_90) ./ sum(confusionMat_nb_SHAP_90, 1)';
%
f1_score_nb_SHAP_10 = 2 * (precision_nb_SHAP_10 .* recall_nb_SHAP_10 ./ (precision_nb_SHAP_10 + recall_nb_SHAP_10)); % F1 Score
f1_score_nb_SHAP_20 = 2 * (precision_nb_SHAP_20 .* recall_nb_SHAP_20 ./ (precision_nb_SHAP_20 + recall_nb_SHAP_20)); % F1 Score
f1_score_nb_SHAP_30 = 2 * (precision_nb_SHAP_30 .* recall_nb_SHAP_30 ./ (precision_nb_SHAP_30 + recall_nb_SHAP_30)); % F1 Score
f1_score_nb_SHAP_40 = 2 * (precision_nb_SHAP_40 .* recall_nb_SHAP_40 ./ (precision_nb_SHAP_40 + recall_nb_SHAP_40)); % F1 Score
f1_score_nb_SHAP_50 = 2 * (precision_nb_SHAP_50 .* recall_nb_SHAP_50 ./ (precision_nb_SHAP_50 + recall_nb_SHAP_50)); % F1 Score
f1_score_nb_SHAP_60 = 2 * (precision_nb_SHAP_60 .* recall_nb_SHAP_60 ./ (precision_nb_SHAP_60 + recall_nb_SHAP_60)); % F1 Score
f1_score_nb_SHAP_70 = 2 * (precision_nb_SHAP_70 .* recall_nb_SHAP_70 ./ (precision_nb_SHAP_70 + recall_nb_SHAP_70)); % F1 Score
f1_score_nb_SHAP_80 = 2 * (precision_nb_SHAP_80 .* recall_nb_SHAP_80 ./ (precision_nb_SHAP_80 + recall_nb_SHAP_80)); % F1 Score
f1_score_nb_SHAP_90 = 2 * (precision_nb_SHAP_90 .* recall_nb_SHAP_90 ./ (precision_nb_SHAP_90 + recall_nb_SHAP_90)); % F1 Score

% Neural Network
confusionMat_nn_SHAP_10 = confusionmat(Ytest, predictions_nn_SHAP_10);
confusionMat_nn_SHAP_20 = confusionmat(Ytest, predictions_nn_SHAP_20);
confusionMat_nn_SHAP_30 = confusionmat(Ytest, predictions_nn_SHAP_30);
confusionMat_nn_SHAP_40 = confusionmat(Ytest, predictions_nn_SHAP_40);
confusionMat_nn_SHAP_50 = confusionmat(Ytest, predictions_nn_SHAP_50);
confusionMat_nn_SHAP_60 = confusionmat(Ytest, predictions_nn_SHAP_60);
confusionMat_nn_SHAP_70 = confusionmat(Ytest, predictions_nn_SHAP_70);
confusionMat_nn_SHAP_80 = confusionmat(Ytest, predictions_nn_SHAP_80);
%
accuracy_nn_SHAP_10 = sum(diag(confusionMat_nn_SHAP_10)) / sum(confusionMat_nn_SHAP_10(:));
accuracy_nn_SHAP_20 = sum(diag(confusionMat_nn_SHAP_20)) / sum(confusionMat_nn_SHAP_20(:));
accuracy_nn_SHAP_30 = sum(diag(confusionMat_nn_SHAP_30)) / sum(confusionMat_nn_SHAP_30(:));
accuracy_nn_SHAP_40 = sum(diag(confusionMat_nn_SHAP_40)) / sum(confusionMat_nn_SHAP_40(:));
accuracy_nn_SHAP_50 = sum(diag(confusionMat_nn_SHAP_50)) / sum(confusionMat_nn_SHAP_50(:));
accuracy_nn_SHAP_60 = sum(diag(confusionMat_nn_SHAP_60)) / sum(confusionMat_nn_SHAP_60(:));
accuracy_nn_SHAP_70 = sum(diag(confusionMat_nn_SHAP_70)) / sum(confusionMat_nn_SHAP_70(:));
accuracy_nn_SHAP_80 = sum(diag(confusionMat_nn_SHAP_80)) / sum(confusionMat_nn_SHAP_80(:));
%
recall_nn_SHAP_10 = diag(confusionMat_nn_SHAP_10) ./ sum(confusionMat_nn_SHAP_10, 2);
recall_nn_SHAP_20 = diag(confusionMat_nn_SHAP_20) ./ sum(confusionMat_nn_SHAP_20, 2);
recall_nn_SHAP_30 = diag(confusionMat_nn_SHAP_30) ./ sum(confusionMat_nn_SHAP_30, 2);
recall_nn_SHAP_40 = diag(confusionMat_nn_SHAP_40) ./ sum(confusionMat_nn_SHAP_40, 2);
recall_nn_SHAP_50 = diag(confusionMat_nn_SHAP_50) ./ sum(confusionMat_nn_SHAP_50, 2);
recall_nn_SHAP_60 = diag(confusionMat_nn_SHAP_60) ./ sum(confusionMat_nn_SHAP_60, 2);
recall_nn_SHAP_70 = diag(confusionMat_nn_SHAP_70) ./ sum(confusionMat_nn_SHAP_70, 2);
recall_nn_SHAP_80 = diag(confusionMat_nn_SHAP_80) ./ sum(confusionMat_nn_SHAP_80, 2);
%
precision_nn_SHAP_10 = diag(confusionMat_nn_SHAP_10) ./ sum(confusionMat_nn_SHAP_10, 1)';
precision_nn_SHAP_20 = diag(confusionMat_nn_SHAP_20) ./ sum(confusionMat_nn_SHAP_20, 1)';
precision_nn_SHAP_30 = diag(confusionMat_nn_SHAP_30) ./ sum(confusionMat_nn_SHAP_30, 1)';
precision_nn_SHAP_40 = diag(confusionMat_nn_SHAP_40) ./ sum(confusionMat_nn_SHAP_40, 1)';
precision_nn_SHAP_50 = diag(confusionMat_nn_SHAP_50) ./ sum(confusionMat_nn_SHAP_50, 1)';
precision_nn_SHAP_60 = diag(confusionMat_nn_SHAP_60) ./ sum(confusionMat_nn_SHAP_60, 1)';
precision_nn_SHAP_70 = diag(confusionMat_nn_SHAP_70) ./ sum(confusionMat_nn_SHAP_70, 1)';
precision_nn_SHAP_80 = diag(confusionMat_nn_SHAP_80) ./ sum(confusionMat_nn_SHAP_80, 1)';
%
f1_score_nn_SHAP_10 = 2 * (precision_nn_SHAP_10 .* recall_nn_SHAP_10) ./ (precision_nn_SHAP_10 + recall_nn_SHAP_10); % F1 Score
f1_score_nn_SHAP_20 = 2 * (precision_nn_SHAP_20 .* recall_nn_SHAP_20) ./ (precision_nn_SHAP_20 + recall_nn_SHAP_20); % F1 Score
f1_score_nn_SHAP_30 = 2 * (precision_nn_SHAP_30 .* recall_nn_SHAP_30) ./ (precision_nn_SHAP_30 + recall_nn_SHAP_30); % F1 Score
f1_score_nn_SHAP_40 = 2 * (precision_nn_SHAP_40 .* recall_nn_SHAP_40) ./ (precision_nn_SHAP_40 + recall_nn_SHAP_40); % F1 Score
f1_score_nn_SHAP_50 = 2 * (precision_nn_SHAP_50 .* recall_nn_SHAP_50) ./ (precision_nn_SHAP_50 + recall_nn_SHAP_50); % F1 Score
f1_score_nn_SHAP_60 = 2 * (precision_nn_SHAP_60 .* recall_nn_SHAP_60) ./ (precision_nn_SHAP_60 + recall_nn_SHAP_60); % F1 Score
f1_score_nn_SHAP_70 = 2 * (precision_nn_SHAP_70 .* recall_nn_SHAP_70) ./ (precision_nn_SHAP_70 + recall_nn_SHAP_70); % F1 Score
f1_score_nn_SHAP_80 = 2 * (precision_nn_SHAP_80 .* recall_nn_SHAP_80) ./ (precision_nn_SHAP_80 + recall_nn_SHAP_80); % F1 Score

% Ensemble   
confusionMat_ens_SHAP_10 = confusionmat(Ytest, predictions_ens_SHAP_10);
confusionMat_ens_SHAP_20 = confusionmat(Ytest, predictions_ens_SHAP_20);
confusionMat_ens_SHAP_30 = confusionmat(Ytest, predictions_ens_SHAP_30);
confusionMat_ens_SHAP_40 = confusionmat(Ytest, predictions_ens_SHAP_40);
confusionMat_ens_SHAP_50 = confusionmat(Ytest, predictions_ens_SHAP_50);
confusionMat_ens_SHAP_60 = confusionmat(Ytest, predictions_ens_SHAP_60);
confusionMat_ens_SHAP_70 = confusionmat(Ytest, predictions_ens_SHAP_70);
confusionMat_ens_SHAP_80 = confusionmat(Ytest, predictions_ens_SHAP_80);
confusionMat_ens_SHAP_90 = confusionmat(Ytest, predictions_ens_SHAP_90);
%
accuracy_ens_SHAP_10 = sum(diag(confusionMat_ens_SHAP_10)) / sum(confusionMat_ens_SHAP_10(:));
accuracy_ens_SHAP_20 = sum(diag(confusionMat_ens_SHAP_20)) / sum(confusionMat_ens_SHAP_20(:));
accuracy_ens_SHAP_30 = sum(diag(confusionMat_ens_SHAP_30)) / sum(confusionMat_ens_SHAP_30(:));
accuracy_ens_SHAP_40 = sum(diag(confusionMat_ens_SHAP_40)) / sum(confusionMat_ens_SHAP_40(:));
accuracy_ens_SHAP_50 = sum(diag(confusionMat_ens_SHAP_50)) / sum(confusionMat_ens_SHAP_50(:));
accuracy_ens_SHAP_60 = sum(diag(confusionMat_ens_SHAP_60)) / sum(confusionMat_ens_SHAP_60(:));
accuracy_ens_SHAP_70 = sum(diag(confusionMat_ens_SHAP_70)) / sum(confusionMat_ens_SHAP_70(:));
accuracy_ens_SHAP_80 = sum(diag(confusionMat_ens_SHAP_80)) / sum(confusionMat_ens_SHAP_80(:));
accuracy_ens_SHAP_90 = sum(diag(confusionMat_ens_SHAP_90)) / sum(confusionMat_ens_SHAP_90(:));
%
recall_ens_SHAP_10 = diag(confusionMat_ens_SHAP_10) ./ sum(confusionMat_ens_SHAP_10, 2);
recall_ens_SHAP_20 = diag(confusionMat_ens_SHAP_20) ./ sum(confusionMat_ens_SHAP_20, 2);
recall_ens_SHAP_30 = diag(confusionMat_ens_SHAP_30) ./ sum(confusionMat_ens_SHAP_30, 2);
recall_ens_SHAP_40 = diag(confusionMat_ens_SHAP_40) ./ sum(confusionMat_ens_SHAP_40, 2);
recall_ens_SHAP_50 = diag(confusionMat_ens_SHAP_50) ./ sum(confusionMat_ens_SHAP_50, 2);
recall_ens_SHAP_60 = diag(confusionMat_ens_SHAP_60) ./ sum(confusionMat_ens_SHAP_60, 2);
recall_ens_SHAP_70 = diag(confusionMat_ens_SHAP_70) ./ sum(confusionMat_ens_SHAP_70, 2);
recall_ens_SHAP_80 = diag(confusionMat_ens_SHAP_80) ./ sum(confusionMat_ens_SHAP_80, 2);
recall_ens_SHAP_90 = diag(confusionMat_ens_SHAP_90) ./ sum(confusionMat_ens_SHAP_90, 2);
% 
precision_ens_SHAP_10 = diag(confusionMat_ens_SHAP_10) ./ sum(confusionMat_ens_SHAP_10, 1)';
precision_ens_SHAP_20 = diag(confusionMat_ens_SHAP_20) ./ sum(confusionMat_ens_SHAP_20, 1)';
precision_ens_SHAP_30 = diag(confusionMat_ens_SHAP_30) ./ sum(confusionMat_ens_SHAP_30, 1)';
precision_ens_SHAP_40 = diag(confusionMat_ens_SHAP_40) ./ sum(confusionMat_ens_SHAP_40, 1)';
precision_ens_SHAP_50 = diag(confusionMat_ens_SHAP_50) ./ sum(confusionMat_ens_SHAP_50, 1)';
precision_ens_SHAP_60 = diag(confusionMat_ens_SHAP_60) ./ sum(confusionMat_ens_SHAP_60, 1)';
precision_ens_SHAP_70 = diag(confusionMat_ens_SHAP_70) ./ sum(confusionMat_ens_SHAP_70, 1)';
precision_ens_SHAP_80 = diag(confusionMat_ens_SHAP_80) ./ sum(confusionMat_ens_SHAP_80, 1)';
precision_ens_SHAP_90 = diag(confusionMat_ens_SHAP_90) ./ sum(confusionMat_ens_SHAP_90, 1)';
%
f1_score_ens_SHAP_10 = 2 * (precision_ens_SHAP_10 .* recall_ens_SHAP_10) ./ (precision_ens_SHAP_10 + recall_ens_SHAP_10); % F1 Score
f1_score_ens_SHAP_20 = 2 * (precision_ens_SHAP_20 .* recall_ens_SHAP_20) ./ (precision_ens_SHAP_20 + recall_ens_SHAP_20); % F1 Score
f1_score_ens_SHAP_30 = 2 * (precision_ens_SHAP_30 .* recall_ens_SHAP_30) ./ (precision_ens_SHAP_30 + recall_ens_SHAP_30); % F1 Score
f1_score_ens_SHAP_40 = 2 * (precision_ens_SHAP_40 .* recall_ens_SHAP_40) ./ (precision_ens_SHAP_40 + recall_ens_SHAP_40); % F1 Score
f1_score_ens_SHAP_50 = 2 * (precision_ens_SHAP_50 .* recall_ens_SHAP_50) ./ (precision_ens_SHAP_50 + recall_ens_SHAP_50); % F1 Score
f1_score_ens_SHAP_60 = 2 * (precision_ens_SHAP_60 .* recall_ens_SHAP_60) ./ (precision_ens_SHAP_60 + recall_ens_SHAP_60); % F1 Score
f1_score_ens_SHAP_70 = 2 * (precision_ens_SHAP_70 .* recall_ens_SHAP_70) ./ (precision_ens_SHAP_70 + recall_ens_SHAP_70); % F1 Score
f1_score_ens_SHAP_80 = 2 * (precision_ens_SHAP_80 .* recall_ens_SHAP_80) ./ (precision_ens_SHAP_80 + recall_ens_SHAP_80); % F1 Score
f1_score_ens_SHAP_90 = 2 * (precision_ens_SHAP_90 .* recall_ens_SHAP_90) ./ (precision_ens_SHAP_90 + recall_ens_SHAP_90); % F1 Score
% % 
%     % Exibir a acurácia e a matriz de confusão
% % disp(['Acurácia_KNN_SHAP: ', num2str(accuracy_knn_SHAP)]);
% % disp(['Accuracy_Naive Bayes_SHAP: ', num2str(accuracy_nb_SHAP)]);
% % disp(['Accuracy_Neural Network_SHAP: ', num2str(accuracy_nn_SHAP)]);
% % disp(['Accuracy_Ensemble_SHAP: ', num2str(accuracy_ens_SHAP)]);
% % % disp('Matriz de Confusão_KNN_SHAP:');
% % % disp(confusionMat_knn_SHAP);
% % % disp('Matriz de Confusão_Naive Bayes_SHAP:');
% % % disp(confusionMat_nb_SHAP);
% % % disp('Matriz de Confusão_Neural Network_SHAP:');
% % % disp(confusionMat_nn_SHAP);
% % % disp('Matriz de Confusão_Ensemble_SHAP:');
% % % disp(confusionMat_ens_SHAP);
% Plote Matriz
figure Name 'Matrix Confusion - Features selected by Shap Method (Conditional) - Features thresholds with best accuracy and F1-score'
subplot(2, 2, 1);
confusionchart(confusionMat_knn_SHAP_40,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('KNN - Top 40 Features Selected by Shap (Conditional)');
subplot(2, 2, 2);
confusionchart(confusionMat_nb_SHAP_60,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Naive Bayes - Top 60 Features Selected by Shap (Conditional)');
subplot(2, 2, 3);
confusionchart(confusionMat_nn_SHAP_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Neural Network - Top 20 Features Selected by Shap (Conditional)');
subplot(2, 2, 4);
confusionchart(confusionMat_ens_SHAP_50,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Ensemble - Top 50 Features Selected by Shap (Conditional)');

%% Nova classificação dos dados com os features selecionados pelo método Shap
% Ranqueando os preditores mais importantes por shap (interventional), para cada classificador
% KNN
% n_SHAP = 90;
MeanSHAP_int_knn = explainer_knn_int.MeanAbsoluteShapley.collision + explainer_knn_int.MeanAbsoluteShapley.normal + explainer_knn_int.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_int_knn] = sort(MeanSHAP_int_knn, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_int_train_knn_10 = X_train(:, sorted_indices_int_knn(1:10)); % dados de treino com melhores features
SHAP_int_test_knn_10 = X_test(:, sorted_indices_int_knn(1:10)); % dados de teste com melhores features
SHAP_int_train_knn_20 = X_train(:, sorted_indices_int_knn(1:20)); % dados de treino com melhores features
SHAP_int_test_knn_20 = X_test(:, sorted_indices_int_knn(1:20)); % dados de teste com melhores features
SHAP_int_train_knn_30 = X_train(:, sorted_indices_int_knn(1:30)); % dados de treino com melhores features
SHAP_int_test_knn_30 = X_test(:, sorted_indices_int_knn(1:30)); % dados de teste com melhores features
SHAP_int_train_knn_40 = X_train(:, sorted_indices_int_knn(1:40)); % dados de treino com melhores features
SHAP_int_test_knn_40 = X_test(:, sorted_indices_int_knn(1:40)); % dados de teste com melhores features
SHAP_int_train_knn_50 = X_train(:, sorted_indices_int_knn(1:50)); % dados de treino com melhores features
SHAP_int_test_knn_50 = X_test(:, sorted_indices_int_knn(1:50)); % dados de teste com melhores features
SHAP_int_train_knn_60 = X_train(:, sorted_indices_int_knn(1:60)); % dados de treino com melhores features
SHAP_int_test_knn_60 = X_test(:, sorted_indices_int_knn(1:60)); % dados de teste com melhores features
SHAP_int_train_knn_70 = X_train(:, sorted_indices_int_knn(1:70)); % dados de treino com melhores features
SHAP_int_test_knn_70 = X_test(:, sorted_indices_int_knn(1:70)); % dados de teste com melhores features
SHAP_int_train_knn_80 = X_train(:, sorted_indices_int_knn(1:80)); % dados de treino com melhores features
SHAP_int_test_knn_80 = X_test(:, sorted_indices_int_knn(1:80)); % dados de teste com melhores features
SHAP_int_train_knn_90 = X_train(:, sorted_indices_int_knn(1:90)); % dados de treino com melhores features
SHAP_int_test_knn_90 = X_test(:, sorted_indices_int_knn(1:90)); % dados de teste com melhores features
% Naive Bayes
MeanSHAP_int_nb = explainer_nb_int.MeanAbsoluteShapley.collision + explainer_nb_int.MeanAbsoluteShapley.normal + explainer_nb_int.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_int_nb] = sort(MeanSHAP_int_nb, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_int_train_nb_10 = X_train(:, sorted_indices_int_nb(1:10)); % dados de treino com melhores features
SHAP_int_test_nb_10 = X_test(:, sorted_indices_int_nb(1:10)); % dados de teste com melhores features
SHAP_int_train_nb_20 = X_train(:, sorted_indices_int_nb(1:20)); % dados de treino com melhores features
SHAP_int_test_nb_20 = X_test(:, sorted_indices_int_nb(1:20)); % dados de teste com melhores features
SHAP_int_train_nb_30 = X_train(:, sorted_indices_int_nb(1:30)); % dados de treino com melhores features
SHAP_int_test_nb_30 = X_test(:, sorted_indices_int_nb(1:30)); % dados de teste com melhores features
SHAP_int_train_nb_40 = X_train(:, sorted_indices_int_nb(1:40)); % dados de treino com melhores features
SHAP_int_test_nb_40 = X_test(:, sorted_indices_int_nb(1:40)); % dados de teste com melhores features
SHAP_int_train_nb_50 = X_train(:, sorted_indices_int_nb(1:50)); % dados de treino com melhores features
SHAP_int_test_nb_50 = X_test(:, sorted_indices_int_nb(1:50)); % dados de teste com melhores features
SHAP_int_train_nb_60 = X_train(:, sorted_indices_int_nb(1:60)); % dados de treino com melhores features
SHAP_int_test_nb_60 = X_test(:, sorted_indices_int_nb(1:60)); % dados de teste com melhores features
SHAP_int_train_nb_70 = X_train(:, sorted_indices_int_nb(1:70)); % dados de treino com melhores features
SHAP_int_test_nb_70 = X_test(:, sorted_indices_int_nb(1:70)); % dados de teste com melhores features
SHAP_int_train_nb_80 = X_train(:, sorted_indices_int_nb(1:80)); % dados de treino com melhores features
SHAP_int_test_nb_80 = X_test(:, sorted_indices_int_nb(1:80)); % dados de teste com melhores features
SHAP_int_train_nb_90 = X_train(:, sorted_indices_int_nb(1:90)); % dados de treino com melhores features
SHAP_int_test_nb_90 = X_test(:, sorted_indices_int_nb(1:90)); % dados de teste com melhores features
% Neural Network
MeanSHAP_int_nn = explainer_nn_int.MeanAbsoluteShapley.collision + explainer_nn_int.MeanAbsoluteShapley.normal + explainer_nn_int.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_int_nn] = sort(MeanSHAP_int_nn, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_int_train_nn_10 = X_train(:, sorted_indices_int_nn(1:10)); % dados de treino com melhores features
SHAP_int_test_nn_10 = X_test(:, sorted_indices_int_nn(1:10)); % dados de teste com melhores features
SHAP_int_train_nn_20 = X_train(:, sorted_indices_int_nn(1:20)); % dados de treino com melhores features
SHAP_int_test_nn_20 = X_test(:, sorted_indices_int_nn(1:20)); % dados de teste com melhores features
SHAP_int_train_nn_30 = X_train(:, sorted_indices_int_nn(1:30)); % dados de treino com melhores features
SHAP_int_test_nn_30 = X_test(:, sorted_indices_int_nn(1:30)); % dados de teste com melhores features
SHAP_int_train_nn_40 = X_train(:, sorted_indices_int_nn(1:40)); % dados de treino com melhores features
SHAP_int_test_nn_40 = X_test(:, sorted_indices_int_nn(1:40)); % dados de teste com melhores features
SHAP_int_train_nn_50 = X_train(:, sorted_indices_int_nn(1:50)); % dados de treino com melhores features
SHAP_int_test_nn_50 = X_test(:, sorted_indices_int_nn(1:50)); % dados de teste com melhores features
SHAP_int_train_nn_60 = X_train(:, sorted_indices_int_nn(1:60)); % dados de treino com melhores features
SHAP_int_test_nn_60 = X_test(:, sorted_indices_int_nn(1:60)); % dados de teste com melhores features
SHAP_int_train_nn_70 = X_train(:, sorted_indices_int_nn(1:70)); % dados de treino com melhores features
SHAP_int_test_nn_70 = X_test(:, sorted_indices_int_nn(1:70)); % dados de teste com melhores features
SHAP_int_train_nn_80 = X_train(:, sorted_indices_int_nn(1:80)); % dados de treino com melhores features
SHAP_int_test_nn_80 = X_test(:, sorted_indices_int_nn(1:80)); % dados de teste com melhores features
SHAP_int_train_nn_90 = X_train(:, sorted_indices_int_nn(1:90)); % dados de treino com melhores features
SHAP_int_test_nn_90 = X_test(:, sorted_indices_int_nn(1:90)); % dados de teste com melhores features
% Ensemble
MeanSHAP_int_ens = explainer_ens_int.MeanAbsoluteShapley.collision + explainer_ens_int.MeanAbsoluteShapley.normal + explainer_ens_int.MeanAbsoluteShapley.obstruction;
[~,sorted_indices_int_ens] = sort(MeanSHAP_int_ens, 'descend');% X_top_features = X(:, top_features_indices);% top_features_indices = sorted_indices(1:30);
SHAP_int_train_ens_10 = X_train(:, sorted_indices_int_ens(1:10)); % dados de treino com melhores features
SHAP_int_test_ens_10 = X_test(:, sorted_indices_int_ens(1:10)); % dados de teste com melhores features
SHAP_int_train_ens_20 = X_train(:, sorted_indices_int_ens(1:20)); % dados de treino com melhores features
SHAP_int_test_ens_20 = X_test(:, sorted_indices_int_ens(1:20)); % dados de teste com melhores features
SHAP_int_train_ens_30 = X_train(:, sorted_indices_int_ens(1:30)); % dados de treino com melhores features
SHAP_int_test_ens_30 = X_test(:, sorted_indices_int_ens(1:30)); % dados de teste com melhores features
SHAP_int_train_ens_40 = X_train(:, sorted_indices_int_ens(1:40)); % dados de treino com melhores features
SHAP_int_test_ens_40 = X_test(:, sorted_indices_int_ens(1:40)); % dados de teste com melhores features
SHAP_int_train_ens_50 = X_train(:, sorted_indices_int_ens(1:50)); % dados de treino com melhores features
SHAP_int_test_ens_50 = X_test(:, sorted_indices_int_ens(1:50)); % dados de teste com melhores features
SHAP_int_train_ens_60 = X_train(:, sorted_indices_int_ens(1:60)); % dados de treino com melhores features
SHAP_int_test_ens_60 = X_test(:, sorted_indices_int_ens(1:60)); % dados de teste com melhores features
SHAP_int_train_ens_70 = X_train(:, sorted_indices_int_ens(1:70)); % dados de treino com melhores features
SHAP_int_test_ens_70 = X_test(:, sorted_indices_int_ens(1:70)); % dados de teste com melhores features
SHAP_int_train_ens_80 = X_train(:, sorted_indices_int_ens(1:80)); % dados de treino com melhores features
SHAP_int_test_ens_80 = X_test(:, sorted_indices_int_ens(1:80)); % dados de teste com melhores features
SHAP_int_train_ens_90 = X_train(:, sorted_indices_int_ens(1:90)); % dados de treino com melhores features
SHAP_int_test_ens_90 = X_test(:, sorted_indices_int_ens(1:90)); % dados de teste com melhores features

% % Treinar o modelo novamente usando apenas os recursos selecionados
  modelknn_SHAP_int_10 = fitcknn(SHAP_int_train_knn_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_20 = fitcknn(SHAP_int_train_knn_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_30 = fitcknn(SHAP_int_train_knn_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_40 = fitcknn(SHAP_int_train_knn_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_50 = fitcknn(SHAP_int_train_knn_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_60 = fitcknn(SHAP_int_train_knn_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_70 = fitcknn(SHAP_int_train_knn_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_80 = fitcknn(SHAP_int_train_knn_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelknn_SHAP_int_90 = fitcknn(SHAP_int_train_knn_90,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnb_SHAP_int_10 = fitcnb(SHAP_int_train_nb_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_20 = fitcnb(SHAP_int_train_nb_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_30 = fitcnb(SHAP_int_train_nb_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_40 = fitcnb(SHAP_int_train_nb_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_50 = fitcnb(SHAP_int_train_nb_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_60 = fitcnb(SHAP_int_train_nb_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_70 = fitcnb(SHAP_int_train_nb_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_80 = fitcnb(SHAP_int_train_nb_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnb_SHAP_int_90 = fitcnb(SHAP_int_train_nb_90,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
%
  modelnn_SHAP_int_10 = fitcnet(SHAP_int_train_nn_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_20 = fitcnet(SHAP_int_train_nn_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_30 = fitcnet(SHAP_int_train_nn_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_40 = fitcnet(SHAP_int_train_nn_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_50 = fitcnet(SHAP_int_train_nn_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_60 = fitcnet(SHAP_int_train_nn_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_70 = fitcnet(SHAP_int_train_nn_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_80 = fitcnet(SHAP_int_train_nn_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelnn_SHAP_int_90 = fitcnet(SHAP_int_train_nn_90,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
% %
  modelens_SHAP_int_10 = fitcensemble(SHAP_int_train_ens_10,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_20 = fitcensemble(SHAP_int_train_ens_20,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_30 = fitcensemble(SHAP_int_train_ens_30,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_40 = fitcensemble(SHAP_int_train_ens_40,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_50 = fitcensemble(SHAP_int_train_ens_50,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_60 = fitcensemble(SHAP_int_train_ens_60,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_70 = fitcensemble(SHAP_int_train_ens_70,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_80 = fitcensemble(SHAP_int_train_ens_80,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));
  modelens_SHAP_int_90 = fitcensemble(SHAP_int_train_ens_90,Y_train,"OptimizeHyperparameters","all",'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations',value1));


% Testar o modelo novamente usando apenas os recursos selecionados
predictions_knn_SHAP_int_10=predict(modelknn_SHAP_int_10,SHAP_int_test_knn_10);
predictions_knn_SHAP_int_20=predict(modelknn_SHAP_int_20,SHAP_int_test_knn_20);
predictions_knn_SHAP_int_30=predict(modelknn_SHAP_int_30,SHAP_int_test_knn_30);
predictions_knn_SHAP_int_40=predict(modelknn_SHAP_int_40,SHAP_int_test_knn_40);
predictions_knn_SHAP_int_50=predict(modelknn_SHAP_int_50,SHAP_int_test_knn_50);
predictions_knn_SHAP_int_60=predict(modelknn_SHAP_int_60,SHAP_int_test_knn_60);
predictions_knn_SHAP_int_70=predict(modelknn_SHAP_int_70,SHAP_int_test_knn_70);
predictions_knn_SHAP_int_80=predict(modelknn_SHAP_int_80,SHAP_int_test_knn_80);
predictions_knn_SHAP_int_90=predict(modelknn_SHAP_int_90,SHAP_int_test_knn_90);
%
predictions_nb_SHAP_int_10=predict(modelnb_SHAP_int_10,SHAP_int_test_nb_10);
predictions_nb_SHAP_int_20=predict(modelnb_SHAP_int_20,SHAP_int_test_nb_20);
predictions_nb_SHAP_int_30=predict(modelnb_SHAP_int_30,SHAP_int_test_nb_30);
predictions_nb_SHAP_int_40=predict(modelnb_SHAP_int_40,SHAP_int_test_nb_40);
predictions_nb_SHAP_int_50=predict(modelnb_SHAP_int_50,SHAP_int_test_nb_50);
predictions_nb_SHAP_int_60=predict(modelnb_SHAP_int_60,SHAP_int_test_nb_60);
predictions_nb_SHAP_int_70=predict(modelnb_SHAP_int_70,SHAP_int_test_nb_70);
predictions_nb_SHAP_int_80=predict(modelnb_SHAP_int_80,SHAP_int_test_nb_80);
predictions_nb_SHAP_int_90=predict(modelnb_SHAP_int_90,SHAP_int_test_nb_90);
%
predictions_nn_SHAP_int_10=predict(modelnn_SHAP_int_10,SHAP_int_test_nn_10);
predictions_nn_SHAP_int_20=predict(modelnn_SHAP_int_20,SHAP_int_test_nn_20);
predictions_nn_SHAP_int_30=predict(modelnn_SHAP_int_30,SHAP_int_test_nn_30);
predictions_nn_SHAP_int_40=predict(modelnn_SHAP_int_40,SHAP_int_test_nn_40);
predictions_nn_SHAP_int_50=predict(modelnn_SHAP_int_50,SHAP_int_test_nn_50);
predictions_nn_SHAP_int_60=predict(modelnn_SHAP_int_60,SHAP_int_test_nn_60);
predictions_nn_SHAP_int_70=predict(modelnn_SHAP_int_70,SHAP_int_test_nn_70);
predictions_nn_SHAP_int_80=predict(modelnn_SHAP_int_80,SHAP_int_test_nn_80);
predictions_nn_SHAP_int_90=predict(modelnn_SHAP_int_90,SHAP_int_test_nn_90);
%
predictions_ens_SHAP_int_10=predict(modelens_SHAP_int_10,SHAP_int_test_ens_10);
predictions_ens_SHAP_int_20=predict(modelens_SHAP_int_20,SHAP_int_test_ens_20);
predictions_ens_SHAP_int_30=predict(modelens_SHAP_int_30,SHAP_int_test_ens_30);
predictions_ens_SHAP_int_40=predict(modelens_SHAP_int_40,SHAP_int_test_ens_40);
predictions_ens_SHAP_int_50=predict(modelens_SHAP_int_50,SHAP_int_test_ens_50);
predictions_ens_SHAP_int_60=predict(modelens_SHAP_int_60,SHAP_int_test_ens_60);
predictions_ens_SHAP_int_70=predict(modelens_SHAP_int_70,SHAP_int_test_ens_70);
predictions_ens_SHAP_int_80=predict(modelens_SHAP_int_80,SHAP_int_test_ens_80);
predictions_ens_SHAP_int_90=predict(modelens_SHAP_int_90,SHAP_int_test_ens_90);


Avaliar a performance dos modelos nos dados de teste com os features selecionados
KNN
confusionMat_knn_SHAP_int_10 = confusionmat(Ytest, predictions_knn_SHAP_int_10);
confusionMat_knn_SHAP_int_20 = confusionmat(Ytest, predictions_knn_SHAP_int_20);
confusionMat_knn_SHAP_int_30 = confusionmat(Ytest, predictions_knn_SHAP_int_30);
confusionMat_knn_SHAP_int_40 = confusionmat(Ytest, predictions_knn_SHAP_int_40);
confusionMat_knn_SHAP_int_50 = confusionmat(Ytest, predictions_knn_SHAP_int_50);
confusionMat_knn_SHAP_int_60 = confusionmat(Ytest, predictions_knn_SHAP_int_60);
confusionMat_knn_SHAP_int_70 = confusionmat(Ytest, predictions_knn_SHAP_int_70);
confusionMat_knn_SHAP_int_80 = confusionmat(Ytest, predictions_knn_SHAP_int_80);
confusionMat_knn_SHAP_int_90 = confusionmat(Ytest, predictions_knn_SHAP_int_90);
%
accuracy_knn_SHAP_int_10 = sum(diag(confusionMat_knn_SHAP_int_10)) / sum(confusionMat_knn_SHAP_int_10(:));
accuracy_knn_SHAP_int_20 = sum(diag(confusionMat_knn_SHAP_int_20)) / sum(confusionMat_knn_SHAP_int_20(:));
accuracy_knn_SHAP_int_30 = sum(diag(confusionMat_knn_SHAP_int_30)) / sum(confusionMat_knn_SHAP_int_30(:));
accuracy_knn_SHAP_int_40 = sum(diag(confusionMat_knn_SHAP_int_40)) / sum(confusionMat_knn_SHAP_int_40(:));
accuracy_knn_SHAP_int_50 = sum(diag(confusionMat_knn_SHAP_int_50)) / sum(confusionMat_knn_SHAP_int_50(:));
accuracy_knn_SHAP_int_60 = sum(diag(confusionMat_knn_SHAP_int_60)) / sum(confusionMat_knn_SHAP_int_60(:));
accuracy_knn_SHAP_int_70 = sum(diag(confusionMat_knn_SHAP_int_70)) / sum(confusionMat_knn_SHAP_int_70(:));
accuracy_knn_SHAP_int_80 = sum(diag(confusionMat_knn_SHAP_int_80)) / sum(confusionMat_knn_SHAP_int_80(:));
accuracy_knn_SHAP_int_90 = sum(diag(confusionMat_knn_SHAP_int_90)) / sum(confusionMat_knn_SHAP_int_90(:));
%
recall_knn_SHAP_int_10 = diag(confusionMat_knn_SHAP_int_10) ./ sum(confusionMat_knn_SHAP_int_10, 2);
recall_knn_SHAP_int_20 = diag(confusionMat_knn_SHAP_int_20) ./ sum(confusionMat_knn_SHAP_int_20, 2);
recall_knn_SHAP_int_30 = diag(confusionMat_knn_SHAP_int_30) ./ sum(confusionMat_knn_SHAP_int_30, 2);
recall_knn_SHAP_int_40 = diag(confusionMat_knn_SHAP_int_40) ./ sum(confusionMat_knn_SHAP_int_40, 2);
recall_knn_SHAP_int_50 = diag(confusionMat_knn_SHAP_int_50) ./ sum(confusionMat_knn_SHAP_int_50, 2);
recall_knn_SHAP_int_60 = diag(confusionMat_knn_SHAP_int_60) ./ sum(confusionMat_knn_SHAP_int_60, 2);
recall_knn_SHAP_int_70 = diag(confusionMat_knn_SHAP_int_70) ./ sum(confusionMat_knn_SHAP_int_70, 2);
recall_knn_SHAP_int_80 = diag(confusionMat_knn_SHAP_int_80) ./ sum(confusionMat_knn_SHAP_int_80, 2);
recall_knn_SHAP_int_90 = diag(confusionMat_knn_SHAP_int_90) ./ sum(confusionMat_knn_SHAP_int_90, 2);
%
precision_knn_SHAP_int_10 = diag(confusionMat_knn_SHAP_int_10) ./ sum(confusionMat_knn_SHAP_int_10, 1)';
precision_knn_SHAP_int_20 = diag(confusionMat_knn_SHAP_int_20) ./ sum(confusionMat_knn_SHAP_int_20, 1)';
precision_knn_SHAP_int_30 = diag(confusionMat_knn_SHAP_int_30) ./ sum(confusionMat_knn_SHAP_int_30, 1)';
precision_knn_SHAP_int_40 = diag(confusionMat_knn_SHAP_int_40) ./ sum(confusionMat_knn_SHAP_int_40, 1)';
precision_knn_SHAP_int_50 = diag(confusionMat_knn_SHAP_int_50) ./ sum(confusionMat_knn_SHAP_int_50, 1)';
precision_knn_SHAP_int_60 = diag(confusionMat_knn_SHAP_int_60) ./ sum(confusionMat_knn_SHAP_int_60, 1)';
precision_knn_SHAP_int_70 = diag(confusionMat_knn_SHAP_int_70) ./ sum(confusionMat_knn_SHAP_int_70, 1)';
precision_knn_SHAP_int_80 = diag(confusionMat_knn_SHAP_int_80) ./ sum(confusionMat_knn_SHAP_int_80, 1)';
precision_knn_SHAP_int_90 = diag(confusionMat_knn_SHAP_int_90) ./ sum(confusionMat_knn_SHAP_int_90, 1)';
%
f1_score_knn_SHAP_int_10 = 2 * (precision_knn_SHAP_int_10 .* recall_knn_SHAP_int_10) ./ (precision_knn_SHAP_int_10 + recall_knn_SHAP_int_10); % F1 Score
f1_score_knn_SHAP_int_20 = 2 * (precision_knn_SHAP_int_20 .* recall_knn_SHAP_int_20) ./ (precision_knn_SHAP_int_20 + recall_knn_SHAP_int_20); % F1 Score
f1_score_knn_SHAP_int_30 = 2 * (precision_knn_SHAP_int_30 .* recall_knn_SHAP_int_30) ./ (precision_knn_SHAP_int_30 + recall_knn_SHAP_int_30); % F1 Score
f1_score_knn_SHAP_int_40 = 2 * (precision_knn_SHAP_int_40 .* recall_knn_SHAP_int_40) ./ (precision_knn_SHAP_int_40 + recall_knn_SHAP_int_40); % F1 Score
f1_score_knn_SHAP_int_50 = 2 * (precision_knn_SHAP_int_50 .* recall_knn_SHAP_int_50) ./ (precision_knn_SHAP_int_50 + recall_knn_SHAP_int_50); % F1 Score
f1_score_knn_SHAP_int_60 = 2 * (precision_knn_SHAP_int_60 .* recall_knn_SHAP_int_60) ./ (precision_knn_SHAP_int_60 + recall_knn_SHAP_int_60); % F1 Score
f1_score_knn_SHAP_int_70 = 2 * (precision_knn_SHAP_int_70 .* recall_knn_SHAP_int_70) ./ (precision_knn_SHAP_int_70 + recall_knn_SHAP_int_70); % F1 Score
f1_score_knn_SHAP_int_80 = 2 * (precision_knn_SHAP_int_80 .* recall_knn_SHAP_int_80) ./ (precision_knn_SHAP_int_80 + recall_knn_SHAP_int_80); % F1 Score
f1_score_knn_SHAP_int_90 = 2 * (precision_knn_SHAP_int_90 .* recall_knn_SHAP_int_90) ./ (precision_knn_SHAP_int_90 + recall_knn_SHAP_int_90); % F1 Score

% Naive Bayes
confusionMat_nb_SHAP_int_10 = confusionmat(Ytest, predictions_nb_SHAP_int_10);
confusionMat_nb_SHAP_int_20 = confusionmat(Ytest, predictions_nb_SHAP_int_20);
confusionMat_nb_SHAP_int_30 = confusionmat(Ytest, predictions_nb_SHAP_int_30);
confusionMat_nb_SHAP_int_40 = confusionmat(Ytest, predictions_nb_SHAP_int_40);
confusionMat_nb_SHAP_int_50 = confusionmat(Ytest, predictions_nb_SHAP_int_50);
confusionMat_nb_SHAP_int_60 = confusionmat(Ytest, predictions_nb_SHAP_int_60);
confusionMat_nb_SHAP_int_70 = confusionmat(Ytest, predictions_nb_SHAP_int_70);
confusionMat_nb_SHAP_int_80 = confusionmat(Ytest, predictions_nb_SHAP_int_80);
confusionMat_nb_SHAP_int_90 = confusionmat(Ytest, predictions_nb_SHAP_int_90);
%
accuracy_nb_SHAP_int_10 = sum(diag(confusionMat_nb_SHAP_int_10)) / sum(confusionMat_nb_SHAP_int_10(:));
accuracy_nb_SHAP_int_20 = sum(diag(confusionMat_nb_SHAP_int_20)) / sum(confusionMat_nb_SHAP_int_20(:));
accuracy_nb_SHAP_int_30 = sum(diag(confusionMat_nb_SHAP_int_30)) / sum(confusionMat_nb_SHAP_int_30(:));
accuracy_nb_SHAP_int_40 = sum(diag(confusionMat_nb_SHAP_int_40)) / sum(confusionMat_nb_SHAP_int_40(:));
accuracy_nb_SHAP_int_50 = sum(diag(confusionMat_nb_SHAP_int_50)) / sum(confusionMat_nb_SHAP_int_50(:));
accuracy_nb_SHAP_int_60 = sum(diag(confusionMat_nb_SHAP_int_60)) / sum(confusionMat_nb_SHAP_int_60(:));
accuracy_nb_SHAP_int_70 = sum(diag(confusionMat_nb_SHAP_int_70)) / sum(confusionMat_nb_SHAP_int_70(:));
accuracy_nb_SHAP_int_80 = sum(diag(confusionMat_nb_SHAP_int_80)) / sum(confusionMat_nb_SHAP_int_80(:));
accuracy_nb_SHAP_int_90 = sum(diag(confusionMat_nb_SHAP_int_90)) / sum(confusionMat_nb_SHAP_int_90(:));
%
recall_nb_SHAP_int_10 = diag(confusionMat_nb_SHAP_int_10) ./ sum(confusionMat_nb_SHAP_int_10, 2);
recall_nb_SHAP_int_20 = diag(confusionMat_nb_SHAP_int_20) ./ sum(confusionMat_nb_SHAP_int_20, 2);
recall_nb_SHAP_int_30 = diag(confusionMat_nb_SHAP_int_30) ./ sum(confusionMat_nb_SHAP_int_30, 2);
recall_nb_SHAP_int_40 = diag(confusionMat_nb_SHAP_int_40) ./ sum(confusionMat_nb_SHAP_int_40, 2);
recall_nb_SHAP_int_50 = diag(confusionMat_nb_SHAP_int_50) ./ sum(confusionMat_nb_SHAP_int_50, 2);
recall_nb_SHAP_int_60 = diag(confusionMat_nb_SHAP_int_60) ./ sum(confusionMat_nb_SHAP_int_60, 2);
recall_nb_SHAP_int_70 = diag(confusionMat_nb_SHAP_int_70) ./ sum(confusionMat_nb_SHAP_int_70, 2);
recall_nb_SHAP_int_80 = diag(confusionMat_nb_SHAP_int_80) ./ sum(confusionMat_nb_SHAP_int_80, 2);
recall_nb_SHAP_int_90 = diag(confusionMat_nb_SHAP_int_90) ./ sum(confusionMat_nb_SHAP_int_90, 2);
%
precision_nb_SHAP_int_10 = diag(confusionMat_nb_SHAP_int_10) ./ sum(confusionMat_nb_SHAP_int_10, 1)';
precision_nb_SHAP_int_20 = diag(confusionMat_nb_SHAP_int_20) ./ sum(confusionMat_nb_SHAP_int_20, 1)';
precision_nb_SHAP_int_30 = diag(confusionMat_nb_SHAP_int_30) ./ sum(confusionMat_nb_SHAP_int_30, 1)';
precision_nb_SHAP_int_40 = diag(confusionMat_nb_SHAP_int_40) ./ sum(confusionMat_nb_SHAP_int_40, 1)';
precision_nb_SHAP_int_50 = diag(confusionMat_nb_SHAP_int_50) ./ sum(confusionMat_nb_SHAP_int_50, 1)';
precision_nb_SHAP_int_60 = diag(confusionMat_nb_SHAP_int_60) ./ sum(confusionMat_nb_SHAP_int_60, 1)';
precision_nb_SHAP_int_70 = diag(confusionMat_nb_SHAP_int_70) ./ sum(confusionMat_nb_SHAP_int_70, 1)';
precision_nb_SHAP_int_80 = diag(confusionMat_nb_SHAP_int_80) ./ sum(confusionMat_nb_SHAP_int_80, 1)';
precision_nb_SHAP_int_90 = diag(confusionMat_nb_SHAP_int_90) ./ sum(confusionMat_nb_SHAP_int_90, 1)';
%
f1_score_nb_SHAP_int_10 = 2 * (precision_nb_SHAP_int_10 .* recall_nb_SHAP_int_10 ./ (precision_nb_SHAP_int_10 + recall_nb_SHAP_int_10)); % F1 Score
f1_score_nb_SHAP_int_20 = 2 * (precision_nb_SHAP_int_20 .* recall_nb_SHAP_int_20 ./ (precision_nb_SHAP_int_20 + recall_nb_SHAP_int_20)); % F1 Score
f1_score_nb_SHAP_int_30 = 2 * (precision_nb_SHAP_int_30 .* recall_nb_SHAP_int_30 ./ (precision_nb_SHAP_int_30 + recall_nb_SHAP_int_30)); % F1 Score
f1_score_nb_SHAP_int_40 = 2 * (precision_nb_SHAP_int_40 .* recall_nb_SHAP_int_40 ./ (precision_nb_SHAP_int_40 + recall_nb_SHAP_int_40)); % F1 Score
f1_score_nb_SHAP_int_50 = 2 * (precision_nb_SHAP_int_50 .* recall_nb_SHAP_int_50 ./ (precision_nb_SHAP_int_50 + recall_nb_SHAP_int_50)); % F1 Score
f1_score_nb_SHAP_int_60 = 2 * (precision_nb_SHAP_int_60 .* recall_nb_SHAP_int_60 ./ (precision_nb_SHAP_int_60 + recall_nb_SHAP_int_60)); % F1 Score
f1_score_nb_SHAP_int_70 = 2 * (precision_nb_SHAP_int_70 .* recall_nb_SHAP_int_70 ./ (precision_nb_SHAP_int_70 + recall_nb_SHAP_int_70)); % F1 Score
f1_score_nb_SHAP_int_80 = 2 * (precision_nb_SHAP_int_80 .* recall_nb_SHAP_int_80 ./ (precision_nb_SHAP_int_80 + recall_nb_SHAP_int_80)); % F1 Score
f1_score_nb_SHAP_int_90 = 2 * (precision_nb_SHAP_int_90 .* recall_nb_SHAP_int_90 ./ (precision_nb_SHAP_int_90 + recall_nb_SHAP_int_90)); % F1 Score

% Neural Network
confusionMat_nn_SHAP_int_10 = confusionmat(Ytest, predictions_nn_SHAP_int_10);
confusionMat_nn_SHAP_int_20 = confusionmat(Ytest, predictions_nn_SHAP_int_20);
confusionMat_nn_SHAP_int_30 = confusionmat(Ytest, predictions_nn_SHAP_int_30);
confusionMat_nn_SHAP_int_40 = confusionmat(Ytest, predictions_nn_SHAP_int_40);
confusionMat_nn_SHAP_int_50 = confusionmat(Ytest, predictions_nn_SHAP_int_50);
confusionMat_nn_SHAP_int_60 = confusionmat(Ytest, predictions_nn_SHAP_int_60);
confusionMat_nn_SHAP_int_70 = confusionmat(Ytest, predictions_nn_SHAP_int_70);
confusionMat_nn_SHAP_int_80 = confusionmat(Ytest, predictions_nn_SHAP_int_80);
confusionMat_nn_SHAP_int_90 = confusionmat(Ytest, predictions_nn_SHAP_int_90);
%
accuracy_nn_SHAP_int_10 = sum(diag(confusionMat_nn_SHAP_int_10)) / sum(confusionMat_nn_SHAP_int_10(:));
accuracy_nn_SHAP_int_20 = sum(diag(confusionMat_nn_SHAP_int_20)) / sum(confusionMat_nn_SHAP_int_20(:));
accuracy_nn_SHAP_int_30 = sum(diag(confusionMat_nn_SHAP_int_30)) / sum(confusionMat_nn_SHAP_int_30(:));
accuracy_nn_SHAP_int_40 = sum(diag(confusionMat_nn_SHAP_int_40)) / sum(confusionMat_nn_SHAP_int_40(:));
accuracy_nn_SHAP_int_50 = sum(diag(confusionMat_nn_SHAP_int_50)) / sum(confusionMat_nn_SHAP_int_50(:));
accuracy_nn_SHAP_int_60 = sum(diag(confusionMat_nn_SHAP_int_60)) / sum(confusionMat_nn_SHAP_int_60(:));
accuracy_nn_SHAP_int_70 = sum(diag(confusionMat_nn_SHAP_int_70)) / sum(confusionMat_nn_SHAP_int_70(:));
accuracy_nn_SHAP_int_80 = sum(diag(confusionMat_nn_SHAP_int_80)) / sum(confusionMat_nn_SHAP_int_80(:));
accuracy_nn_SHAP_int_90 = sum(diag(confusionMat_nn_SHAP_int_90)) / sum(confusionMat_nn_SHAP_int_90(:));
%
recall_nn_SHAP_int_10 = diag(confusionMat_nn_SHAP_int_10) ./ sum(confusionMat_nn_SHAP_int_10, 2);
recall_nn_SHAP_int_20 = diag(confusionMat_nn_SHAP_int_20) ./ sum(confusionMat_nn_SHAP_int_20, 2);
recall_nn_SHAP_int_30 = diag(confusionMat_nn_SHAP_int_30) ./ sum(confusionMat_nn_SHAP_int_30, 2);
recall_nn_SHAP_int_40 = diag(confusionMat_nn_SHAP_int_40) ./ sum(confusionMat_nn_SHAP_int_40, 2);
recall_nn_SHAP_int_50 = diag(confusionMat_nn_SHAP_int_50) ./ sum(confusionMat_nn_SHAP_int_50, 2);
recall_nn_SHAP_int_60 = diag(confusionMat_nn_SHAP_int_60) ./ sum(confusionMat_nn_SHAP_int_60, 2);
recall_nn_SHAP_int_70 = diag(confusionMat_nn_SHAP_int_70) ./ sum(confusionMat_nn_SHAP_int_70, 2);
recall_nn_SHAP_int_80 = diag(confusionMat_nn_SHAP_int_80) ./ sum(confusionMat_nn_SHAP_int_80, 2);
recall_nn_SHAP_int_90 = diag(confusionMat_nn_SHAP_int_90) ./ sum(confusionMat_nn_SHAP_int_90, 2);
%
precision_nn_SHAP_int_10 = diag(confusionMat_nn_SHAP_int_10) ./ sum(confusionMat_nn_SHAP_int_10, 1)';
precision_nn_SHAP_int_20 = diag(confusionMat_nn_SHAP_int_20) ./ sum(confusionMat_nn_SHAP_int_20, 1)';
precision_nn_SHAP_int_30 = diag(confusionMat_nn_SHAP_int_30) ./ sum(confusionMat_nn_SHAP_int_30, 1)';
precision_nn_SHAP_int_40 = diag(confusionMat_nn_SHAP_int_40) ./ sum(confusionMat_nn_SHAP_int_40, 1)';
precision_nn_SHAP_int_50 = diag(confusionMat_nn_SHAP_int_50) ./ sum(confusionMat_nn_SHAP_int_50, 1)';
precision_nn_SHAP_int_60 = diag(confusionMat_nn_SHAP_int_60) ./ sum(confusionMat_nn_SHAP_int_60, 1)';
precision_nn_SHAP_int_70 = diag(confusionMat_nn_SHAP_int_70) ./ sum(confusionMat_nn_SHAP_int_70, 1)';
precision_nn_SHAP_int_80 = diag(confusionMat_nn_SHAP_int_80) ./ sum(confusionMat_nn_SHAP_int_80, 1)';
precision_nn_SHAP_int_90 = diag(confusionMat_nn_SHAP_int_90) ./ sum(confusionMat_nn_SHAP_int_90, 1)';
%
f1_score_nn_SHAP_int_10 = 2 * (precision_nn_SHAP_int_10 .* recall_nn_SHAP_int_10) ./ (precision_nn_SHAP_int_10 + recall_nn_SHAP_int_10); % F1 Score
f1_score_nn_SHAP_int_20 = 2 * (precision_nn_SHAP_int_20 .* recall_nn_SHAP_int_20) ./ (precision_nn_SHAP_int_20 + recall_nn_SHAP_int_20); % F1 Score
f1_score_nn_SHAP_int_30 = 2 * (precision_nn_SHAP_int_30 .* recall_nn_SHAP_int_30) ./ (precision_nn_SHAP_int_30 + recall_nn_SHAP_int_30); % F1 Score
f1_score_nn_SHAP_int_40 = 2 * (precision_nn_SHAP_int_40 .* recall_nn_SHAP_int_40) ./ (precision_nn_SHAP_int_40 + recall_nn_SHAP_int_40); % F1 Score
f1_score_nn_SHAP_int_50 = 2 * (precision_nn_SHAP_int_50 .* recall_nn_SHAP_int_50) ./ (precision_nn_SHAP_int_50 + recall_nn_SHAP_int_50); % F1 Score
f1_score_nn_SHAP_int_60 = 2 * (precision_nn_SHAP_int_60 .* recall_nn_SHAP_int_60) ./ (precision_nn_SHAP_int_60 + recall_nn_SHAP_int_60); % F1 Score
f1_score_nn_SHAP_int_70 = 2 * (precision_nn_SHAP_int_70 .* recall_nn_SHAP_int_70) ./ (precision_nn_SHAP_int_70 + recall_nn_SHAP_int_70); % F1 Score
f1_score_nn_SHAP_int_80 = 2 * (precision_nn_SHAP_int_80 .* recall_nn_SHAP_int_80) ./ (precision_nn_SHAP_int_80 + recall_nn_SHAP_int_80); % F1 Score
f1_score_nn_SHAP_int_90 = 2 * (precision_nn_SHAP_int_90 .* recall_nn_SHAP_int_90) ./ (precision_nn_SHAP_int_90 + recall_nn_SHAP_int_90); % F1 Score

% Ensemble   
confusionMat_ens_SHAP_int_10 = confusionmat(Ytest, predictions_ens_SHAP_int_10);
confusionMat_ens_SHAP_int_20 = confusionmat(Ytest, predictions_ens_SHAP_int_20);
confusionMat_ens_SHAP_int_30 = confusionmat(Ytest, predictions_ens_SHAP_int_30);
confusionMat_ens_SHAP_int_40 = confusionmat(Ytest, predictions_ens_SHAP_int_40);
confusionMat_ens_SHAP_int_50 = confusionmat(Ytest, predictions_ens_SHAP_int_50);
confusionMat_ens_SHAP_int_60 = confusionmat(Ytest, predictions_ens_SHAP_int_60);
confusionMat_ens_SHAP_int_70 = confusionmat(Ytest, predictions_ens_SHAP_int_70);
confusionMat_ens_SHAP_int_80 = confusionmat(Ytest, predictions_ens_SHAP_int_80);
confusionMat_ens_SHAP_int_90 = confusionmat(Ytest, predictions_ens_SHAP_int_90);
%
accuracy_ens_SHAP_int_10 = sum(diag(confusionMat_ens_SHAP_int_10)) / sum(confusionMat_ens_SHAP_int_10(:));
accuracy_ens_SHAP_int_20 = sum(diag(confusionMat_ens_SHAP_int_20)) / sum(confusionMat_ens_SHAP_int_20(:));
accuracy_ens_SHAP_int_30 = sum(diag(confusionMat_ens_SHAP_int_30)) / sum(confusionMat_ens_SHAP_int_30(:));
accuracy_ens_SHAP_int_40 = sum(diag(confusionMat_ens_SHAP_int_40)) / sum(confusionMat_ens_SHAP_int_40(:));
accuracy_ens_SHAP_int_50 = sum(diag(confusionMat_ens_SHAP_int_50)) / sum(confusionMat_ens_SHAP_int_50(:));
accuracy_ens_SHAP_int_60 = sum(diag(confusionMat_ens_SHAP_int_60)) / sum(confusionMat_ens_SHAP_int_60(:));
accuracy_ens_SHAP_int_70 = sum(diag(confusionMat_ens_SHAP_int_70)) / sum(confusionMat_ens_SHAP_int_70(:));
accuracy_ens_SHAP_int_80 = sum(diag(confusionMat_ens_SHAP_int_80)) / sum(confusionMat_ens_SHAP_int_80(:));
accuracy_ens_SHAP_int_90 = sum(diag(confusionMat_ens_SHAP_int_90)) / sum(confusionMat_ens_SHAP_int_90(:));
%
recall_ens_SHAP_int_10 = diag(confusionMat_ens_SHAP_int_10) ./ sum(confusionMat_ens_SHAP_int_10, 2);
recall_ens_SHAP_int_20 = diag(confusionMat_ens_SHAP_int_20) ./ sum(confusionMat_ens_SHAP_int_20, 2);
recall_ens_SHAP_int_30 = diag(confusionMat_ens_SHAP_int_30) ./ sum(confusionMat_ens_SHAP_int_30, 2);
recall_ens_SHAP_int_40 = diag(confusionMat_ens_SHAP_int_40) ./ sum(confusionMat_ens_SHAP_int_40, 2);
recall_ens_SHAP_int_50 = diag(confusionMat_ens_SHAP_int_50) ./ sum(confusionMat_ens_SHAP_int_50, 2);
recall_ens_SHAP_int_60 = diag(confusionMat_ens_SHAP_int_60) ./ sum(confusionMat_ens_SHAP_int_60, 2);
recall_ens_SHAP_int_70 = diag(confusionMat_ens_SHAP_int_70) ./ sum(confusionMat_ens_SHAP_int_70, 2);
recall_ens_SHAP_int_80 = diag(confusionMat_ens_SHAP_int_80) ./ sum(confusionMat_ens_SHAP_int_80, 2);
recall_ens_SHAP_int_90 = diag(confusionMat_ens_SHAP_int_90) ./ sum(confusionMat_ens_SHAP_int_90, 2);
% 
precision_ens_SHAP_int_10 = diag(confusionMat_ens_SHAP_int_10) ./ sum(confusionMat_ens_SHAP_int_10, 1)';
precision_ens_SHAP_int_20 = diag(confusionMat_ens_SHAP_int_20) ./ sum(confusionMat_ens_SHAP_int_20, 1)';
precision_ens_SHAP_int_30 = diag(confusionMat_ens_SHAP_int_30) ./ sum(confusionMat_ens_SHAP_int_30, 1)';
precision_ens_SHAP_int_40 = diag(confusionMat_ens_SHAP_int_40) ./ sum(confusionMat_ens_SHAP_int_40, 1)';
precision_ens_SHAP_int_50 = diag(confusionMat_ens_SHAP_int_50) ./ sum(confusionMat_ens_SHAP_int_50, 1)';
precision_ens_SHAP_int_60 = diag(confusionMat_ens_SHAP_int_60) ./ sum(confusionMat_ens_SHAP_int_60, 1)';
precision_ens_SHAP_int_70 = diag(confusionMat_ens_SHAP_int_70) ./ sum(confusionMat_ens_SHAP_int_70, 1)';
precision_ens_SHAP_int_80 = diag(confusionMat_ens_SHAP_int_80) ./ sum(confusionMat_ens_SHAP_int_80, 1)';
precision_ens_SHAP_int_90 = diag(confusionMat_ens_SHAP_int_90) ./ sum(confusionMat_ens_SHAP_int_90, 1)';
%
f1_score_ens_SHAP_int_10 = 2 * (precision_ens_SHAP_int_10 .* recall_ens_SHAP_int_10) ./ (precision_ens_SHAP_int_10 + recall_ens_SHAP_int_10); % F1 Score
f1_score_ens_SHAP_int_20 = 2 * (precision_ens_SHAP_int_20 .* recall_ens_SHAP_int_20) ./ (precision_ens_SHAP_int_20 + recall_ens_SHAP_int_20); % F1 Score
f1_score_ens_SHAP_int_30 = 2 * (precision_ens_SHAP_int_30 .* recall_ens_SHAP_int_30) ./ (precision_ens_SHAP_int_30 + recall_ens_SHAP_int_30); % F1 Score
f1_score_ens_SHAP_int_40 = 2 * (precision_ens_SHAP_int_40 .* recall_ens_SHAP_int_40) ./ (precision_ens_SHAP_int_40 + recall_ens_SHAP_int_40); % F1 Score
f1_score_ens_SHAP_int_50 = 2 * (precision_ens_SHAP_int_50 .* recall_ens_SHAP_int_50) ./ (precision_ens_SHAP_int_50 + recall_ens_SHAP_int_50); % F1 Score
f1_score_ens_SHAP_int_60 = 2 * (precision_ens_SHAP_int_60 .* recall_ens_SHAP_int_60) ./ (precision_ens_SHAP_int_60 + recall_ens_SHAP_int_60); % F1 Score
f1_score_ens_SHAP_int_70 = 2 * (precision_ens_SHAP_int_70 .* recall_ens_SHAP_int_70) ./ (precision_ens_SHAP_int_70 + recall_ens_SHAP_int_70); % F1 Score
f1_score_ens_SHAP_int_80 = 2 * (precision_ens_SHAP_int_80 .* recall_ens_SHAP_int_80) ./ (precision_ens_SHAP_int_80 + recall_ens_SHAP_int_80); % F1 Score
f1_score_ens_SHAP_int_90 = 2 * (precision_ens_SHAP_int_90 .* recall_ens_SHAP_int_90) ./ (precision_ens_SHAP_int_90 + recall_ens_SHAP_int_90); % F1 Score
% 
% % 
% %     % Exibir a acurácia e a matriz de confusão
% % % disp(['Acurácia_KNN_SHAP: ', num2str(accuracy_knn_SHAP)]);
% % % disp(['Accuracy_Naive Bayes_SHAP: ', num2str(accuracy_nb_SHAP)]);
% % % disp(['Accuracy_Neural Network_SHAP: ', num2str(accuracy_nn_SHAP)]);
% % % disp(['Accuracy_Ensemble_SHAP: ', num2str(accuracy_ens_SHAP)]);
% % % % disp('Matriz de Confusão_KNN_SHAP:');
% % % % disp(confusionMat_knn_SHAP);
% % % % disp('Matriz de Confusão_Naive Bayes_SHAP:');
% % % % disp(confusionMat_nb_SHAP);
% % % % disp('Matriz de Confusão_Neural Network_SHAP:');
% % % % disp(confusionMat_nn_SHAP);
% % % % disp('Matriz de Confusão_Ensemble_SHAP:');
% % % % disp(confusionMat_ens_SHAP);
% Plote Matriz
figure Name 'Matrix Confusion - Features selected by Shap Method (Interventional) - Features thresholds with best accuracy and F1-score'
subplot(2, 2, 1);
confusionchart(confusionMat_knn_SHAP_int_30,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('KNN - Top 30 Features Selected by Shap (Interventional)');
subplot(2, 2, 2);
confusionchart(confusionMat_nb_SHAP_int_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Naive Bayes - Top 20 Features Selected by Shap (Interventional)');
subplot(2, 2, 3);
confusionchart(confusionMat_nn_SHAP_int_30,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Neural Network - Top 30 Features Selected by Shap (Interventional)');
subplot(2, 2, 4);
confusionchart(confusionMat_ens_SHAP_int_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Ensemble - Top 20 Features Selected by Shap (Interventional)');

Plote Matriz - Melhores Resultados (comparação com os resultados com todos os features)
figure Name 'Matrix Confusion - Features selected by best Method - Features thresholds with best accuracy and F1-score'
subplot(2, 2, 1);
confusionchart(confusionMat_knn_SHAP_40,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('KNN - Top 40 Features Selected by Shap (Conditional)');
subplot(2, 2, 2);
confusionchart(confusionMat_nb_SHAP_int_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Naive Bayes - Top 20 Features Selected by Shap (Interventional)');
subplot(2, 2, 3);
confusionchart(confusionMat_nn_SHAP_20,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Neural Network - Top 20 Features Selected by Shap (Conditional)');
subplot(2, 2, 4);
confusionchart(confusionMat_ens_SHAP_50,classesm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Ensemble - Top 50 Features Selected by Shap (Conditional)');









% Validação da interpretabilidade com o método Shap (condicional) nos dados de teste (esta etapa não tem relação com a seleção de atributos)
% %  % Xtest = table2array(X_test); % Transformação para interpretabilidade
% % explainer_knn_test_int = shapley(modelknn, Xtest,QueryPoints=Xtest,Method="interventional"); 
% % % explainer_nb_test = shapley(modelnb, Xtest,QueryPoints=Xtest,Method="conditional");
% % % explainer_nn_test = shapley(modelnn, Xtest,QueryPoints=Xtest,Method="conditional"); 
% % % explainer_ens_test = shapley(modelens, Xtest,QueryPoints=Xtest,Method="conditional"); 
% % % Plotar gráficos Shap
% % % Plot visualization of mean(abs(shap)) bar plot, and swarmchart for each output class. Note that these multi-query-point plots require R2024a or higher
% % figure(1); tiledlayout(2,2); nexttile(1);
% % % % Plot the mean(abs(shap)) plot for this multi-query-point shapley object
% %  plot(explainer_knn_test_int,"NumImportantPredictors",90);
% % % Plot the shapley summary swarmchart for each output class
% % % KNN
% % for i=2:4
% %     nexttile(i);
% %     swarmchart(explainer_knn_test_int,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=90)
% % end
% % % % Naive Bayes
% % % figure(1); tiledlayout(2,2); nexttile(1);  
% % % plot(explainer_nb_test,"NumImportantPredictors",20); 
% % % for i=2:4
% % %     nexttile(i);
% % %     swarmchart(explainer_nb_test,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=20)
% % % end
% % % % Neural Network
% % % figure(1); tiledlayout(2,2); nexttile(1);    
% % % plot(explainer_nn_test,"NumImportantPredictors",90);
% % % for i=2:4
% % %     nexttile(i);
% % %     swarmchart(explainer_nn_test,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=90)
% % % end
% % % % Ensemble
% % % figure(1); tiledlayout(2,2); nexttile(1);    
% % % plot(explainer_ens_test,"NumImportantPredictors",20); 
% % % for i=2:4
% % %     nexttile(i);
% % %     swarmchart(explainer_ens_test,ClassName=classess{i-1},ColorMap='bluered',NumImportantPredictors=20)
% % % end