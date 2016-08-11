%%Synthetic data

X = (-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);

%%training/validation set 

Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

%%LS-SVM with the RBF kernal and arbitrary values 
gam = 200;
sig2 = 10;
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

%%visualized results on training  
plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});

%%results of trained model on test data
YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},...
{alpha,b},Xtest);

%% results of test 
plot(Xtest,Ytest,'.');
hold on;
plot(Xtest,YtestEst,'r+');
legend('Ytest','YtestEst');

%% hyper parameter tunning 
cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10);
cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2});


optFun = 'simplex';
globalOptFun = 'ds';
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', ...
globalOptFun},optFun,'crossvalidatelssvm',{10,'mse'})

%% model with tuned parameters 

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});

%% application of the bayesian framework 
sig2 = 0.5; gam = 10;
criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)

%% criteria optimization 
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

%% IRIS
gam = 10; sig2 = .5;
bay_modoutClass({X,Y,'c',gam,sig2},'figure');

%% 
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*(X(:,1))) +0.3.*randn(100,1);
[selected, ranking] = bay_lssvmARD({X,Y,'class',gam,sig2});

%% 

X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) +0.1.*rand(size(X));
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));


%% 
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
plotlssvm(model);