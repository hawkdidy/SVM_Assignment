% set the parameters to some value
gam = 100;
sig2 = 1;
% generate random indices
idx = randperm(size(X,1));
% create the training and validation sets
% using the randomized indices

Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

%training model 
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2, ...
'RBF_kernel'});

%evaluating model
estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, ...
{alpha,b},Xval);

% Performance in terms of misclassification
err = sum(estYval~=Yval);
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yval)*100)

% TUNING
%1RBF Kernel sig2 tunning
gam = 10;
sig2list = [0.1, 1, 10];
errlist = [];


for sig2 = sig2list
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
    estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);

    err = sum(estYval~=Yval);
    errlist = [errlist; err/length(Yval)*100];
    
end
plot(sig2list, errlist);
ylabel('error rate %'); xlabel('sig2')
title('RBF Kernel sig2')

%RBF Kernel gamm tunning
sig2 = 1;
gamlist = [1, 10, 100];
errlistg = [];
for gam = gamlist
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
    
    estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},Xval);

    err = sum(estYval~=Yval);
    errlistg = [errlistg; err/length(Yval)*100];

end

plot(gamlist, errlistg);
ylabel('error rate %'); xlabel('gam')
title('RBF Kernel gam')

%Perform crossvalidation using 10 folds:

performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'}, 10,'misclass');

performancel = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass')

%Use tunelssvm procedure to optimize the hyperpatameters. Execute:
parameter = char('csa','ds');
model = {X,Y,'c',[],[],'RBF_kernel','csa'};

optroutine = char('simplex','gridsearch');
optroutine = optroutine(1,:);
[gam,sig2,cost] = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'});
% ROC curve
Xv = Xt; 

[alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'RBF_kernel'});
[Ysim,Ylatent] = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xv);
roc(Ylatent,Yval);
