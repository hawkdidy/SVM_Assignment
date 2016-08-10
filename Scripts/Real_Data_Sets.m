%% 1. Look at the plotted data. What seems to be important properties of the data?

scatter(X(:,1),X(:,2),30, Y)

%% 2. Try a linear model. Do you think it’s suf?cient?
mdl = LinearModel.fit(X,Y)
plot(mdl)

%% 3. Try the RBF kernel and tune its parameter. 


%csa and ds
model = {X,Y,'c',[],[],'RBF_kernel','DS'};

%simplex and gridsearch

[gam,sig2,cost] = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'});

%% 4. Consider the ROC curve. What does it say about the choice between 2. and 3.?
% 4 ROC curve

%CORRESPONDING GAM AND SIG2 FROM THE TUNING



[alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'lin_kernel'});
[Ysim,Ylatent] = simlssvm({X,Y,'c',gam,sig2,'lin_kernel'},{alpha,b},Xt);
roc(Ylatent,Yval);

% 5. Judge your ?nal model. Is the methodology perfectly suited for this data-set?
