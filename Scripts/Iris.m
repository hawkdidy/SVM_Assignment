load('C:\Users\hakim\Documents\GitHub\SVM_Assignment\Datasets\iris.mat')
gam = 10;
type='c'; 
%linear kernel 
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
 
%plotting
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});
%error


[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)


%polynomial kernal
t = 1;
for degree = 1:5
[alpha,b] = trainlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'});

figure

plotlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'},{alpha,b})

title({'Polynomial kernel' 'Degree:' degree})
end

%% RBF Kernel sig2

sig2list = [0.01, 0.5, 4, 10, 20, 100];
errlist = [];
for sig2 = sig2list
    [alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'RBF_kernel'});
    Ytest = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xt);
    err = sum(Ytest~=Yt);
    errlist = [errlist; err/length(Yt)*100];
    
end
plot(sig2list, errlist)
ylabel('error rate %'); xlabel('sig2')
title('RBF Kernel sig2')
%% RBF Kernel gamm
sig2 = 12;
gamlist = [0.01, 0.5, 4, 10, 20, 100];
errlistg = [];
for gam = gamlist
    [alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'RBF_kernel'});
    Ytest = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xt);
    err = sum(Ytest~=Yt);
    errlistg = [errlistg; err/length(Yt)*100];

end

plot(gamlist, errlistg)
ylabel('error rate '); xlabel('gamma ')
title('RBF Kernel gam')
