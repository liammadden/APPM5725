%% Liam Madden
% APPM 5720
% Homework 6

%% Abstract
% Here, we test our gradient check function. We try it on 3 1D functions: a
% linear one, a quadratic one, and a cubic one. We also try it on a high
% dimensional quadratic. Then we try it on scaled functions. Then we try it
% on a gradient that is slightly incorrect. And finally, we try it on our
% negative log-liklihood function. Displayed are the errors for each of
% these cases respectively. As you can see, the error is small for the 1D
% functions, confirming that the gradients are correct. The error is small
% for high dimensional quadratic as well, confirming that its gradient is
% correct too. Scaling make the error larger, however, it is still small
% enough to confirm the gradient. The error is ~30 for the incorrect
% gradient though, showing that the gradient is not accurate. Finally, the
% error is small for the negative log-liklihood function, which was what we
% really needed to confirm.

%% Clear
clear all
close all

%% Try on 1-D functions
a = @(x) x^3;
b = @(x) 3*x^2;
c = @(x) 6*x;
d = @(x) 6;
x = 10;
dx = 1;
[h1,atest1,atest2] = gradCheck(a,b,x);
% does a'=b?
disp(min([atest1(length(h1)) atest2(length(h1))]))
% yes!
[~,btest1,btest2] = gradCheck(b,c,x);
% does b'=c?
disp(min([btest1(length(h1)) btest2(length(h1))]))
% yes!
[~,ctest1,ctest2] = gradCheck(c,d,x);
% does c'=d?
disp(min([ctest1(length(h1)) ctest2(length(h1))]))
% yes!

%% Try on a high-dimensional quadratic
N = 100;
A = rand(N);
e = @(x) x'*A*x;
f = @(x) (A+A')*x;
X = ones(N,1);
dX = randi(2,[N,1]);
[h2,etest1,etest2] = gradCheck(e,f,X);
% does e'=f?
disp(min([etest1(length(h2)) etest2(length(h2))]))
% yes!

%% Try scaling
sa = @(x) 100*a(x);
sb = @(x) 100*b(x);
[~,satest1,satest2] = gradCheck(sa,sb,x);
% does sa'=sb?
disp(min([satest1(length(h1)) satest2(length(h1))]))
% yes!

%% Can it distinguish slightly off gradients?
ib = @(x) .9*b(x);
[~,iatest1,iatest2] = gradCheck(a,ib,x);
% does a'=ib?
disp(min([iatest1(length(h1)) iatest2(length(h1))]))
% no!

%% Now try it on our negative log-liklihood function
load('spamData.mat')
Xtrain = log(Xtrain+0.1);
Xtest = log(Xtest+0.1);
ytrain = 2*ytrain-1;
ytest = 2*ytest-1;
p = length(Xtrain(1,:));
ntrain = length(Xtrain(:,1));
ntest = length(Xtest(:,1));
sigma = @(a) 1./(1+exp(-a));
muTrain = @(w) sigma(ytrain.*(Xtrain*w));
lTrain = @(w) -sum(log(muTrain(w)));
gradlTrain = @(w) -Xtrain'*(ytrain.*(1-muTrain(w)));
w0 = ones(p,1);
[h,test1,test2] = gradCheck(lTrain,gradlTrain,w0);
% does lTrain'=gradlTrain?
disp(min([test1(length(h1)) test2(length(h1))]))
% yes!