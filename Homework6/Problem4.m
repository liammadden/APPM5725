%% Liam Madden
% APPM 5720
% Homework 6
% Problem 4

%% Abstract
% In this experiment we found an optimal decision variable, w, to predict
% whether an email is spam or not, where our data points had 57 features.
% We found w using training data, and the prediction was 5.22% accurate for
% the training data. We then applied w to the test data, and the prediction
% was 5.79% accurate. Also included is a plot of the convergence of the
% gradient descent method for w.

%% Clear
clear all
close all

%% Load
load('spamData.mat')

%% Format
Xtrain = log(Xtrain+0.1);
Xtest = log(Xtest+0.1);
ytrain = 2*ytrain-1;
ytest = 2*ytest-1;

%% Indices
p = length(Xtrain(1,:));
ntrain = length(Xtrain(:,1));
ntest = length(Xtest(:,1));

%% Objective function (negative log-likelihood)
sigma = @(a) 1./(1+exp(-a));
muTrain = @(w) sigma(ytrain.*(Xtrain*w));
lTrain = @(w) -sum(log(muTrain(w)));
gradlTrain = @(w) -Xtrain'*(ytrain.*(1-muTrain(w)));
w0 = ones(p,1);
[h,test1,test2] = gradCheck(lTrain,gradlTrain,w0);

%% Training
tol = .001;
w = gradDesc(w0,lTrain,gradlTrain,tol,10000,0,1);

%% Plot
iter = [1:length(w(1,:))]';
figure(1)
plot(iter,log(lTrain(w(:,iter))),'-k')
xlabel('Number of iterations')
ylabel('Log of negative log-liklihood')
title('Gradient descent convergence')

%% Training mis-classification rate
wstar = w(:,length(w(1,:)));
sigmawstar = sigma(Xtrain*wstar);
ytrainApprox = ones(ntrain,1);
for i = 1:ntrain
    if sigmawstar(i) <= 0.5
        ytrainApprox(i) = -1;
    else
    end
end
errorTrain = norm(ytrainApprox-ytrain,1)/2/ntrain;
disp(errorTrain)

%% Testing mis-classification rate
sigmawstarTest = sigma(Xtest*wstar);
ytestApprox = ones(ntest,1);
for i = 1:ntest
    if sigmawstarTest(i) <= 0.5
        ytestApprox(i) = -1;
    else
    end
end
errorTest = norm(ytestApprox-ytest,1)/2/ntest;
disp(errorTest)