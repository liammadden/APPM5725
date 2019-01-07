%% Liam Madden
% APPM 5720
% Homework 10
% Problems 1-3

%% Abstract
% Here, we compared the basis pursuit problem to the least squares problem.
% First, we solved for the basis pursuit minimizer using CVX. Then, we used
% the minimizer to calculate the value of tau to make the two problems
% equivalent. Then we solved for the least squares minimizer using FISTA
% and found that the two minimizers were the same. Then, we solved for tau
% another way. We used duality to solve for the derivative of the objective
% as a function of tau, and then applied Newton's method. Newton's method
% converged to the same tau as the first part.

%% Clear
clear all
close all
clc

%% Construct
rng(254)
m = 10;
n = 20;
A = rand([m,n]);
b = rand([m,1]);

%% CVX
cvx_begin quiet
    variables xBP(n)
    minimize norm(xBP,1)
    subject to
        A*xBP-b == 0;
cvx_end

%% FISTA
tau = norm(xBP,1);
x0 = zeros(n,1);
f = @(x) .5*norm(A*x-b)^2;
gradf = @(x) A'*(A*x-b);
Lf = norm(A'*A);
prox = @(x,gamma) project_l1(x,tau);
tol = 1e-6;
maxIter = 1e4;
minIter = 1e3;
x = fista(x0,f,gradf,Lf,prox,tol,maxIter,minIter);
xLS = x(:,length(x(1,:)));

%% Compare xBP vs xLS
disp(norm(xBP-xLS));

%% Tau
tau1 = 1;
newtol = 1e-3;
err = newtol + 1;
counter = 1;
while err >= newtol
    prox1 = @(x,gamma) project_l1(x,tau1);
    x = fista(x0,f,gradf,Lf,prox1,tol,maxIter,minIter);
    xtau = x(:,length(x(1,:)));
    z = b - A*xtau;
    nu = z/norm(z);
    lambda = norm(A'*nu,Inf);
    newtau1 = tau1+f(xtau)/lambda;
    err = abs(newtau1-tau1);
    tau1 = newtau1;
    counter = counter + 1;
end

%% Compare tauBP to tauLS
disp(abs(tau-tau1));