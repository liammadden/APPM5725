%% Liam Madden
% APPM 5720
% Homework 10
% Problem 4

%% Abstract
% In this experiment we tried to recover a sound file after downsampling
% it. The sound file we used was Handel's Messiah. We downsampled by a
% factor of 4. At this point, the sound file was not even distinguishable
% as music. Then solved, using Newton's method, for the parameter tau. This
% took 3 iterations. We used this parameter to make our least squares
% problem equivalent to the basis pursuit problem. We solved the latter for
% x and then computed our recovered signal from x. The recovered sound file
% sounds just like the original but with a little more static.

%% Clear
clear all
close all
clc

%% Load
load handel.mat
%spectrogram(y,5e2,[],[],Fs)
N = length(y);
M = floor(N/4);

%% Sample
rng(329)
idx = randperm(N,M);
idx = sort(idx);
psi = @(x) x(idx);
psiadj = @(x) accumarray(idx',x,[N 1]);
b = psi(y);

%% Objective
win = sin( pi*( (1:1024) + 1/2)/(1024) );
A = @(x) psi(adjointShortTimeDCT(x,win,N));
Aadj = @(x) forwardShortTimeDCT(psiadj(x),win);
f = @(x) .5*norm(A(x)-b)^2;
Lf = 1; % norm(A'*A);
gradf = @(x) Aadj(A(x)-b);
x0 = zeros(147456,1);

%% Recover
tol = 1e-6;
maxIter = 1e4;
minIter = 1e3;
newtol = 1e-5;
err = newtol + 1;
counter = 1;
tau = 1;
while err >= newtol
    prox = @(x,gamma) project_l1(x,tau);
    x = fista(x0,f,gradf,Lf,prox,tol,maxIter,minIter);
    xtau = x(:,length(x(1,:)));
    z = b - A(xtau);
    nu = z/norm(z);
    lambda = norm(Aadj(nu),Inf);
    newtau = tau+f(xtau)/lambda;
    err = abs(newtau-tau);
    tau = newtau;
    counter = counter + 1;
end
yhat = adjointShortTimeDCT(xtau,win,N);

%% Listen og
playerObj1 = audioplayer(y,Fs);
play( playerObj1 )

%% Listen downsampled
playerObj2 = audioplayer(b,Fs/4);
play( playerObj2 )

%% Listen recovered
playerObj3 = audioplayer(yhat,Fs);
play( playerObj3 )