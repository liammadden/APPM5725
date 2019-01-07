%% Liam Madden
% APPM 5720
% Homework 6
% Problem 3

%% Abstract
% Here, we test out gradient descent function by applying it to a function
% that we know the minimum to. The algorithm does, in fact, converge to the
% minimizer.

%% Clear
clear all
close all

%% Try gradDesc
A = [1 1; 1 2; 1 3; 1 4];
b = [0 0 0 1]';
f = @(x) .5*norm(A*x-b)^2;
gradf = @(x)  A'*(A*x-b);
L = norm(A'*A);
x0 = randi(2,[2,1])-1;
tol = .0001;
x = gradDesc(x0,f,gradf,tol,1000,L,1);
xstar = (A'*A)\(A'*b);
error = zeros(length(x(1,:)),1);
for i = 1:length(error)
    error(i) = norm(x(:,i)-xstar);
end

%% Plot
figure(1)
plot(1:length(x),error,'-k')
xlabel({'Number of iterations','The error converges to zero. The tolerance was .0001.'})
ylabel('Error')
title('Gradient descent convergence')