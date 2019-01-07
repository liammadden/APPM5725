%% Liam Madden
% APPM 5720
% Homework 4

%% Abstract
% In this experiment, we started with a discrete signal. Then, we blurred
% it using circular convolution, and we added noise to it. Acting as though
% we had been given the blurred and noised signal without knowing how it
% started, our goal was to recover the original signal. To do so, we solved
% the least squares problem in CVX and got the value of the dual variable
% from this. Then, we used the dual variable to determine our equivalent
% sparse least squares problem. We solved this in CVX using the matrix
% representation of our "blur" function and in fasta using the actual
% "blur" function. The latter took about 1/20th the time of the former to
% compute.

%% Clear
clear all
close all

%% Blur
% First, define the signal
N = 100;
x = zeros(N,1);
x(10) = 1;
x(13) = -1;
x(50) = 0.3;
x(70) = -0.2;
% Then, define the blur function
B = @(x) ifft(fft(x).*fft(exp(-[-2:2].^2/2)',N));
% and determine its matrix representation
matrixB = implicit2explicit(B,N);
% Now, apply the blur function to x and add noise
sigma = 0.02; % standard deviation
z = normrnd(0,sigma,N,1); % noise
y = B(x) + z; % y=matrixB*x+z

%% Deblur
epsilon = sigma*sqrt(N);
% Recover x using least squares in CVX with matrixB
cvx_begin
    variables xr(N)
    dual variable lambda
    minimize norm(xr,1)
    subject to
        lambda : sum_square(matrixB*xr-y) <= epsilon^2;
cvx_end

%% Plot
count = 1:N;
figure(1)
plot(count,x,'-k',count,y,'-c',count,xr,'-g')
xlabel({'The recovered signal is almost exactly the same as the original signal.','The blurred signal is the original signal, blurred, with noise,','and an offset. The offset is due to the fact that we used','circular convolution instead of linear convolution. We used circular','convolution in order for our code to be more efficient.'})
title('Original, blurred, and recovered signal')
legend('Original signal','Blurred signal','Deblurred signal')

%% Dual
% Check that the dual x is the same as the recovered x
cvx_begin
    variables xd(N)
    minimize norm(xd,1)+lambda*sum_square(matrixB*xd-y)
cvx_end
disp(norm(xd-xr)); % muy poco

%% Adjoint
% The adjoint of B should be
Bstar = @(x) ifft(fft(x).*conj(fft(exp(-[-2:2].^2/2)',N)));
% verify that it is actually the adjoint
verify = testAdjoint(B,Bstar,N); % the adjoint is verified (verify = 1)

%% CVX
% Now, time the problem using CVX and matrixB
tau = 1/2/lambda;
tic
cvx_begin
    variables x3(N)
    minimize tau*norm(x3,1)+0.5*sum_square(matrixB*x3-y)
cvx_end
t1 = toc;
disp(norm(x3-xr)); % I mean, this was the exact same as how we calculated xd

%% Fasta
% Now, time the problem using fasta and B
opts = [];
opts.recordObjective = true;
opts.verbose=true;
opts.stringHeader='    ';
initialGuess = zeros(N,1);
tic
[x4,outs] = fasta_sparseLeastSquares(B,Bstar,y,tau,initialGuess,opts);
t2 = toc;

%% Compare CVX to fasta
disp(norm(x4-x3)); % the two different methods (CVX and fasta) give solutions that are the same up to three digits of accuracy
disp(t1);
disp(t2); % fasta is much faster!!!!!