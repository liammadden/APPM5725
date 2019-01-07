function y = adjoint(functionhandle,x)
% functionhandle is a linear mapping from R^N (column) to R^N (column)
% x is a column
% y = functionhandle^*(x)

N = length(x);
y = zeros(N,1);
for i = 1:N
    e = zeros(N,1);
    e(i) = 1;
    y(i) = x'*functionhandle(e);
end

end