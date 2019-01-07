function A = implicit2explicit(functionhandle,N)
% functionhandle is a linear function mapping from R^N to R^N. Thus, A*x =
% functionhandle(x)

A = zeros(N,N);
for i = 1:N
    e = zeros(N,1);
    e(i) = 1;
    A(:,i) = functionhandle(e); % input to functionhandle is a column, output is a column
end

end