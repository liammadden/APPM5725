function verify = testAdjoint(functionhandle1,functionhandle2,N)
% functionhandle1 is the linear map, functionhandle2 is its adjoint
% verify = 1 means the adjoint is verified, verify = 0 means the adjoint is
% wrong

epsilon = 10^-9;
tests = 5;
verify = 1;
for i = 1:tests
    x = rand([N,1]);
    y = rand([N,1]);
    ip1 = functionhandle1(x)'*y;
    ip2 = x'*functionhandle2(y);
    error = abs(ip1-ip2);
    if error >= epsilon
        verify = 0;
    else
    end
end

end