% this yields identical results to cca.py
% except for a scale factor?

load('~/github/thunder/data/cca_test.mat')

X = bsxfun(@minus,X,mean(X));
Y = bsxfun(@minus,Y,mean(Y));

[UOne,SOne,VOne] = svd(X,'econ');
[UTwo,STwo,VTwo] = svd(Y,'econ');

useDimNum = 3;

[canonCoeffOne,canonCoeffTwo,r,canonVarOne,canonVarTwo] = canoncorr(VOne(:,1:useDimNum),VTwo(:,1:useDimNum));

out11 = X * VOne(:,1:useDimNum) * canonCoeffOne(:,1)
out12 = Y * VTwo(:,1:useDimNum) * canonCoeffTwo(:,1)
