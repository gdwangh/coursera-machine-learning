function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
min_err=1;

for i=1:8,
	Ci = C_list(i);
	for j=1:8,
		sigma_j = sigma_list(j);
		model= svmTrain(X, y, Ci, @(x1, x2) gaussianKernel(x1, x2, sigma_j)); 
		predictions = svmPredict(model, Xval);
		pred_err = mean(double(predictions ~= yval));
		%fprintf("Ci=%f, sigma_j=%f, pred_err=%f, min_err=%f\n", Ci, sigma_j, pred_err, min_err);
		if (pred_err < min_err),
			min_err = pred_err;
			C = Ci;
			sigma = sigma_j;
			%fprintf("	pred_err < min_err, %f--->C, %f --> sigma\n", Ci, sigma_j)
		end;
	end;
end;


% =========================================================================

end
