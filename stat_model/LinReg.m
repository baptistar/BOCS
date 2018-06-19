% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

classdef LinReg

	properties

		estimator	% type of estimator: mle, bayes, horseshoe, sparseHS
		order 		% maximum order of monomials included in model
		nVars		% number of input variables
		nCoeffs 	% number of parameters to estimate

		xTrain		% input training data - predictors/covariates
		yTrain 		% input training data - output

		xInf  		% X data with Inf output value - separate from training
		yInf 		% Y data with Inf output value - separate from training

		alpha_mle 	% MLE alpha vector
		alpha_hs    % horseshoe alpha vector
		alphaMean   % bayes posterior alpha mean
		alphaCov    % bayes posterior alpha covariance
		a 	 		% bayes posterior a hyper-parameter
		b  			% bayes posterior b hyper-parameter

	end

	methods
		function LR = LinReg(nVars, order, estimator, varargin)

			p = ImprovedInputParser;
			addRequired(p,'nVars');
			addRequired(p,'order');
			addRequired(p,'estimator');
			parse(p,nVars,order,estimator,varargin{:});
			LR = passMatchedArgsToProperties(p, LR);

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function LR = setupData(LR)

			% limit data to unique points
			[X, x_idx] = unique(LR.xTrain,'rows');
			y = LR.yTrain(x_idx);

			% set upper threshold
			infT = 1e6;

			% separate samples based on Inf output
			y_Infidx  = find(abs(y) > infT);
			y_nInfidx = setdiff(1:length(y), y_Infidx);

			% save samples in LR
			LR.xInf = X(y_Infidx,:);
			LR.yInf = y(y_Infidx);

			LR.xTrain = X(y_nInfidx,:);
			LR.yTrain = y(y_nInfidx);

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function LR = train(LR, inputs)

			% Set nSamps (Gibbs iterations to run)
			inputs.nSamps = 1e3;

			% set data
			LR.xTrain = inputs.x_vals;
			LR.yTrain = inputs.y_vals;

			% setup data for training
			LR = LR.setupData();

			% compute all covariates based on order
			LR.xTrain = order_effects(LR.xTrain, LR.order);
			if strcmp(LR.estimator, 'bayes') || strcmp(LR.estimator, 'mle')
				LR.xTrain = [ones(size(LR.xTrain,1),1), LR.xTrain];
			end

			% find the total number of coefficients
			LR.nCoeffs = size(LR.xTrain,2);

			% run training for different estimators
			if strcmp(LR.estimator, 'bayes')
				LR = LR.bayesTrain(inputs.aPr, inputs.bPr);
			elseif strcmp(LR.estimator, 'mle')
				LR = LR.mleTrain();
			elseif (strcmp(LR.estimator, 'horseshoe') || strcmp(LR.estimator, 'sparseHS'))
				LR = LR.horseshoeTrain(inputs.nSamps);
			else
				error('Estimator is not implemented!')
			end

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function LR = bayesTrain(LR, aPr, bPr)

			nCoeffs_ = LR.nCoeffs;

			% set alpha prior parameters
			aMeanPr = (pi/2)/(nCoeffs_)*ones(nCoeffs_,1);
			aCovPr  = eye(nCoeffs_);
			aInvPr  = inv(aCovPr);

			% extract data
			X = LR.xTrain;
			y = LR.yTrain;
			n = size(X,1);

			% compute alpha posterior parameters
			alphaPrec_ = aInvPr + X'*X;
			alphaCov_  = aCovPr - aCovPr*X'*((eye(n) + X*aCovPr*X')\(X*aCovPr)); % Sherman-Morrison formula
			alphaMean_ = alphaCov_*(aInvPr*aMeanPr + X'*y);

			% symmetrizing covariance
			alphaCov_  = 0.5*(alphaCov_ + alphaCov_');

			% compute hyper-parameter posterior parameters
			aPost = aPr + n/2;
			bPost = bPr + 0.5*(aMeanPr'*aInvPr*aMeanPr + y'*y - alphaMean_'*alphaPrec_*alphaMean_);

			% save results in LR
			LR.alphaMean = alphaMean_;
			LR.alphaCov  = alphaCov_;
			LR.a = aPost;
			LR.b = bPost;

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function LR = mleTrain(LR)

			% extract data
			X = LR.xTrain;
			y = LR.yTrain;
			
			% Compute MLE estimator and save in structure
			regParam = 1e-6;
			LR.alpha_mle = (X'*X + regParam*eye(LR.nCoeffs))\(X'*y);
		
		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function LR = horseshoeTrain(LR,nSamps)

			% extract data
			nCoeffs_ = LR.nCoeffs;
			X = LR.xTrain;
			y = LR.yTrain;

			[n_samp, orig_coeff] = size(X);

			% check if x_train contains columns with zeros or duplicates
			idx_zero = find(ismember(X',zeros(1,n_samp),'rows'));
			if (~isempty(idx_zero) || size(unique(X','rows'),1) < size(X,2))

				% remove columns of zeroes
				X(:,idx_zero) = [];
				idx_col = idx_zero;

				% find linearly independent columns to keep
				%idx_licol = LR.licols(X);
				%X = X(:,idx_licol);

				% set idx_col
				%idx_nz  = setdiff(1:orig_coeff,idx_zero);      % kept nz columns
				%idx_nli = setdiff(1:length(idx_nz),idx_licol); % non-li columns
				%idx_col = [idx_zero; idx_nz(idx_nli)'];

			else
				idx_col = [];
			end

			% run Gibbs sampler for nSamps steps
			attempt = 1;
			while(attempt)

				try
					[alpha,a0,~,~] = bhs(X, y, nSamps, 0, 1);
				catch
					continue
				end

				% run until alpha matrix does not contain any NaNs
				if ~any(any(isnan(alpha)))
					attempt = 0;
				end

			end
			
			% append zeros back - note alpha(1,:) is linear intercept
			alpha_pad = zeros(nCoeffs_, 1);
			alpha_pad(setdiff(1:nCoeffs_,idx_col)) = alpha(:,end);
			LR.alpha_hs = [a0(end); alpha_pad];

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function col_idx = licols(LR, X)
		% Extract a linearly independent set of columns from matrix X
		%
		%  X: The given input matrix
		%  tol: A rank estimation tolerance. Default=1e-10
		%  idx: The indices (into X) of the extracted columns

			% set tolerance
			tol=1e-10;

			% compute QR decomposition
			[~, R, E] = qr(X,0); 
			if ~isvector(R)
			    diagr = abs(diag(R));
			else
			    diagr = R(1);   
			end

			% rank estimation
			r = find(diagr >= tol*diagr(1), 1, 'last');

			% extract linearly independent columns
			col_idx = sort(E(1:r));

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function alpha = sampleCoeff(LR)

			% Setup objective function
			if strcmp(LR.estimator, 'bayes')
				sigma2 = 1/gamrnd(LR.a, 1/LR.b);
				alpha  = mvnrnd(LR.alphaMean, sigma2*LR.alphaCov)';
			elseif strcmp(LR.estimator, 'mle')
				alpha  = LR.alpha_mle;
			elseif strcmp(LR.estimator, 'horseshoe')
				alpha  = LR.alpha_hs;
			elseif strcmp(LR.estimator, 'sparseHS')
				alpha  = LR.alpha_hs;
				SpThreshold = 0.1;
				alpha(abs(alpha) < SpThreshold) = 0;
			else
				error('Estimator is not implemented!')
			end

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
		function out = surrogate_model(LR, x, alpha)
		% SURROGATE_MODEL: Function evaluates the linear model
		% Assumption: input x only contains one row

			% generate x_all (all basis vectors) based on model order
			x_all = [1, order_effects(x, LR.order)];

			% check if x maps to an Inf output
			barrier = 0;
			if ismember(LR.xInf, x, 'rows')
				barrier = Inf;
			end

			% compute objective with barrier
			out = x_all*alpha + barrier;

		end %endFunction
		% --------------------------------------------------------------
		% --------------------------------------------------------------
	end %endMethods
end %endClass
