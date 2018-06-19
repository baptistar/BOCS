function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = Contamination(x, runlength, seed, ~)
% function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = Contamination(x, runlength, seed, other);
% x is a vector containing binary var for yes/no to prevention efforts done
% at the stage
% runlength is the number of independent generations of simulated time
% seed is the index of the substreams to use (integer >= 1)
% other is not used
% Returns cost of prevention treatment, constraint, and ConstraintCov
% If contraints not satisfied,
% prints comparison between probabilities that the contamination fractions are within
% the thresholds versus the probabilities (1-epsilon) that need to be exceeded for each stage.
%
%Note: RandStream.setGlobalStream(stream) can only be used for Matlab
%versions 2011 and later
%For earlier versions, use the method RandStream.setDefaultStream(stream)
%
%   *************************************************************
%   ***            Code written by Danielle Lertola           ***
%   ***          dcl96@cornell.edu    June 25th, 2012         ***
%   ***     Note for future update:  Example parameters need  ***
%   ***                              to be more reasonable.   ***
%   ***            Edited by Jennifer Shih                    ***
%   ***          jls493@cornell.edu    June 18th, 2014        ***
%   *************************************************************
%
% Last updated Jun 18, 2014


FnVar=NaN; 
FnGrad = NaN;
FnGradCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

n=length(x);
if (length(x)~=n) ||(sum(x>ones(n,1))>0) || (sum(x<zeros(n,1))>0) || (runlength <= 0) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('\nx has %u elements, elements of x are binary, \nrunlength should be positive and real, seed should be a positive integer.\n',n);
    fn = NaN;
    constraint = NaN;
    ConstraintCov = NaN;
else % main simulation
    %% *********************PARAMETERS*********************
    nGen=runlength;          %number of independent generations
    u=x;                     %prevention binary decision variable
    X=zeros(n,nGen);         %fraction contaminated at each stage for each generation
    epsilon=.05*ones(n,1);   %error probability
    p=.1*ones(n,1);          %proportion limit
    cost=ones(n,1);          %cost for prevention at stage i
    %Beta parameters for initial contamination, contamination rate,
    %restoration rate
    initialAlpha=1;
    initialBeta=30;
    contamAlpha=1;
    contamBeta=17/3;
    restoreAlpha=1;
    restoreBeta=3/7;
    
    %% GENERATE RANDOM NUMBER STREAMS
    % Generate new streams for
    [InitialStream, ContaminationStream, RestorationStream] = RandStream.create('mrg32k3a', 'NumStreams', 3);
    % Set the substream to the "seed"
    InitialStream.Substream = seed;
    ContaminationStream.Substream = seed;
    RestorationStream.Substream = seed;

    %% Generate initial fraction of contamination
    OldStream = RandStream.setGlobalStream(InitialStream); % Temporarily store old stream, for versions 2011 and later
    %OldStream = RandStream.setDefaultStream(InitialStream);%for versions 2010 and earlier
    % Generate initial fraction of contamination for stage 1 for each
    % generation
    initialX=betarnd(initialAlpha,initialBeta,1,nGen);
    
    %% Generate rates of contamination
    RandStream.setGlobalStream(ContaminationStream); %for Matlab versions 2011 and later
    %RandStream.setDefaultStream(ContaminationStream); % for versions 2010 and earlier
    
    % Generate rates of contamination for each stage and generation
    Lambda=betarnd(contamAlpha,contamBeta,n,nGen);
    
    %% Generate rates of restoration
    RandStream.setGlobalStream(RestorationStream); %for Matlab versions 2011 and later
    %RandStream.setDefaultStream(RestorationStream); %for Matlab versions 2010 and earlier
    % Generate rates of restoration for each stage and generation
    Gamma=betarnd(restoreAlpha,restoreBeta,n,nGen);
    
    RandStream.setGlobalStream(OldStream);                   % Restore old random number stream
    %RandStream.setDefaultStream(OldStream); %for versions 2010 and earlier
    %% Determinating fraction of contamination at each stage
    X(1,:)=Lambda(1,:)*(1-u(1)).*(1-initialX) + (1-Gamma(1,:)*u(1)).*initialX;
    for i= 2:n
        X(i,:)=Lambda(i,:)*(1-u(i)).*(1-X(i-1,:)) + (1-Gamma(i,:)*u(i)).*X(i-1,:);
    end

    %mu=mean(X,2);
    %sigma=std(X,0,2);
    limit=1-epsilon;
    %prob=normcdf(p,mu,sigma);
    %cost of contamination control
    fn=sum(cost.*u);
    %if sum(limit>=prob)==0,
    %    fprintf('\nGiven starting solution all contamination fractions are less than the threshold with probability > (1-epsilon)');
    %    fprintf('\nSuccessful with cost %4.2f.\n',sum(cost.*u));
    %else
    %    string=['\nGiven starting solution the probability that the contamination fractions \n'...
    %            'are less than the threshold is <= (1-epsilon) for at least one stage.\n' ...
    %            'Therefore the constraints are not satisfied.\n\n'...
    %            'Column 1 contains the probability that the contamination fractions are less than the threshold.\n' ...
    %            'Column 1 must be greater than column 2 (1-epsilon) for each stage in order for the constraint to be satsified.\n'];
    %    fprintf(string);
    %    results=[prob limit]
    %end
    %checking probability that Xi is <=pi is less than 1-eps
    con=zeros(n,runlength); %matrix of if Xi<pi for each trial and i
    for j=1:runlength
        con(:,j)= (X(:,j)<=p); 
    end
    con=con';
    le=sum(sum(con,2)==n);
    constraint=zeros(1,n);
    for k=1:n
        constraint(k)=(sum(con(:,k))/runlength)-limit(k);
    end
    ConstraintCov=cov(con); 

end
end