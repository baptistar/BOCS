%
% Bayesian Optimization of Combinatorial Structures
%
% Copyright (C) 2018 R. Baptista & M. Poloczek
%
% BOCS is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% BOCS is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License 
% along with BOCS.  If not, see <http://www.gnu.org/licenses/>.
%
% Copyright (C) 2018 MIT & University of Arizona
% Authors: Ricardo Baptista & Matthias Poloczek
% E-mails: rsb@mit.edu & poloczek@email.arizona.edu
%

clear; close all; clc
addpath(genpath('../stat_model'))
addpath(genpath('../test_problems'))
addpath(genpath('../tools'))
addpath(genpath('../plotting'))
addpath(genpath('../validation'))

% Find files for each run in Data folder
folder = 'ising';
files = dir(['../results/' folder '/test*.mat']);

% Set significant digits for mean/variance output
digs = '%5.6f';

% Make folder for figures
mkdir(['../results/' folder '/figures'])

% Find total number of data files
n_test = length(files);

% Load parameters
load(['../results/' folder '/' files(1).name],'inputs_t');
n_runs      = inputs_t.n_runs;
n_init      = inputs_t.n_init;
n_vars      = inputs_t.n_vars;
EvalBudget  = inputs_t.evalBudget;
lambda_vals = inputs_t.lambda_vals;
n_iter      = inputs_t.evalBudget - inputs_t.n_init;

% plotting definitions
pmean  = @(x) nanmean(x);
perbar = @(x) 1.96*nanstd(x)/sqrt(size(x,1));

% Process data for each lambda and alpha value
for l=1:length(lambda_vals)

    % declare cells to store data (or gap if plot_gap == 1)
    data_rnd        = zeros(n_test, n_iter);
    data_sa         = zeros(n_test, n_iter);
    data_bo         = zeros(n_test, n_iter);
    data_ols        = zeros(n_test, n_iter);
    data_smc        = zeros(n_test, n_iter);
    data_smac       = zeros(n_test, n_iter);
    
    data_bayesSA1   = zeros(n_test, n_iter);
    data_bayesSA2   = zeros(n_test, n_iter);
    data_bayesSA3   = zeros(n_test, n_iter);
    data_bayesSDP   = zeros(n_test, n_iter);
    
    data_mleSA1     = zeros(n_test, n_iter);
    data_mleSA2     = zeros(n_test, n_iter);
    data_mleSA3     = zeros(n_test, n_iter);
    data_mleSDP     = zeros(n_test, n_iter);

    data_hsSA1      = zeros(n_test, n_iter);
    data_hsSA2      = zeros(n_test, n_iter);
    data_hsSA3      = zeros(n_test, n_iter);
    data_hsSDP      = zeros(n_test, n_iter);
    
    % Load data from each test
    for t=1:n_test

        load(['../results/' folder '/' files(t).name]);

        % Define functions to process data
        extractObj  = @(c,l) cummin(cell2mat(cellfun(@(struct) struct.objVals, c(l,1), 'UniformOutput', 0)),1);
        extractTime = @(c,l) cumsum(cell2mat(cellfun(@(struct) struct.runTime, c(l,1), 'UniformOutput', 0)),1);

        % Extract and save data
        rnd_data           = extractObj(rnd, l);
        data_rnd(t,:)      = rnd_data(n_init+1:EvalBudget,:)';

        sa_data            = extractObj(sa, l);
        data_sa(t,:)       = sa_data(n_init+1:EvalBudget,:)';

        bo_data            = extractObj(bo, l);
        data_bo(t,:)       = bo_data(n_init+1:EvalBudget,:)';

        ols_data           = extractObj(ols, l);
        data_ols(t,:)      = ols_data(n_init+1:EvalBudget,:)';

        smc_data           = extractObj(smc, l);
        data_smc(t,:)      = smc_data(n_init+1:EvalBudget,:)';

        smac_data          = extractObj(smac, l);
        data_smac(t,:)     = smac_data(n_init+1:EvalBudget,:)';

        data_bayesSA1(t,:) = extractObj(bayes.stSA1, l);
        data_bayesSA2(t,:) = extractObj(bayes.stSA2, l);
        data_bayesSA3(t,:) = extractObj(bayes.stSA3, l);
        data_bayesSDP(t,:) = extractObj(bayes.sdp, l);

        data_mleSA1(t,:)   = extractObj(mle.stSA1, l);
        data_mleSA2(t,:)   = extractObj(mle.stSA2, l);
        data_mleSA3(t,:)   = extractObj(mle.stSA3, l);
        data_mleSDP(t,:)   = extractObj(mle.sdp, l);

        data_hsSA1(t,:)    = extractObj(hs.stSA1, l);
        data_hsSA2(t,:)    = extractObj(hs.stSA2, l);
        data_hsSA3(t,:)    = extractObj(hs.stSA3, l);
        data_hsSDP(t,:)    = extractObj(hs.sdp, l);

    end

    % Plot data with errorbars
    figure('visible','off')
    hold on

    H(1) =shadedErrorBar(1:n_iter, data_rnd,      {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);
    H(2) =shadedErrorBar(1:n_iter, data_sa,       {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);
    H(3) =shadedErrorBar(1:n_iter, data_bo,       {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);
    H(4) =shadedErrorBar(1:n_iter, data_ols,      {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);
    H(5) =shadedErrorBar(1:n_iter, data_smc,      {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);
    H(6) =shadedErrorBar(1:n_iter, data_smac,     {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);
    
    H(7) =shadedErrorBar(1:n_iter, data_bayesSA1, {pmean, perbar}, 'lineprops','-.','patchSaturation',0.2);
    H(8) =shadedErrorBar(1:n_iter, data_bayesSA2, {pmean, perbar}, 'lineprops','--','patchSaturation',0.2);
    H(9) =shadedErrorBar(1:n_iter, data_bayesSA3, {pmean, perbar}, 'lineprops',':','patchSaturation',0.2);
    H(10)=shadedErrorBar(1:n_iter, data_bayesSDP, {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);

    H(11)=shadedErrorBar(1:n_iter, data_mleSA1,   {pmean, perbar}, 'lineprops','-.','patchSaturation',0.2);
    H(12)=shadedErrorBar(1:n_iter, data_mleSA2,   {pmean, perbar}, 'lineprops','--','patchSaturation',0.2);
    H(13)=shadedErrorBar(1:n_iter, data_mleSA3,   {pmean, perbar}, 'lineprops',':','patchSaturation',0.2);
    H(14)=shadedErrorBar(1:n_iter, data_mleSDP,   {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);
    
    H(15)=shadedErrorBar(1:n_iter, data_hsSA1,    {pmean, perbar}, 'lineprops','-.','patchSaturation',0.2);
    H(16)=shadedErrorBar(1:n_iter, data_hsSA2,    {pmean, perbar}, 'lineprops','--','patchSaturation',0.2);
    H(17)=shadedErrorBar(1:n_iter, data_hsSA3,    {pmean, perbar}, 'lineprops',':','patchSaturation',0.2);
    H(18)=shadedErrorBar(1:n_iter, data_hsSDP,    {pmean, perbar}, 'lineprops','-','patchSaturation',0.2);

    xlabel('Iteration $t$')
    ylabel('Best $f(x_{t})$')
    xlim([1,n_iter])
    set(gca,'Yscale','log')
    legendflex([H.mainLine]', ...
        {'Random','SA', 'EI', 'OLS', 'PS', 'SMAC', ...
        'B.SA1',  'B.SA2',  'B.SA3', 'B.SDP', ...
        'MLE.SA1', 'MLE.SA2', 'MLE.SA3', 'MLE.SDP', ...
        'HS.SA1',  'HS.SA2', 'HS.SA3', 'HS.SDP'}, ...
        'ncol',6,'fontsize',12,'anchor',[2 6],'buffer',[0 10])
    z = get(gca,'position');
    z(4) = z(4) - 0.12;
    set(gca,'position',z);
    hold off
    lambda_str = strrep(num2str(lambda_vals(l)),'.','p');
    print('-depsc',['../results/' folder '/figures/' folder '_lambda' lambda_str]);

    % Print results
    fprintf(['lambda = ' num2str(lambda_vals(l)) '\n']);
    fprintf(['Rnd       = ' digs ' , ' digs ' \n'], pmean(data_rnd(:,end)),      perbar(data_rnd(:,end)));
    fprintf(['SA        = ' digs ' , ' digs ' \n'], pmean(data_sa(:,end)),       perbar(data_sa(:,end)));
    fprintf(['BO        = ' digs ' , ' digs ' \n'], pmean(data_bo(:,end)),       perbar(data_bo(:,end)));
    fprintf(['OLS       = ' digs ' , ' digs ' \n'], pmean(data_ols(:,end)),      perbar(data_ols(:,end)));
    fprintf(['SMC       = ' digs ' , ' digs ' \n'], pmean(data_smc(:,end)),      perbar(data_smc(:,end)));
    fprintf(['SMAC      = ' digs ' , ' digs ' \n'], pmean(data_smac(:,end)),     perbar(data_smac(:,end)));

    fprintf(['Bayes.SA1 = ' digs ' , ' digs ' \n'], pmean(data_bayesSA1(:,end)), perbar(data_bayesSA1(:,end)));
    fprintf(['Bayes.SA2 = ' digs ' , ' digs ' \n'], pmean(data_bayesSA2(:,end)), perbar(data_bayesSA2(:,end)));
    fprintf(['Bayes.SA3 = ' digs ' , ' digs ' \n'], pmean(data_bayesSA3(:,end)), perbar(data_bayesSA3(:,end)));
    fprintf(['Bayes.SDP = ' digs ' , ' digs ' \n'], pmean(data_bayesSDP(:,end)), perbar(data_bayesSDP(:,end)));

    fprintf(['MLE.SA1   = ' digs ' , ' digs ' \n'], pmean(data_mleSA1(:,end)),   perbar(data_mleSA1(:,end)));
    fprintf(['MLE.SA2   = ' digs ' , ' digs ' \n'], pmean(data_mleSA2(:,end)),   perbar(data_mleSA2(:,end)));
    fprintf(['MLE.SA3   = ' digs ' , ' digs ' \n'], pmean(data_mleSA3(:,end)),   perbar(data_mleSA3(:,end)));
    fprintf(['MLE.SDP   = ' digs ' , ' digs ' \n'], pmean(data_mleSDP(:,end)),   perbar(data_mleSDP(:,end)));

    fprintf(['HS.SA1    = ' digs ' , ' digs ' \n'], pmean(data_hsSA1(:,end)),    perbar(data_hsSA1(:,end)));
    fprintf(['HS.SA2    = ' digs ' , ' digs ' \n'], pmean(data_hsSA2(:,end)),    perbar(data_hsSA2(:,end)));
    fprintf(['HS.SA3    = ' digs ' , ' digs ' \n'], pmean(data_hsSA3(:,end)),    perbar(data_hsSA3(:,end)));
    fprintf(['HS.SDP    = ' digs ' , ' digs ' \n'], pmean(data_hsSDP(:,end)),    perbar(data_hsSDP(:,end)));

end

% -- END OF FILE --
