%% SA_driver.m
%  Additional techno-economic sensitivity analysis for the SAF manuscript,
%  responding to CEJ reviewer requests (catalyst lifetime, hydrogen price,
%  feedstock/pyrolysis-oil cost, and general TEA robustness).
%
%  Metric:   Minimum Selling Price of SAF, MSP ($/kg) = SAF price making NPV=0.
%  Engine:   reuses the published cash-flow model (frozen Aspen results in
%            processData) via cash_flow_SA.m + msp_npv_SA.m. No Aspen re-run.
%
%  RUN (headless):   matlab -batch "cd('D:\SAF_Nurul\saf_nurul_agent\sa'); SA_driver"
%  or in the MATLAB GUI:  open this folder and press Run.
%
%  Outputs (written next to this script):
%     sa_oat.csv       full one-at-a-time sweeps (parameter, value, MSP)
%     sa_tornado.csv   low/high MSP per parameter for the tornado chart
%  Then: python plot_sa.py   ->  tornado.png, spider.png
% -------------------------------------------------------------------------
clear; clc;

%% ---- Configuration (edit paths/case here) ------------------------------
CREATE_DIR = 'D:\SAF_Nurul\Create_251104_2\Create_251104_2';
BASE_MAT   = fullfile(CREATE_DIR, 'matlabBC.mat');   % converged base-case workspace
SA_DIR     = fileparts(mfilename('fullpath'));
addpath(SA_DIR); addpath(CREATE_DIR);

% Include catalyst replacement cost in ALL runs (see note in README).
% The published model charges NO catalyst replacement, so lifetime has no
% effect there. Set true so catalyst lifetime is a live lever.
USE_CATALYST_REPLACEMENT = true;

%% ---- Load frozen base case --------------------------------------------
S = load(BASE_MAT, 'processData', 'COST', 'msp');
processData = S.processData;
baseCOST    = S.COST;
if isfield(S,'msp') && isscalar(S.msp), x0 = S.msp; else, x0 = 1.2; end

% Ensure catalyst inventory field exists (matches the hard-coded $59,707.35)
if ~isfield(baseCOST,'catalyst') || ~isfield(baseCOST.catalyst,'C_inventory')
    baseCOST.catalyst.C_inventory = 59707.35;
end
baseCOST.capex_mult = 1.0;

%% ---- Base MSP: published vs SA baseline -------------------------------
cost_pub = baseCOST; cost_pub.catalyst.replace = false;
MSP_published = solve_msp(processData, cost_pub, x0);

baseCOST.catalyst.replace = USE_CATALYST_REPLACEMENT;
MSP_base = solve_msp(processData, baseCOST, x0);
x0 = MSP_base;   % warm start

fprintf('\n================ BASE-CASE MSP ================\n');
fprintf('  Published model (no catalyst replacement): %.4f $/kg\n', MSP_published);
fprintf('  SA baseline (catalyst replacement = %d):    %.4f $/kg\n', USE_CATALYST_REPLACEMENT, MSP_base);
fprintf('  delta from costing catalyst replacement:    %+.4f $/kg\n', MSP_base - MSP_published);

%% ---- Parameter definitions --------------------------------------------
% Each row: display name | override key | base value | grid (row vector)
% Grids are chosen from the model's own base values and literature bounds.
bBIO = baseCOST.BIOMASS;  bNG = baseCOST.NG;
P = {
%   name                         key              base            grid
   'Feedstock cost ($/kg)'      ,'BIOMASS'       , bBIO          , bBIO*[0.50 0.75 1.00 1.25 1.50]
   'Catalyst lifetime (yr)'     ,'catalyst_year' , baseCOST.catalyst.year , [1 2 3 4 5]
   'Discount rate (-)'          ,'irate'         , baseCOST.irate, [0.08 0.10 0.12 0.15 0.18 0.20]
   'Plant life (yr)'            ,'plant_life'    , baseCOST.plant_life , [10 12 15 18 20]
   'CAPEX multiplier (-)'       ,'capex_mult'    , 1.0           , [0.70 0.85 1.00 1.15 1.30]
   'Natural gas price ($/kg)'   ,'NG'            , bNG           , bNG*[0.50 0.75 1.00 1.25 1.50]
};

%% ---- One-at-a-time sweeps ---------------------------------------------
oat = {};   % rows: {name, value, msp}
for k = 1:size(P,1)
    name = P{k,1}; key = P{k,2}; grid = P{k,4};
    for v = grid
        c   = apply_override(baseCOST, key, v);
        msp = solve_msp(processData, c, x0);
        oat(end+1,:) = {name, v, msp}; %#ok<SAGROW>
        fprintf('  %-26s %10.4g  ->  MSP = %.4f $/kg\n', name, v, msp);
    end
end

% write sa_oat.csv
fid = fopen(fullfile(SA_DIR,'sa_oat.csv'),'w');
fprintf(fid,'parameter,value,msp\n');
for r = 1:size(oat,1)
    fprintf(fid,'%s,%.10g,%.10g\n', oat{r,1}, oat{r,2}, oat{r,3});
end
fclose(fid);

%% ---- Tornado (low/high vs base) ---------------------------------------
% Low/high = worst/best plausible for each parameter (first/last grid point,
% except catalyst lifetime where SHORT life = higher cost).
fid = fopen(fullfile(SA_DIR,'sa_tornado.csv'),'w');
fprintf(fid,'parameter,base_msp,low_value,low_msp,high_value,high_msp\n');
for k = 1:size(P,1)
    key = P{k,2}; grid = P{k,4};
    lo = grid(1); hi = grid(end);
    msp_lo = solve_msp(processData, apply_override(baseCOST,key,lo), x0);
    msp_hi = solve_msp(processData, apply_override(baseCOST,key,hi), x0);
    fprintf(fid,'%s,%.10g,%.10g,%.10g,%.10g,%.10g\n', ...
        P{k,1}, MSP_base, lo, msp_lo, hi, msp_hi);
end
fclose(fid);

fprintf('\nWrote sa_oat.csv and sa_tornado.csv in %s\n', SA_DIR);
fprintf('Next:  python plot_sa.py\n');

%% ======================= local functions ===============================
function msp = solve_msp(processData, COST, x0)
    f = @(x) msp_npv_SA(processData, COST, x);
    try
        msp = fzero(f, x0);
    catch
        msp = fzero(f, [0.1 20]);   % fall back to a bracket
    end
end

function c = apply_override(c, key, v)
    switch key
        case 'catalyst_year'
            c.catalyst.year    = v;
            c.catalyst.replace = true;      % lifetime only matters if replacement is charged
        case 'plant_life'
            c.plant_life = round(v);
        otherwise
            c.(key) = v;                    % H2, BIOMASS, NG, irate, capex_mult, ...
    end
end
