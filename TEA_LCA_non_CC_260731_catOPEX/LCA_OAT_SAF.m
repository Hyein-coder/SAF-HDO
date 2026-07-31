% LCA_OAT_SAF.m
% One-at-a-time (OAT) sensitivity of SAF life-cycle GWP on the 5 in-use LCA
% variables (see LCA_sensitivity_variables.md). Replaces the old 5-electricity-
% scenario structure: electricity grid factor is now ONE perturbed knob.
%
% Method: run Aspen ONCE to get physical flows, then perturb LCA emission
% factors analytically (no Aspen re-runs). GWP metric = allocated SAF
% intensity, identical formula to LCA_analysis_SAF.m (mass-basis baseline).
%
% Non-CC case only. Edit baseBkp + pct below.

clear; clc;

%% ---- config ----
simFolder = 'D:\SAF_Nurul\20260330_NewSim\';                 % edit to your machine
baseBkp   = '260401_pyrolysis_oil_CC_case_j_HOadjustment3_non_CC.bkp';
outXlsx   = fullfile(pwd, 'LCA_OAT_results.xlsx');
outFig    = fullfile(pwd, 'LCA_OAT_tornado.png');

% per-variable perturbation fraction (±). #1 biomass wider per literature.
pct.biomass = 0.30;   % GWP_biomass  (lit range ±20–50%)
pct.elec    = 0.20;   % EL_GRID
pct.ng      = 0.20;   % GWP_NG
pct.util    = 0.20;   % STEAM, COOL, WWT together
% #6 allocation is categorical: baseline mass-basis vs energy-basis (no ±%).

LHV_saf = 48.168;  % MJ/kg  (from LCA_analysis_SAF.m)
LHV_gas = 42.0;    % MJ/kg

pfad = fullfile(simFolder, baseBkp);
assert(isfile(pfad), 'Base .bkp not found: %s', pfad);

%% ---- run Aspen once, full pipeline (populates processData.LCA.* + Output) ----
fprintf('Running base case: %s\n', pfad);
h = actxserver('Apwn.Document');
invoke(h,'InitFromArchive2', pfad);
h.Engine.Reinit; h.Engine.Run2;

processData = struct(); processData.LCA = struct();
for f = {'elec','steam','ref','WF','NG'}, processData.LCA.(f{1}) = []; end

processData = accessData_SAF(processData, h);
unitCount   = size(processData.Units,1);
COST_PARAMETER; LCA_PARAMETER;
processData = CAPEX_SAF(processData, unitCount, COST, h);
processData = OPEX_SAF(processData, unitCount, COST, h);
processData = cash_flow_generic_SAF(processData, COST);
processData = LCA_analysis_SAF(processData, LCA);   % baseline reference
release(h);

%% ---- extract physical flows (invariant under EF perturbation) ----
P.elec = sum(processData.LCA.elec(:));            % kWh
P.steam= sum(processData.LCA.steam(:));           % MJ
P.ref  = sum(processData.LCA.ref(:));             % cooling qty
P.wwt  = processData.Output.WASTE_W.Massflow;     % kg
P.wf   = processData.Input.WF.Massflow;           % kg biomass
P.ng   = processData.Input.NG.Massflow;           % kg NG
P.msaf = processData.Output.SAF.Massflow;         % kg
P.mgas = processData.Output.Gasoline.Massflow;    % kg
P.emi  = sum(processData.LCA.emi(:));             % 0 in non-CC

% baseline emission-factor knobs
B.GWP_biomass = 0.1772;   % local const in LCA_analysis_SAF.m (not in LCA struct)
B.BM          = LCA.BM;
B.EL          = LCA.EL_GRID;
B.GWP_NG      = 0.2221;   % local const in LCA_analysis_SAF.m
B.NG_kg       = LCA.NG_kg;
B.STEAM       = LCA.STEAM;
B.COOL        = LCA.COOL;
B.WWT         = LCA.WWT;

%% ---- GWP model (pure fn of knobs) ----
% total burden [kgCO2eq/hr]; mirrors LCA_analysis_SAF.m S(k) with single EL
S = @(k) P.elec*k.EL + P.steam*k.STEAM + P.ref*k.COOL + P.wwt*k.WWT + ...
         (P.wf*k.GWP_biomass)*k.BM + (P.ng*k.GWP_NG)*k.NG_kg + P.emi;
% allocated SAF intensity. 'mass' = existing baseline (per SAF LHV, total mass);
% 'energy' = proper energy allocation across SAF+gasoline heating values.
Imass   = @(k) S(k) / ((P.msaf + P.mgas) * LHV_saf);
Ienergy = @(k) S(k) / (P.msaf*LHV_saf + P.mgas*LHV_gas);

I0 = Imass(B);   % baseline metric

% self-check: must equal LCA_analysis_SAF grid-scenario result S(1)
ref_alloc = processData.LCA_alloc_SAF(1);
assert(abs(I0 - ref_alloc)/ref_alloc < 1e-9, ...
    'baseline GWP (%.6g) != LCA_analysis_SAF grid (%.6g)', I0, ref_alloc);

%% ---- OAT loop ----
% each row: perturb one knob low/high, others at baseline
specs = {  % name, field(s), pct
  'Feedstock GWP (biomass)', {'GWP_biomass'}, pct.biomass
  'Electricity grid factor', {'EL'},          pct.elec
  'Natural gas GWP',         {'GWP_NG'},       pct.ng
  'Utility EFs (steam/cool/WWT)', {'STEAM','COOL','WWT'}, pct.util
};

names={}; Ilo=[]; Ihi=[]; lo_lbl={}; hi_lbl={};
for r = 1:size(specs,1)
    flds = specs{r,2}; p = specs{r,3};
    klo = B; khi = B;
    for j=1:numel(flds), klo.(flds{j})=B.(flds{j})*(1-p); khi.(flds{j})=B.(flds{j})*(1+p); end
    names{end+1,1}=specs{r,1};
    Ilo(end+1,1)=Imass(klo); Ihi(end+1,1)=Imass(khi);
    lo_lbl{end+1,1}=sprintf('-%.0f%%',p*100); hi_lbl{end+1,1}=sprintf('+%.0f%%',p*100);
end

% #6 allocation method: discrete energy-basis vs baseline mass-basis
names{end+1,1}='Allocation (energy vs mass)';
Ialt = Ienergy(B);
Ilo(end+1,1)=min(I0,Ialt); Ihi(end+1,1)=max(I0,Ialt);
lo_lbl{end+1,1}='energy/mass'; hi_lbl{end+1,1}='energy/mass';

%% ---- results table (sorted by swing) ----
dLo = Ilo - I0; dHi = Ihi - I0;
swing = max(abs(dLo),abs(dHi));
[~,ord] = sort(swing,'descend');

T = table(names(ord), lo_lbl(ord), hi_lbl(ord), Ilo(ord), Ihi(ord), ...
          dLo(ord), dHi(ord), 100*swing(ord)/I0, ...
    'VariableNames',{'Variable','Low','High','GWP_low','GWP_high', ...
                     'Delta_low','Delta_high','Swing_pct_of_base'});

fprintf('\nBaseline SAF GWP = %.6g (kgCO2eq per functional unit)\n', I0);
disp(T);
writetable(T, outXlsx, 'Sheet','OAT');
fprintf('Saved: %s\n', outXlsx);

%% ---- tornado plot ----
fig = figure('Visible','off','Position',[100 100 720 380]);
y = 1:height(T);
barh(y, T.Delta_low,  'FaceColor',[0.30 0.55 0.85]); hold on;
barh(y, T.Delta_high, 'FaceColor',[0.85 0.40 0.35]);
set(gca,'YTick',y,'YTickLabel',T.Variable,'YDir','reverse');
xline(0,'k-'); xlabel('\Delta GWP from baseline'); grid on;
title(sprintf('LCA OAT tornado (baseline GWP = %.4g)', I0));
legend({'low','high'},'Location','southeast');
exportgraphics(fig, outFig, 'Resolution',150); close(fig);
fprintf('Saved: %s\n', outFig);
