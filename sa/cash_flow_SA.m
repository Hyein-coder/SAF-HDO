function processData = cash_flow_SA(processData, COST, x)
% CASH_FLOW_SA  Sensitivity-analysis variant of cash_flow_generic_SAF_opt.
%   Identical economics to the published model, with three additions used
%   ONLY by the sensitivity study (all controlled by optional COST fields,
%   so the base case is reproduced exactly when they are absent):
%     1) COST.capex_mult   - scalar multiplier on installed equipment cost
%                            (C_EQP) for CAPEX sensitivity. Default 1.
%     2) COST.catalyst.C_inventory - catalyst inventory cost ($). Default
%                            59707.35 (the hard-coded value in the original).
%     3) Catalyst REPLACEMENT is actually charged as ANNUALIZED OPEX =
%        C_inventory / COST.catalyst.year, every operating year. Shorter
%        lifetime => higher annual cost => higher MSP, so lifetime is a live
%        lever. Enable with COST.catalyst.replace = true. Default false =>
%        original behaviour (lifetime has NO effect, matching as-published).
%     4) COST.contingency - contingency fraction of C_DPI (tech-maturity /
%        FOAK risk premium). Default 0.15 (as published). Emerging bioconversion
%        (AACE Class 5 / pioneer plant) -> 0.30-0.50.
%   SAF is priced at x (the decision variable) so the MSP solver can drive
%   NPV(x) = 0. See msp_npv_SA.m.
%
%   Field-robust: works for both carbon-capture and non-CC saved cases
%   (missing CapitalCost/OperatingCost/Output fields are treated as 0).

option.COSTSHEET = false;
option.FIGURE    = false;

% ---- optional SA levers (defaults reproduce the published model) --------
if isfield(COST,'capex_mult'),                 capex_mult = COST.capex_mult; else, capex_mult = 1.0; end
if isfield(COST,'catalyst') && isfield(COST.catalyst,'C_inventory'), C_catalyst = COST.catalyst.C_inventory; else, C_catalyst = 59707.35; end
if isfield(COST,'catalyst') && isfield(COST.catalyst,'replace'),     cat_replace = COST.catalyst.replace; else, cat_replace = false; end
if isfield(COST,'contingency'),                cont_frac  = COST.contingency; else, cont_frac  = 0.15; end

CC = processData.CapitalCost;
OC = processData.OperatingCost;

%% TOTAL SALES  (SAF priced at the decision variable x)
S(1) = processData.Output.SAF.Massflow      * COST.utility.operating_hour * x;
S(2) = processData.Output.Gasoline.Massflow * COST.utility.operating_hour * COST.Gasoline;

%% Total Capital Investment (TCI)
C_EQP = sumfield(CC,'reactor') + sumfield(CC,'flash2') + sumfield(CC,'flash3') + ...
    sumfield(CC,'compressor') + sumfield(CC,'distillation') + sumfield(CC,'pump') + ...
    sumfield(CC,'heatX') + sumfield(CC,'heater') + sumfield(CC,'separator') + ...
    sumfield(CC,'dryer') + sumfield(CC,'PSA') + sumfield(CC,'Capture');
C_EQP = C_EQP * capex_mult;                 % SA lever (default 1.0)

C_comp = 20000;                             % computers and software
C_TBM  = C_EQP + C_comp + C_catalyst;       % Total bare-module investment

C_site = 0.1 * C_TBM;                        % site preparation
C_serv = 0;                                  % service facilities

%% Allocated facility cost
index_ratio = 816 / 567;                     % 2022 / 2013
S_steam  = sumfield(OC,'STEAM')         / (COST.utility.steam150psig*1000*1993*38) / COST.utility.operating_hour * 0.454;
S_elec   = sumfield(OC,'ELECTRICITY')   / COST.utility.electricity / COST.utility.operating_hour * 0.001;
S_cw     = sumfield(OC,'REFRIGERATION') / COST.utility.pw          / COST.utility.operating_hour * 0.0044;
S_pw     = 0;
S_refrig = 0;

C_alloc_steam  = index_ratio * 930     * (S_steam^0.81);
C_alloc_elec   = index_ratio * 2900000 * (S_elec^0.83);
C_alloc_cw     = index_ratio * 1100    * (S_cw^0.68);
C_alloc_pw     = index_ratio * 1700    * (S_pw^0.96);
C_alloc_refrig = index_ratio * 12500   * (S_refrig^0.77);
sum_alloc      = C_alloc_steam + C_alloc_elec + C_alloc_cw + C_alloc_pw + C_alloc_refrig;
C_alloc        = 0.05*C_TBM + sum_alloc;

C_DPI     = C_TBM + C_site + C_serv + C_alloc;
C_cont    = cont_frac*C_DPI;                 % SA lever (default 0.15)
C_TDC     = C_DPI + C_cont;
C_land    = 0.02*C_TDC;
C_royal   = 0;
C_startup = 0.02*C_TDC;
C_TPI     = C_TDC + C_land + C_royal + C_startup;

%% Total production cost (annual)
C_feed(1) = processData.Input.WF.Massflow * COST.utility.operating_hour * COST.BIOMASS;
C_feed(2) = processData.Input.NG.Massflow * COST.utility.operating_hour * COST.NG;

C_util(1) = sumfield(OC,'STEAM');
C_util(2) = sumfield(OC,'ELECTRICITY');
C_util(3) = sumfield(OC,'REFRIGERATION');
C_util(4) = sumfield(OC,'WWT');
% Optional CO2-capture operating cost (only if this case has capture data)
if isfield(processData.Output,'CO2') && isfield(COST,'CO2_CAPTURE')
    C_util(5) = sum(processData.Output.CO2.Massflow * COST.utility.operating_hour * COST.CO2_CAPTURE);
end
if isfield(OC,'Capture') && isnumeric(OC.Capture)
    C_cap_op = OC.Capture;
else
    C_cap_op = 0;
end

C_O(1) = COST.utility.DWandB * COST.utility.workers * COST.utility.operating_hour;
C_O(2) = 0.15*C_O(1);
C_O(3) = 0.06*C_O(2);
C_O(4) = COST.utility.TechAssist2Manufacturing * COST.utility.workers / COST.utility.shift;
C_O(5) = COST.utility.ControlLab             * COST.utility.workers / COST.utility.shift;

C_M(1) = 0.035*C_TDC;
C_M(2) = 0.25*C_M(1);
C_M(3) = 1.00*C_M(1);
C_M(4) = 0.05*C_M(1);

MOSWB   = C_O(1) + C_O(2) + C_M(1) + C_M(2);
C_OH(1) = 0.071*MOSWB;
C_OH(2) = 0.024*MOSWB;
C_OH(3) = 0.059*MOSWB;
C_OH(4) = 0.074*MOSWB;

C_prop  = 0.02*C_TDC;

C_D(1) = 0.08*(C_TDC-1.18*C_alloc);
C_D(2) = 0.06*1.18*C_alloc;
C_D(3) = 0;
C_D(4) = 0;

COM = sum([C_feed C_util C_O C_M C_OH C_prop C_D]);

GE(1) = 0.03*sum(S) + 0.01*sum(S);
GE(2) = 0.048*sum(S);
GE(3) = 0.005*sum(S);
GE(4) = 0.02*sum(S);
GE(5) = 0.0125*sum(S);

%% CASH FLOWS
n = COST.plant_life;
CF_C_TDC = zeros(1,n); CF_C_WC = zeros(1,n); CF_C_D = zeros(1,n);
CF_C = zeros(1,n); CF_NetEarnings = zeros(1,n);
CF_DCF = zeros(1,n); CF_PV = zeros(1,n); CF_NPV = zeros(1,n);

% Catalyst replacement as annualized OPEX = inventory / lifetime (see README).
if cat_replace
    C_cat_annual = C_catalyst / COST.catalyst.year;
else
    C_cat_annual = 0;
end

for i = 1:n
    % --- Total production cost, with optional catalyst replacement -------
    oy = i - COST.construction_period;                 % operating year (>=1 once running)
    if oy >= 1
        C = COM + sum(GE) + C_cap_op + C_cat_annual;   % charge catalyst OPEX while running
    else
        C = COM + sum(GE) + C_cap_op;
    end

    C_WC = 0.0833*COM + 0.0192*sum(S) + 0.0833*sum(S) - 0.0833*sum(C_feed);

    if i <= COST.construction_period
        CF_C_TDC(i) = -C_TDC/COST.construction_period;
        if i == COST.construction_period
            CF_C_WC(i) = -C_WC;
        end
    end

    if i == COST.construction_period
        CF_C_D(i+1:i+length(COST.MACRS)) = sum(C_D)*COST.MACRS;
    end

    if i >= COST.construction_period+1
        CF_C(i)           = C - sum(C_D);
        CF_NetEarnings(i) = (sum(S) - CF_C(i) - CF_C_D(i)) .* (1-COST.tax);
    end

    if i == n
        CF_C_WC(i) = +C_WC;
    end
end

for i = 1:n
    CF_DCF(i) = (CF_NetEarnings(i) + CF_C_D(i)) + CF_C_TDC(i) + CF_C_WC(i);
    CF_PV(i)  = CF_DCF(i)/(1+COST.irate)^i;
    CF_NPV(i) = sum(CF_PV(1:i));
end

processData.NPV = CF_NPV(end);
processData.EconomicData.C_TDC  = C_TDC;
processData.EconomicData.C_TPI  = C_TPI;
processData.EconomicData.COM    = COM;
processData.EconomicData.C_feed = C_feed;
processData.EconomicData.S      = S;
end

%% ------------------------------------------------------------------------
function s = sumfield(st, f)
% Sum of a struct field, or 0 if the field is absent (CC vs non-CC cases).
if isfield(st, f)
    s = sum(st.(f)(:));
else
    s = 0;
end
end
