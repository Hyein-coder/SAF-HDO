function processData = cash_flow_generic_SAF_opt(processData, COST, x)
option.COSTSHEET = false;
option.FIGURE = false;

%% TOTAL SALES
S(1) = processData.Output.SAF.Massflow * COST.utility.operating_hour * x;
S(2) = processData.Output.Gasoline.Massflow * COST.utility.operating_hour * COST.Gasoline;

%% Components of Total Capital Investment (TCI)
% Bare module cost can be calculated by Bare-Module Factors of Guthrie (1974) pg. 549 of Seider book.
% Total bare-module costs for fabricated equipment + Total bare-module costs for process machinery + Total bare-module costs for spares + Total bare-module costs for storage and surge tanks.
% Costs for separator/thickener/filter/dryer/initial input/electrolyzer
% added (22/01/27. electrolyzer and initial input were added first).
C_EQP = sum(processData.CapitalCost.reactor) + ...
    sum(processData.CapitalCost.flash2) + ...       % flash
    sum(processData.CapitalCost.flash3) + ...       
    sum(processData.CapitalCost.compressor) + ...   % compressor
    sum(processData.CapitalCost.distillation) + ... % distillation
    sum(processData.CapitalCost.pump) + ...         % extraction
    sum(processData.CapitalCost.heatX) + ...        % pressure swing adsorption
    sum(processData.CapitalCost.heater) + ...       % heater
    sum(processData.CapitalCost.separator) + ...    % separation vessels
    sum(processData.CapitalCost.dryer)  + ...     % dryer costs
    sum(processData.CapitalCost.PSA);
% + ...
    % sum(processData.CapitalCost.Capture);

% C_catalyst = 5000; % Total cost for initial catalyst charges + electrolyte
C_catalyst = 59707.35; %$
C_comp = 20000; % Total bare-module costs for computers and software
% C_TBM = C_EQP + C_comp; % + C_catalyst % Total bare-module investment, TBM
C_TBM = C_EQP + C_comp + C_catalyst; % Total bare-module investment, TBM

C_site = 0.1 * C_TBM; % Cost of site preparation (10-20% C_TBM for new plant, 4-6% for existing integrated complex)
C_serv = 0; % Cost of service facilities

%% Section for allocated facility cost calculation
% C_alloc     = 0; % Allocated costs for utility plants and related facilities

% Capital costs for allocated facility
% index_ratio = 596.2 / 567;
index_ratio = 816 / 567; % 2022 / 2013
% assume 150psig steam for steam cost
S_steam     = sum(processData.OperatingCost.STEAM) / (COST.utility.steam150psig*1000*1993*38) / COST.utility.operating_hour * 0.454; % kg/hr to lb/hr
S_elec      = sum(processData.OperatingCost.ELECTRICITY) / COST.utility.electricity / COST.utility.operating_hour * 0.001; % kW to MW
S_cw        = sum(processData.OperatingCost.REFRIGERATION) / COST.utility.pw / COST.utility.operating_hour * 0.0044; % kg/hr to gal/min
% S_pw      = processData.Input.H2O.Massflow * 0.0044; % kg/hr to gal/min
S_pw        = 0;
S_refrig    = 0;

C_alloc_steam   = index_ratio * 930 * (S_steam^0.81); % S in lb/hr, steam facility
C_alloc_elec    = index_ratio * 2900000 * (S_elec^0.83); % S in MW, electricity facility
C_alloc_cw      = index_ratio * 1100 * (S_cw^0.68); % S in gpm, cooling water facility
C_alloc_pw      = index_ratio * 1700 * (S_pw^0.96); % S in gpm, cooling water facility
C_alloc_refrig  = index_ratio * 12500 * (S_refrig^0.77); % S in tons
sum_alloc       = C_alloc_steam + C_alloc_elec + C_alloc_cw + C_alloc_pw + C_alloc_refrig;

C_alloc = 0.05*C_TBM + sum_alloc; % allocated facility costs

%%
C_DPI       = C_TBM + C_site + C_serv + C_alloc; % Total of direct permanent investment, DPI

C_cont      = (0.15)*C_DPI; % Cost of contigencies and contractor's fee (Guthrie add 3%, total 18% but 15% is common)
C_TDC       = C_DPI + C_cont; % Total depreciable capital, TDC

C_land      = 0.02*C_TDC; % Cost of land
C_royal     = 0; % If there is loyalty then = 0.02*C_TDC and 0.03*sum(S) annually; % Cost of royalties
C_startup   = 0.02*C_TDC; % Cost of Plant startup
C_TPI       = C_TDC + C_land + C_royal + C_startup; % Total permanent investment, TPI

% Working Capital and Total Capital Investment will be calculated after production cost

%% Total Production Cost (C), annual base
C_feed(1) = processData.Input.WF.Massflow * COST.utility.operating_hour * COST.BIOMASS;
C_feed(2) = processData.Input.NG.Massflow * COST.utility.operating_hour * COST.NG;

%% 
% Utilities
try
    C_util(1) = sum(processData.OperatingCost.STEAM); % Steam
    C_util(2) = sum(processData.OperatingCost.ELECTRICITY); % electricity
    C_util(3) = sum(processData.OperatingCost.REFRIGERATION); % refrigeration
    C_util(4) = sum(processData.OperatingCost.WWT); % Waste water treatment
    C_util(5) = sum(processData.Output.CO2.Massflow * COST.utility.operating_hour * COST.CO2_CAPTURE);
    C_util(6) = processData.OperatingCost.Capture;
catch
    C_util(1) = sum(processData.OperatingCost.STEAM); % Steam
    C_util(2) = sum(processData.OperatingCost.ELECTRICITY); % electricity
    C_util(3) = sum(processData.OperatingCost.REFRIGERATION); % refrigeration
    C_util(4) = sum(processData.OperatingCost.WWT); % Waste water treatment
end

% Catalyst OPEX (annualized replacement = C_catalyst / catalyst lifetime).
% Previously computed in OPEX_SAF but excluded from C_util/COM; now included.
C_util(end+1) = processData.OperatingCost.Catalyst; % catalyst replacement

try
    C_cap_op = processData.OperatingCost.Capture;
    if ~isnumeric(C_cap_op)
        C_cap_op = 0;
    end
catch
    C_cap_op = 0;
end

% Operations (labor related) (O)
C_O(1) = COST.utility.DWandB * COST.utility.workers * COST.utility.operating_hour; % Direct wages and benefits (DW&B)
C_O(2) = 0.15*C_O(1); % Direct salaries and benefits    15% of DW&B
C_O(3) = 0.06*C_O(2); % Operating supplies and services 6% of DW&B
C_O(4) = COST.utility.TechAssist2Manufacturing * COST.utility.workers / COST.utility.shift; % Technical assistance to manufacturing
C_O(5) = COST.utility.ControlLab * COST.utility.workers / COST.utility.shift; % Control laboratory

% Maintenance
C_M(1) = 0.035*C_TDC; % Wages and benefits (MW&B) Fluid handling process 3.5% of C_{TDC}
C_M(2) = 0.25*C_M(1); % Salaries and benefits   25% of MW&B
C_M(3) = 1.00*C_M(1); % Materials and services  100% of MW&B
C_M(4) = 0.05*C_M(1); % Maintenance overhead    5% of MW&B

% Operating overhead
MOSWB = C_O(1) + C_O(2) + C_M(1) + C_M(2); % M&O-SW&B = DW&B = Direct salaries and benefits + MW&B + Maintenance and benefits
C_OH(1) = 0.071*MOSWB; % General plant overhead             7.1% of M&O-SW&B
C_OH(2) = 0.024*MOSWB; % Mechanical department services     2.4% of M&O-SW&B
C_OH(3) = 0.059*MOSWB; % Employee relations department      5.9% of M&O-SW&B
C_OH(4) = 0.074*MOSWB; % Business services                  7.4% of M&O-SW&B

% Property taxes and insurance    2% of C_{TDC}
C_prop = 0.02*C_TDC;

% Depreciation
C_D(1) = 0.08*(C_TDC-1.18*C_alloc); % Direct plant      8% of (C_{TDC}-1.18*C_{alloc})
C_D(2) = 0.06*1.18*C_alloc; % Allocated plant           6% of 1.18C_{alloc}
C_D(3) = 0; % Rental fees (office and lab space)
C_D(4) = 0; % Licensing fees

% COST OF MANUFACTURES (COM) sum of above (1)direct manufacturing cost (feedstocks, utilities, O, M); (2)operating overhead; and (3)fixed costs (property, taxes, insurance and depreciation)
COM = sum([C_feed C_util C_O C_M C_OH C_prop C_D]);
% COM_sub = [C_feed C_util C_O C_M C_OH C_prop C_D]
% COM
% GENERAL EXPENSES
GE(1) = 0.03*sum(S) + 0.01*sum(S); % Selling (or transfer) expense  3% (1%) of sales
GE(2) = 0.048*sum(S); % Direct research                             4.8% of sales
GE(3) = 0.005*sum(S); % Allocated research                          0.5% of sales
GE(4) = 0.02*sum(S); % Administrative expense                       2.0% of sales
GE(5) = 0.0125*sum(S); % Management incentive compensation          1.25%of sales
% GE

%% CASH FLOWS

CF_C_TDC = zeros(1, COST.plant_life);
CF_C_WC = zeros(1, COST.plant_life);
CF_C_land = zeros(1, COST.plant_life);
CF_C_startup = zeros(1, COST.plant_life);
CF_C_royal = zeros(1, COST.plant_life);
CF_S_equip = zeros(1, COST.plant_life);
CF_C_D = zeros(1, COST.plant_life);
CF_C = zeros(1, COST.plant_life);
CF_NetEarnings = zeros(1, COST.plant_life);
CF_DCF = zeros(1, COST.plant_life);
CF_PV = zeros(1, COST.plant_life);
CF_NPV = zeros(1, COST.plant_life);

for i = 1:COST.plant_life
    
    % C = COM + sum(GE);
    if mod(i-COST.construction_period-1,COST.catalyst.year) == 0
        % for replacement interval of major components of electrolyzer
        % TOTAL PRODUCTION COST (C)

        C = COM + sum(GE) + C_cap_op; % from Colella et al. (2014)
    else
        % for normal operation
        % TOTAL PRODUCTION COST (C)
        C = COM + sum(GE) + C_cap_op;
    end

    % Working capital and total capital investment
    C_WC    = 0.0833*COM+0.0192*sum(S)+0.0833*sum(S)-0.0833*sum(C_feed); % Working capital = cash reserves + inventory + accounts receivable - acounts payable
    C_TCI   = C_TPI + C_WC; % Total capital investment, TCI

    % Under construction
    if i<=COST.construction_period
        CF_C_TDC(i) = -C_TDC/COST.construction_period;
        if i==COST.construction_period
            CF_C_WC(i) = -C_WC;
        end
    end
    
    % Depreciation with MACRS
    if i==COST.construction_period
        CF_C_D(i+1:i+length(COST.MACRS)) = sum(C_D)*COST.MACRS;
    end

    % After construction
    if i>=COST.construction_period+1
        CF_C(i) = C - sum(C_D); % Cost excluded Depreciation cost
        CF_NetEarnings(i) = (sum(S) - CF_C(i) - CF_C_D(i)).*(1-COST.tax);
    end

    % End of the plant
    if i == COST.plant_life
        CF_C_WC(i) = +C_WC;
    end

end

% Discounted Cash Flow
for i = 1:COST.plant_life
    CF_DCF(i)   = (CF_NetEarnings(i) + CF_C_D(i)) + CF_C_TDC(i) + CF_C_WC(i);
    CF_PV(i)    = CF_DCF(i)/(1+COST.irate)^i;
    CF_NPV(i)   = sum(CF_PV(1:i));
end

% Write to the COSTSHEET (option.COSTSHEET = true)
if option.COSTSHEET
    filename = 'COSTSHEET.xlsx';
    processData = [CF_C_TDC' CF_C_WC' CF_C_land' CF_C_startup' CF_C_royal' CF_S_equip' ...
        CF_C_D' CF_C' sum(S)*ones(COST.plant_life,1) CF_NetEarnings' CF_DCF' CF_PV' CF_NPV'];
    sheet = 3;
    xlRange = 'C3';
    xlswrite(filename,processData,sheet,xlRange)
end

if option.FIGURE
    figure(10)
    plot (CF_DCF,'Linewidth',1.2,'Marker','o');
    hold on
    plot (CF_PV,'Linewidth',1.2,'Marker','o');
    plot (CF_NPV,'Linewidth',1.2,'Marker','o');
    legend({'DCF','PV','NPV'})
    ylabel('Metrics ($)')
    xlabel('Operating time (year)')
    set(gca,'linewidth',1,'layer','top')
    axis square
    grid on
    grid minor
end
CASHFLOW = [CF_C_TDC' CF_C_WC' CF_C_land' CF_C_startup' CF_C_royal' CF_S_equip' ...
    CF_C_D' CF_C' sum(S)*ones(COST.plant_life,1) CF_NetEarnings' CF_DCF' CF_PV' CF_NPV'];

processData.NPV = (CF_NPV(end));
processData.CASHFLOW = CASHFLOW;
processData.EconomicData.C_EQP = C_EQP;
processData.EconomicData.C_comp = C_comp;
processData.EconomicData.C_site = C_site;
processData.EconomicData.C_serv = C_serv;
processData.EconomicData.C_alloc = C_alloc;
processData.EconomicData.C_DPI = C_DPI;
processData.EconomicData.C_cont = C_cont;
processData.EconomicData.C_TDC = C_TDC;
processData.EconomicData.C_land = C_land;
processData.EconomicData.C_royal = C_royal;
processData.EconomicData.C_startup = C_startup;
processData.EconomicData.C_TPI = C_TPI;
processData.EconomicData.C_feed = C_feed;
processData.EconomicData.C_util = C_util;
processData.EconomicData.C_O = C_O;
processData.EconomicData.C_M = C_M;
processData.EconomicData.C_OH = C_OH;
processData.EconomicData.C_prop = C_prop;
processData.EconomicData.C_D = C_D;
processData.EconomicData.C_COM = COM;
processData.EconomicData.GE = GE;
processData.EconomicData.C = C;
processData.EconomicData.C_WC = C_WC;
processData.EconomicData.C_TCI = C_TCI;
processData.EconomicData.S = S;
processData.EconomicData.C_catalyst = C_catalyst;
% processData.EconomicData.CF_C_D = sum(CF_C_D);
% processData.EconomicData.CF_C = sum(CF_C);
% processData.EconomicData.S_tot = S * COST.plant_life;

end