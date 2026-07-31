% predefined COST information

%% Options
COST.plant_life = 15; % plant life (year)
COST.construction_period = 2; % required period for plant construction (year)
COST.tax = 0.3892; % Income Tax = Federal = State
COST.irate = 0.15; % Nominal Interest Rate

%% In-Out parameter
% Inputs
% BaseIndex_2011 = 2250;
% BaseIndex_2022 = 3230;

BaseIndex_2011 = 585.7;  % CEPCI
BaseIndex_2016 = 541.7;
BaseIndex_2022 = 816; 
COST.BIOMASS_2016 = 0.06923; % $/kg ---> https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_63795.pdf p.16
COST.BIOMASS_2011 = 0.08; % $/kg. ---> from NREL report, Feedstock = $80/dry ton
EscalationFactor = BaseIndex_2022 / BaseIndex_2016;
COST.BIOMASS = COST.BIOMASS_2016 * EscalationFactor;
% COST.BIOMASS = 0.02837; % $/kg. ---> Pulpwood biomass price: $28.37 per metric ton. U.S. biomass feedstock prices by product 2023 | Statista

% COST.NG = 0.198; % $/kg ---> https://doi.org/10.1016/j.dib.2021.107514 | $4.18/MMBTU
% COST.NG = 0.1285;  % $/kg ---> Natural Gas Prices - Historical Chart_Macro Trends
% COST.NG = 0.2214;  % $/kg ---> price on Jan, 29th 2025 $3.89/MMBtu EIA https://www.eia.gov/naturalgas/weekly/
COST.NG_2011 = 0.0363; % $/kg ----> NREL report p.71, 76.57 ¢/MMBtu
COST.NG = COST.NG_2011 * EscalationFactor;

% h2o_density = 1000 ; % kg/m3
% COST.DIW = 0.2/h2o_density ; % $/kg

% Outputs for sale
COST.SAF = 1.1321; % $/kg https://doi.org/10.1038/s44286-023-00006-0 ---> 2023
% COST.SAF = 1.92; % $/kg % https://www.globalair.com/airport/region.aspx ----> SAF's price $9.07/gallon (24 Feb 2025)

% COST.Gasoline = 0.92; % $/kg from Boeun Kim, Shamik Misra, Christos T. Maravelias, Lignocellulosic fuel and chemical co-production, Joule,
COST.Gasoline = 1.22; % $/kg from NREL

% Volume 7, Issue 11, 2023, Pages 2403-2407, ISSN 2542-4351, https://doi.org/10.1016/j.joule.2023.10.002
COST.H2 = 1.3;
% COST.H2 = 3.34; % $/kg https://doi.org/10.1038/s44286-023-00006-0 ---> 2023

% COST.BIOMASS = 0.08; % $/kg. ---> from NREL report, Feedstock = $80/dry ton
% % COST.BIOMASS = 0.02837; % $/kg. ---> Pulpwood biomass price: $28.37 per metric ton. U.S. biomass feedstock prices by product 2023 | Statista
% 
% % COST.NG = 0.198; % $/kg ---> https://doi.org/10.1016/j.dib.2021.107514 | $4.18/MMBTU
% COST.NG = 0.0363; % $/kg ----> NREL report p.71, 76.57 ¢/MMBtu
% % COST.NG = 0.1285;  % $/kg ---> Natural Gas Prices - Historical Chart_Macro Trends
% % COST.NG = 0.2214;  % $/kg ---> price on Jan, 29th 2025 $3.89/MMBtu EIA https://www.eia.gov/naturalgas/weekly/
% 
% % h2o_density = 1000 ; % kg/m3
% % COST.DIW = 0.2/h2o_density ; % $/kg
% 
% % Outputs for sale
% % COST.SAF = 1.92; % $/kg % https://www.globalair.com/airport/region.aspx ----> SAF's price $9.07/gallon (24 Feb 2025)
% COST.Gasoline = 0.92; % $/kg from Boeun Kim, Shamik Misra, Christos T. Maravelias, Lignocellulosic fuel and chemical co-production, Joule,
% % Volume 7, Issue 11, 2023, Pages 2403-2407, ISSN 2542-4351, https://doi.org/10.1016/j.joule.2023.10.002
% COST.H2 = 1.3;

%% MACRS (Modified Accelerated Cost Recovery System) -> the current tax depreciation system in the USA
COST.MACRS_3 = [0.3333 0.4445 0.1481 0.0741];
COST.MACRS_5 = [0.2 0.32 0.192 0.1152 0.1152 0.0576];
COST.MACRS_7 = [0.1429 0.2449 0.1749 0.1249 0.0893 0.0892 0.0893 0.0446];
COST.MACRS_10 = [0.1 0.18 0.144 0.1152 0.0922 0.0737 0.0655 0.0655 0.0656 0.0655 0.0328];
COST.MACRS = COST.MACRS_7;
COST.UPF = 1 % scale up factor Aspen -> real process
COST.type = 4; % Reactor Type

%% Metal Cost ($/kg)
%%% catalyst for reference cell are anode (iridium) / cathode (platinum)
%%% cost from http://www.infomine.com (2019/01/22) $/kg
COST.metal.Pt = 25500;
COST.metal.Ir = 46940;
COST.metal.Al = 1.851;
COST.metal.Cu = 6.04287;
COST.metal.Au = 41290;
COST.metal.Ag = 490;
COST.metal.Zn = 2.57698;
COST.metal.Pb = 1.97203;
COST.metal.Ru = 8550;
COST.metal.Pd = 43180;
COST.metal.Ni = 11.61;
COST.metal.Co = 38.00;
COST.metal.Mo = 26.00;
COST.metal.Ti = 4.8; % https://www.metalary.com/titanium-price/
COST.metal.W = 30.3; % https://www.metalary.com/tungsten-price/ --> 2018 price.
%https://www.vanadiumprice.com/
COST.metal.V = 175.267498;

%% Catalyst Cost ($/kg)
% Required catalysts; dechlorination, ETP1, ETP2. (?)
COST.catalyst.year = 3; % year, replacement interval of major components
COST.catalyst.DECHLORIN = 415.2*30; % $/yr
COST.catalyst.ETP1 = 415.2*30; % $/yr
COST.catalyst.ETP2 = 415.2*30; % $/yr
% According to Perez-Fortes et al.(2016), Ru catalyst used 30kg/yr for
% 12kt/yr FA production plant.
% 415.2 $/kg (from Dr. Park).
% COST.catalyst = 415.2*30; % $/yr

%% Feed Cost ($/kg)
%the NETL reference case 10 for monoethanolamine (MEA) (550 MWe subcritical pulverized coal power plant),
%which is the baseline for most CO2 capture costs, capture cost

%% Utility Cost (reference: Seider, pg 604)
% Electricity --> industrial, similar as Korean
% COST.utility.electricity = 0.02;       % $/kW-hr. renewable energy 0.02.
% COST.utility.electricity = 0.07;       % $/kW-hr. current case.
COST.utility.electricity = 0.0585;       % $/kWh. NREL report p.71, 5.85 ¢/kWh
% COST.utility.electricity = 0.10;        % $/kW-hr PV electricity cost from IRENE
% Water (rho = 1000 kg/m3)
COST.utility.steam50psig = 1.9e-6 / 1000; % $/J, 125oC.
COST.utility.steam150psig = 2.2e-6 / 1000 ; % $/J, 175oC.
COST.utility.steam450psig = 2.5e-6 / 1000 ; % $/J, 250oC.
COST.utility.cw = 2.12e-7 / 1000; % $/J. 20oC.
COST.utility.Rchw = 4.40/2/10^9; % $/J - chilled water,40oF 4.44444oC. Add/2 from the value (to match Aspen value trend).
COST.utility.Rp10 = 2.74e-6 / 1000 ; % $/J. -25oC
COST.utility.Rn30 = 3.36e-6 / 1000 ; % $/J. -40oC
COST.utility.Rn90 = 5.88e-6 / 1000 ; % $/J. -65oC
COST.utility.Rn150 = 8.53e-6 / 1000 ; % $/J. -103oC.
COST.utility.Rn200 = 8.9e-6 / 1000 ; % $/J. -270oC. "Very low temperatures"

COST.utility.pw = 0.20/1000;        % $/m3 -> $/kg process water = COST.feed.EL
COST.utility.bfw = 0.50/1000;       % $/m3 -> $'kg boiler-feed water

% Wastewater treatment
COST.utility.wastewt = 0.33 * 1e-3;   % $/kg organic removed. Assume 1000ppm
% Need to check again. how much organic is in WWT?

% Operation
COST.utility.DWandB = 35;                    % $/operator-hr
COST.utility.workers = 10;                   % number of workers
COST.utility.TechAssist2Manufacturing = 60000; % $/(operator/shift)-yr
COST.utility.ControlLab = 65000;             % $/(operator/shift)-yr
COST.utility.shift = 3;                      % shift (assumption)
COST.utility.operating_hour = 8000;          % operating hour (hour/yr)