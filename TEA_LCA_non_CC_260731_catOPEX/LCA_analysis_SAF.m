function processData = LCA_analysis_SAF(processData, LCA)

GWP_biomass = 0.1772; % [kgCO2-eq/kg]
% GHG_NG = 77.1; % [g CO2e/MJ] ---> Life-Cycle Greenhouse Gas Emission Impacts of Forest Residue Preprocessing with Wet Milling (2023)
% HHV_NG = 55.50; % [MJ/kg] ---> https://en.wikipedia.org/wiki/Heat_of_combustion
% GWP_NG = (GHG_NG / 1000) * HHV_NG; % [kgCO2-eq/kg]
GWP_NG = 0.2221; % [kgCO2-eq/kg] ---> 0.021 tonnes CO2e/tonne (https://doi.org/10.1038/s43247-024-01988-2)

F(1) = processData.Input.WF.Massflow * GWP_biomass;
F(2) = processData.Input.NG.Massflow * GWP_NG;

% F(1) = processData.Input.WF.Massflow / (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);
% F(2) = processData.Input.NG.Massflow / (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);

try
    % O(1) = processData.Output.CO2_CO2CAPTURE.Massflow*0.5; % (0.5 = CO2 emitted per kg CO2 captured, kg/kg CO2) 
    O(1) = processData.Output.CO2_CO2CAPTURE.Massflow * 0;
    % O(1) = processData.Output.CO2_CO2CAPTURE.Massflow / (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);
  
catch
    O(1) = 0;
    O(2) = 0;
    % O(3) = 0;
end

processData.LCA.feed = F;
processData.LCA.emi = O;
% total CO2 emission/target product => LCA result

%% Scenarios
% processData.LCA.scenario.elec_GRID = processData.LCA.elec .* LCA.EL_GRID ./ (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);
% processData.LCA.scenario.elec_FUTURE = processData.LCA.elec .* LCA.EL_FUTURE ./ (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);
% processData.LCA.scenario.elec_NUCLEAR = processData.LCA.elec .* LCA.EL_NUCLEAR ./ (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);
% processData.LCA.scenario.elec_WIND = processData.LCA.elec .* LCA.EL_WIND ./ (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);
% processData.LCA.scenario.elec_PHOTO = processData.LCA.elec .* LCA.EL_PHOTO ./ (processData.Output.SAF.Massflow + processData.Output.Gasoline.Massflow);


processData.LCA.scenario.elec_GRID = processData.LCA.elec .* LCA.EL_GRID;
processData.LCA.scenario.elec_FUTURE = processData.LCA.elec .* LCA.EL_FUTURE;
processData.LCA.scenario.elec_NUCLEAR = processData.LCA.elec .* LCA.EL_NUCLEAR;
processData.LCA.scenario.elec_WIND = processData.LCA.elec .* LCA.EL_WIND;
processData.LCA.scenario.elec_PHOTO = processData.LCA.elec .* LCA.EL_PHOTO;

processData.LCA.scenario.steam = processData.LCA.steam .* LCA.STEAM; % [MJ/kg] * [kg/MJ]

processData.LCA.scenario.ref = processData.LCA.ref .* LCA.COOL;
processData.LCA.scenario.WWT = processData.Output.WASTE_W.Massflow * LCA.WWT;

processData.LCA.scenario.Feed.BM = F(1) * LCA.BM; % Biomass
processData.LCA.scenario.Feed.NG = F(2) * LCA.NG_kg;

processData.LCA.scenario.emi = processData.LCA.emi;

% processData.LCA.scenario.feed_grey = processData.LCA.feed .* [LCA.H2_GREY LCA.WATER LCA.CO2];
% processData.LCA.scenario.feed_green = processData.LCA.feed .* [LCA.H2_GREEN LCA.WATER LCA.CO2];

processData.LCA.S(1) = sum(processData.LCA.scenario.elec_GRID) + ...
    sum(processData.LCA.scenario.emi) + ...
    sum(processData.LCA.scenario.steam) + ...
    sum(processData.LCA.scenario.ref) + ...
    sum(processData.LCA.scenario.WWT) + ...
    sum(processData.LCA.scenario.Feed.BM) + ...
    sum(processData.LCA.scenario.Feed.NG);

processData.LCA.S(2) = sum(processData.LCA.scenario.elec_FUTURE) + ...
    sum(processData.LCA.scenario.emi) + ...
    sum(processData.LCA.scenario.steam) + ...
    sum(processData.LCA.scenario.ref) + ...
    sum(processData.LCA.scenario.WWT) + ...
    sum(processData.LCA.scenario.Feed.BM) + ...
    sum(processData.LCA.scenario.Feed.NG);

processData.LCA.S(3) = sum(processData.LCA.scenario.elec_NUCLEAR) + ...
    sum(processData.LCA.scenario.emi) + ...
    sum(processData.LCA.scenario.steam) + ...
    sum(processData.LCA.scenario.ref) + ...
    sum(processData.LCA.scenario.WWT) + ...
    sum(processData.LCA.scenario.Feed.BM) + ...
    sum(processData.LCA.scenario.Feed.NG);

processData.LCA.S(4) = sum(processData.LCA.scenario.elec_WIND) + ...
    sum(processData.LCA.scenario.emi) + ...
    sum(processData.LCA.scenario.steam) + ...
    sum(processData.LCA.scenario.ref) + ...
    sum(processData.LCA.scenario.WWT) + ...
    sum(processData.LCA.scenario.Feed.BM) + ...
    sum(processData.LCA.scenario.Feed.NG);

processData.LCA.S(5) = sum(processData.LCA.scenario.elec_PHOTO) + ...
    sum(processData.LCA.scenario.emi) + ...
    sum(processData.LCA.scenario.steam) + ...
    sum(processData.LCA.scenario.ref) + ...
    sum(processData.LCA.scenario.WWT) + ...
    sum(processData.LCA.scenario.Feed.BM) + ...
    sum(processData.LCA.scenario.Feed.NG);

ratio_SAF = processData.Output.SAF.Massflow/(processData.Output.Gasoline.Massflow + processData.Output.SAF.Massflow);

% processData.LCA_alloc_Gasoline = processData.LCA.S .* (1-ratio_SAF) / 42;
% processData.LCA_alloc_SAF = processData.LCA.S .* (ratio_SAF) / 48.168;

% processData.LCA_alloc_Gasoline = (processData.LCA.S .* (1-ratio_SAF) / 42)/(processData.Output.Gasoline.Massflow + processData.Output.SAF.Massflow);
% processData.LCA_alloc_SAF = (processData.LCA.S .* (ratio_SAF) / 48.168)/(processData.Output.Gasoline.Massflow + processData.Output.SAF.Massflow);

processData.LCA_alloc_Gasoline = (processData.LCA.S / 42)/(processData.Output.Gasoline.Massflow + processData.Output.SAF.Massflow);
processData.LCA_alloc_SAF = (processData.LCA.S / 48.168)/(processData.Output.Gasoline.Massflow + processData.Output.SAF.Massflow);