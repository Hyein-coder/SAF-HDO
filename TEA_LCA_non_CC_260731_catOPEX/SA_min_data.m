% % clear; clc;
% % 
% % simFolder = 'D:\523506\A\Create_251120_SA\Simulation\sensitivity_20251121\grid_20251121_173949_param_0\converged\';
% % list_simulations = {
% %     'caseSA_0.bkp';
% %     'caseSA_1.bkp';
% %     'caseSA_2.bkp';
% %     'caseSA_3.bkp';
% %     'caseSA_5.bkp';
% %     'caseSA_6.bkp';
% %     'caseSA_7.bkp';
% %     'caseSA_8.bkp';
% %     'caseSA_9.bkp';
% %     'caseSA_10.bkp';
% %     };
% % 
% % % list_simulations = {
% % %     '251122_pyrolysis_oil_CC_case_a_run.bkp';
% % %     '251122_pyrolysis_oil_CC_case_i_run.bkp';
% % %     '251122_pyrolysis_oil_CC_case_f_run.bkp';
% % %     };
% % nSim = length(list_simulations);
% % outputFile = fullfile(simFolder,'SAF_Sensitivity_Analysis_251122_param_0.xlsx');
% % 
% % % TEA
% % TEA_all = struct();
% % 
% % for sim = 1:nSim
% %     pfad = fullfile(simFolder, list_simulations{sim});
% %     fprintf('Processing simulation (TEA): %s\n', pfad);
% % 
% %     h = actxserver('Apwn.Document');
% %     invoke(h,'InitFromArchive2', pfad);
% %     h.Engine.Reinit;
% %     h.Engine.Run2;
% % 
% %     clear processData
% %     processData = accessData_SAF([], h);
% %     unitCount = size(processData.Units,1);
% % 
% %     if ~isfield(processData, 'LCA'); processData.LCA = struct(); end
% %     if ~isfield(processData.LCA,'elec');  processData.LCA.elec = []; end
% %     if ~isfield(processData.LCA,'steam'); processData.LCA.steam = []; end
% %     if ~isfield(processData.LCA,'ref');   processData.LCA.ref = []; end
% %     if ~isfield(processData.LCA,'WF');    processData.LCA.WF = []; end
% %     if ~isfield(processData.LCA,'NG');    processData.LCA.NG = []; end
% % 
% %     COST_PARAMETER
% %     LCA_PARAMETER
% % 
% %     processData = CAPEX_SAF(processData, unitCount, COST, h);
% %     processData = OPEX_SAF(processData, unitCount, COST, h);
% %     processData = cash_flow_generic_SAF(processData, COST);
% %     processData = LCA_analysis_SAF(processData, LCA);
% % 
% %     fun = @(x)msp_SAF(processData, COST, x);
% %     msp = fsolve(fun, 3);
% %     processData_msp = cash_flow_generic_SAF_opt(processData, COST, msp);
% % 
% %     TEA_all(sim).MSP = msp;
% % 
% %     try TEA_all(sim).SAF_Production = h.Tree.FindNode("\Data\Streams\326\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).SAF_Production = NaN; end
% %     try TEA_all(sim).Gasoline_Production = h.Tree.FindNode("\Data\Streams\324\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).Gasoline_Production = NaN; end
% %     try TEA_all(sim).NG_Feed = h.Tree.FindNode("\Data\Streams\409\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).NG_Feed = NaN; end
% %     try
% %         H2_in = h.Tree.FindNode("\Data\Streams\206\Output\MASSFLOW\MIXED\H2").value;
% %         H2_out = h.Tree.FindNode("\Data\Streams\215\Output\MASSFLOW\MIXED\H2").value;
% %         TEA_all(sim).H2_Consumption = H2_in - H2_out;
% %     catch
% %         TEA_all(sim).H2_Consumption = NaN;
% %     end
% % 
% %     try
% %         TEA_all(sim).CO2_Capture = h.Tree.FindNode("\Data\Streams\426\Output\MASSFLMX\$TOTAL").value;
% %     catch
% %         TEA_all(sim).CO2_Capture = NaN;
% %     end
% % 
% %     % LG generation in HDP = sum of components on stream 215
% %     try
% %         compList = {'CO2','CH4','C2H4','C2H6','C3H6','C3H8','C4H10','NH3'};
% %         sumLG = 0;
% %         for ci = 1:length(compList)
% %             try
% %                 nodePath = sprintf('\\Data\\Streams\\215\\Output\\MASSFLOW\\MIXED\\%s', compList{ci});
% %                 val = h.Tree.FindNode(nodePath).value;
% %                 if isnumeric(val), sumLG = sumLG + val; end
% %             catch
% %                 % component missing -> treat as 0
% %             end
% %         end
% %         TEA_all(sim).LG_HDP = sumLG;
% %     catch
% %         TEA_all(sim).LG_HDP = NaN;
% %     end
% % 
% %     try TEA_all(sim).CH4_HDP = h.Tree.FindNode("\Data\Streams\215\Output\MASSFLOW\MIXED\CH4").value; catch, TEA_all(sim).CH4_HDP = NaN; end
% %     try TEA_all(sim).LG_416 = h.Tree.FindNode("\Data\Streams\416\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).LG_416 = NaN; end
% %     try TEA_all(sim).CH4_416 = h.Tree.FindNode("\Data\Streams\416\Output\MASSFLOW\MIXED\CH4").value; catch, TEA_all(sim).CH4_416 = NaN; end
% %     try TEA_all(sim).LG_SMR = h.Tree.FindNode("\Data\Streams\407\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).LG_SMR = NaN; end
% %     try TEA_all(sim).CH4_SMR = h.Tree.FindNode("\Data\Streams\407\Output\MASSFLOW\MIXED\CH4").value; catch, TEA_all(sim).CH4_SMR = NaN; end
% % 
% %     % Equipment cost
% %     EC = processData_msp.CapitalCost;
% %     equipment_name = {}; equipment_cost = [];
% %     flash_cost = 0;
% %     if isfield(EC,'flash2'); flash_cost = flash_cost + sum(EC.flash2(:)); end
% %     if isfield(EC,'flash3'); flash_cost = flash_cost + sum(EC.flash3(:)); end
% %     equipment_name{end+1,1} = 'flash'; equipment_cost(end+1,1) = flash_cost;
% % 
% %     dryer_cost = 0; separator_cost = 0;
% %     if isfield(EC,'separator')
% %         val = EC.separator;
% %         if isnumeric(val)
% %             if size(val,2)>=1; dryer_cost = sum(val(:,1)); end
% %             if size(val,2)>=2; separator_cost = sum(val(:,2)); end
% %         end
% %     end
% %     equipment_name{end+1,1} = 'dryer'; equipment_cost(end+1,1) = dryer_cost;
% %     equipment_name{end+1,1} = 'separator'; equipment_cost(end+1,1) = separator_cost;
% % 
% %     all_fields = fieldnames(EC);
% %     for i = 1:length(all_fields)
% %         fname = all_fields{i};
% %         if any(strcmp(fname, {'flash2','flash3','separator','dryer'})); continue; end
% %         equipment_name{end+1,1} = lower(fname);
% %         equipment_cost(end+1,1) = sum(EC.(fname)(:));
% %     end
% %     total_equipment = sum(equipment_cost);
% %     equipment_name{end+1,1} = 'total equipment cost';
% %     equipment_cost(end+1,1) = total_equipment;
% %     TEA_all(sim).Equipment = table(equipment_name, equipment_cost, 'VariableNames', {'Component','Cost'});
% % 
% %     % CAPEX
% %     econ_fields = {'C_alloc','C_cont','C_site','C_startup','C_land','C_comp','C_royal','C_serv'};
% %     econ_name = {'Alloc','Contingency','SitePrep','Startup','Land','C_comp','Royalties','Services'};
% %     econ_cost = zeros(length(econ_fields),1);
% %     for i = 1:length(econ_fields)
% %         if isfield(processData_msp.EconomicData, econ_fields{i})
% %             econ_cost(i) = sum(processData_msp.EconomicData.(econ_fields{i})(:));
% %         end
% %     end
% %     capex_name = [{'Equipment Cost'}; econ_name'; {'Total CAPEX'}];
% %     capex_cost = [total_equipment; econ_cost; total_equipment + sum(econ_cost)];
% %     TEA_all(sim).CAPEX = table(capex_name, capex_cost, 'VariableNames', {'Component','Cost'});
% % 
% %     % Feed & Utility
% %     op_fields = fieldnames(processData_msp.OperatingCost);
% %     op_name = cellstr(string(op_fields));
% %     op_cost = zeros(length(op_fields),1);
% %     for i = 1:length(op_fields)
% %         op_cost(i) = sum(processData_msp.OperatingCost.(op_fields{i})(:));
% %     end
% %     biomass_cost = 0; ng_cost = 0;
% %     if isfield(processData_msp.EconomicData,'C_feed')
% %         feed_data = processData_msp.EconomicData.C_feed;
% %         if size(feed_data,2)>=2
% %             biomass_cost = sum(feed_data(:,1));
% %             ng_cost = sum(feed_data(:,2));
% %         end
% %     end
% %     feed_util_name = [op_name; {'Biomass'}; {'Natural Gas'}];
% %     feed_util_cost = [op_cost; biomass_cost; ng_cost];
% %     feed_util_name = [feed_util_name; {'Total FeedUtility'}];
% %     feed_util_cost = [feed_util_cost; sum(feed_util_cost)];
% %     TEA_all(sim).FeedUtility = table(feed_util_name, feed_util_cost, 'VariableNames', {'Component','Cost'});
% % 
% %     % OPEX
% %     if isfield(processData_msp.EconomicData,'C_catalyst')
% %         processData_msp.EconomicData.C_catalyst_opex = (1/3)*processData_msp.EconomicData.C_catalyst;
% %     else
% %         processData_msp.EconomicData.C_catalyst_opex = 0;
% %     end
% %     % capture OPEX field
% %     if isfield(processData_msp.OperatingCost,'Capture')
% %         processData_msp.EconomicData.C_cap_op = sum(processData_msp.OperatingCost.Capture(:));
% %     else
% %         processData_msp.EconomicData.C_cap_op = 0;
% %     end
% %     opex_fields = {'GE','C_M','C_D','C_O','C_prop','C_OH','C_catalyst_opex','C_feed','C_util','C_cap_op'};
% %     opex_name = {'General Expenses','Maintenance','Depreciation','Operation','Property Taxes & Insurance','Operating Overhead','Catalyst OPEX','Feed Cost','Utility Cost','Capture OPEX'};
% %     opex_cost = zeros(length(opex_fields),1);
% %     for i = 1:length(opex_fields)
% %         if isfield(processData_msp.EconomicData, opex_fields{i})
% %             opex_cost(i) = sum(processData_msp.EconomicData.(opex_fields{i})(:));
% %         end
% %     end
% %     opex_name = [opex_name, {'Total OPEX'}];
% %     opex_cost = [opex_cost; sum(opex_cost)];
% %     TEA_all(sim).OPEX = table(opex_name', opex_cost, 'VariableNames', {'Component','Cost'});
% % 
% %     release(h);
% % end
% % 
% % categories = {'Equipment','CAPEX','FeedUtility','OPEX'};
% % combinedTEA = table();
% % 
% % for c = 1:length(categories)
% %     cat = categories{c};
% % 
% %     allComponents = {};
% %     for sim = 1:nSim
% %         allComponents = [allComponents; TEA_all(sim).(cat).Component];
% %     end
% %     allComponents = unique(allComponents,'stable');
% % 
% %     temp = table(allComponents, 'VariableNames', {'Category'});
% % 
% %     for sim = 1:nSim
% %         [~, colName, ~] = fileparts(list_simulations{sim});
% %         costSim = zeros(height(temp),1);
% %         [found, idx] = ismember(temp.Category, TEA_all(sim).(cat).Component);
% %         costSim(found) = TEA_all(sim).(cat).Cost(idx(found));
% %         temp.(colName) = costSim;
% %     end
% % 
% %     temp.ComponentGroup = repmat({cat}, height(temp), 1);
% % 
% %     combinedTEA = [combinedTEA; temp];
% % end
% % 
% % allCols = combinedTEA.Properties.VariableNames;
% % fileCols = setdiff(allCols, {'ComponentGroup','Category'}, 'stable');
% % combinedTEA = combinedTEA(:, [{'ComponentGroup','Category'}, fileCols]);
% % 
% % mspRow = cell(1, width(combinedTEA)); mspRow{1} = 'MSP [$/kg]';
% % for sim = 1:nSim
% %     [~, colName, ~] = fileparts(list_simulations{sim});
% %     idx = find(strcmp(combinedTEA.Properties.VariableNames, colName));
% %     if ~isempty(idx)
% %         mspRow{idx} = TEA_all(sim).MSP;
% %     end
% % end
% % mspTable = cell2table(mspRow, 'VariableNames', combinedTEA.Properties.VariableNames);
% % 
% % extraRows = {};
% % vars = {
% %     'SAF production [kg/hr]',      'SAF_Production';
% %     'Gasoline production [kg/hr]', 'Gasoline_Production';
% %     'NG feed [kg/hr]',             'NG_Feed';
% %     'H2 consumption [kg/hr]',      'H2_Consumption';
% %     'CO2 capture flow [kg/hr]',    'CO2_Capture';
% %     'LG generation in HDP [kg/hr]','LG_HDP';
% %     'CH4 in HDP [kg/hr]',          'CH4_HDP';
% %     'LG in 416 [kg/hr]',           'LG_416';
% %     'CH4 in 416 [kg/hr]',          'CH4_416';
% %     'LG to SMR [kg/hr]',           'LG_SMR';
% %     'CH4 to SMR [kg/hr]',          'CH4_SMR';
% %     };
% % for r = 1:size(vars,1)
% %     row = cell(1, width(combinedTEA));
% %     row{1} = vars{r,1};
% %     for sim = 1:nSim
% %         [~, colName, ~] = fileparts(list_simulations{sim});
% %         idx = find(strcmp(combinedTEA.Properties.VariableNames, colName));
% %         if ~isempty(idx)
% %             % if field missing, use NaN
% %             if isfield(TEA_all(sim), vars{r,2})
% %                 row{idx} = TEA_all(sim).(vars{r,2});
% %             else
% %                 row{idx} = NaN;
% %             end
% %         end
% %     end
% %     extraRows = [extraRows; row];
% % end
% % extraTable = cell2table(extraRows,'VariableNames',combinedTEA.Properties.VariableNames);
% % 
% % combinedTEA_out = [mspTable; extraTable; combinedTEA];
% % 
% % % LCA
% % Results = table();
% % maxElecLength  = 12; maxSteamLength = 10; maxRefLength = 10;
% % maxWWTLength   = 5;  maxWFLength    = 5;  maxNGLength    = 5;
% % maxAllocLength = 5;  maxFeedLength  = 5;  maxEmiLength   = 5;
% % 
% % scenarioFields = {'elec_GRID','elec_FUTURE','elec_NUCLEAR','elec_WIND','elec_PHOTO','steam','ref','WWT','Feed','emi'};
% % 
% % for idx = 1:nSim
% %     pfad = fullfile(simFolder, list_simulations{idx});
% %     fprintf('Processing simulation (LCA): %s\n', pfad);
% % 
% %     h = actxserver('Apwn.Document');
% %     invoke(h,'InitFromArchive2', pfad);
% %     h.Engine.Reinit;
% %     h.Engine.Run2;
% % 
% %     processData = struct(); processData.LCA = struct();
% %     processData.LCA.elec  = []; processData.LCA.steam = []; processData.LCA.ref   = [];
% %     processData.LCA.WF    = []; processData.LCA.NG    = [];
% % 
% %     processData = accessData_SAF(processData, h);
% %     unitCount = size(processData.Units,1);
% % 
% %     COST_PARAMETER; LCA_PARAMETER;
% %     processData = CAPEX_SAF(processData, unitCount, COST, h);
% %     processData = OPEX_SAF(processData, unitCount, COST, h);
% %     processData = cash_flow_generic_SAF(processData, COST);
% %     processData = LCA_analysis_SAF(processData, LCA);
% %     fun = @(x)msp_SAF(processData, COST, x); x0 = 3;
% %     msp = fsolve(fun, x0);
% %     processData_msp = cash_flow_generic_SAF_opt(processData, COST, msp);
% %     Read_results;
% % 
% %     row = table();
% %     row.Case = list_simulations{idx};
% % 
% %     % LCA Allocation
% %     if isfield(processData_msp,'LCA_alloc_Gasoline')
% %         allocGas = processData_msp.LCA_alloc_Gasoline(:)';
% %     else
% %         allocGas = [];
% %     end
% %     allocGas = [allocGas zeros(1,maxAllocLength-length(allocGas))];
% %     for j = 1:maxAllocLength
% %         row.(sprintf('Alloc_Gasoline_%d',j)) = allocGas(j);
% %     end
% % 
% %     if isfield(processData_msp,'LCA_alloc_SAF')
% %         allocSAF = processData_msp.LCA_alloc_SAF(:)';
% %     else
% %         allocSAF = [];
% %     end
% %     allocSAF = [allocSAF zeros(1,maxAllocLength-length(allocSAF))];
% %     for j = 1:maxAllocLength
% %         row.(sprintf('Alloc_SAF_%d',j)) = allocSAF(j);
% %     end
% % 
% %     % LCA main - electricity types
% %     elecTypes = {'Grid','Future','Nuclear','Wind','PV'};
% %     for iType = 1:length(elecTypes)
% %         typeName = elecTypes{iType};
% %         vec = [];
% %         if ~isempty(processData_msp.LCA.elec)
% %             if iscell(processData_msp.LCA.elec)
% %                 if length(processData_msp.LCA.elec) >= iType
% %                     vec = processData_msp.LCA.elec{iType};
% %                 end
% %             else
% %                 vec = processData_msp.LCA.elec;
% %             end
% %         end
% %         vec = [vec(:)' zeros(1,maxElecLength-length(vec))];
% %         for j = 1:maxElecLength
% %             row.(sprintf('Elec_%s_%d', typeName,j)) = vec(j);
% %         end
% %     end
% % 
% %     % Steam
% %     steamVec = [];
% %     if isfield(processData_msp.LCA,'steam') && ~isempty(processData_msp.LCA.steam)
% %         steamVec = processData_msp.LCA.steam(:)';
% %     end
% %     steamVec = [steamVec zeros(1,maxSteamLength-length(steamVec))];
% %     for j = 1:maxSteamLength
% %         row.(sprintf('Steam_%d',j)) = steamVec(j);
% %     end
% % 
% %     % Refrigerant
% %     refVec = [];
% %     if isfield(processData_msp.LCA,'ref') && ~isempty(processData_msp.LCA.ref)
% %         refVec = processData_msp.LCA.ref(:)';
% %     end
% %     refVec = [refVec zeros(1,maxRefLength-length(refVec))];
% %     for j = 1:maxRefLength
% %         row.(sprintf('Ref_%d',j)) = refVec(j);
% %     end
% % 
% %     % WWT
% %     wwtVec = [];
% %     if isfield(processData_msp,'Output') && isfield(processData_msp.Output,'WASTE_W')
% %         wwtVec = processData_msp.Output.WASTE_W.Massflow(:)';
% %     end
% %     wwtVec = [wwtVec zeros(1,maxWWTLength-length(wwtVec))];
% %     for j = 1:maxWWTLength
% %         row.(sprintf('WWT_%d',j)) = wwtVec(j);
% %     end
% % 
% %     % Biomass (WF)
% %     wfVec = [];
% %     if isfield(processData_msp.LCA,'WF') && ~isempty(processData_msp.LCA.WF)
% %         wfVec = processData_msp.LCA.WF(:)';
% %     end
% %     wfVec = [wfVec zeros(1,maxWFLength-length(wfVec))];
% %     for j = 1:maxWFLength
% %         row.(sprintf('Biomass_%d',j)) = wfVec(j);
% %     end
% % 
% %     % Natural Gas (NG)
% %     ngVec = [];
% %     if isfield(processData_msp.LCA,'NG') && ~isempty(processData_msp.LCA.NG)
% %         ngVec = processData_msp.LCA.NG(:)';
% %     end
% %     ngVec = [ngVec zeros(1,maxNGLength-length(ngVec))];
% %     for j = 1:maxNGLength
% %         row.(sprintf('NG_%d',j)) = ngVec(j);
% %     end
% % 
% %     % LCA Scenario
% %     if isfield(processData_msp.LCA,'scenario')
% %         sc = processData_msp.LCA.scenario;
% %         for f = 1:length(scenarioFields)
% %             fieldName = scenarioFields{f};
% %             if isfield(sc, fieldName)
% %                 data = sc.(fieldName);
% %                 if isnumeric(data)
% %                     switch fieldName
% %                         case {'elec_GRID','elec_FUTURE','elec_NUCLEAR','elec_WIND','elec_PHOTO'}
% %                             maxLen = maxElecLength;
% %                         case 'steam'
% %                             maxLen = maxSteamLength;
% %                         case 'ref'
% %                             maxLen = maxRefLength;
% %                         case 'WWT'
% %                             maxLen = maxWWTLength;
% %                         case 'Feed'
% %                             maxLen = maxFeedLength;
% %                         case 'emi'
% %                             maxLen = maxEmiLength;
% %                         otherwise
% %                             maxLen = length(data);
% %                     end
% %                     vec = [data(:)' zeros(1,maxLen-length(data))];
% %                     for j = 1:maxLen
% %                         row.(sprintf('Scenario_%s_%d', fieldName,j)) = vec(j);
% %                     end
% %                 elseif isstruct(data)
% %                     subFields = fieldnames(data);
% %                     for sf = 1:length(subFields)
% %                         subVec = data.(subFields{sf});
% %                         if isnumeric(subVec)
% %                             subVec = subVec(:)';
% %                             for j = 1:length(subVec)
% %                                 row.(sprintf('Scenario_%s_%s_%d', fieldName, subFields{sf}, j)) = subVec(j);
% %                             end
% %                         end
% %                     end
% %                 end
% %             end
% %         end
% %     end
% % 
% %     Results = [Results; row];
% %     release(h);
% % end
% % 
% % % TO EXCEL
% % try
% %     writetable(combinedTEA_out, outputFile, 'Sheet', 'TEA', 'WriteRowNames', false);
% % catch ME
% %     warning('Failed to write TEA sheet: %s', ME.message);
% %     try
% %         writecell([combinedTEA_out.Properties.VariableNames; table2cell(combinedTEA_out)], outputFile, 'Sheet', 'TEA');
% %     catch
% %         warning('Fallback write TEA also failed: %s', ME.message);
% %     end
% % end
% % 
% % try
% %     writetable(Results, outputFile, 'Sheet', 'LCA', 'WriteRowNames', false);
% % catch ME
% %     warning('Failed to write LCA sheet: %s', ME.message);
% % end
% % 
% % fprintf('Results saved to Excel: %s\n', outputFile);


simFolder = 'D:\523506\A\Create_251120_SA\Simulation\';
% list_simulations = {
% % %     '251122_pyrolysis_oil_CC_case_a_run.bkp';
% % %     '251122_pyrolysis_oil_CC_case_i_run.bkp';
% % %     '251122_pyrolysis_oil_CC_case_f_run.bkp';



list_simulations = {
    '251120_pyrolysis_oil_CC_case_a_2_run.bkp';
    '251120_pyrolysis_oil_CC_case_f_2_run.bkp';
    '251120_pyrolysis_oil_CC_case_i_2_run.bkp';
};

% simFolder = 'D:\523506\A\Create_251120_SA\Simulation\sensitivity_20251121\grid_20251121_174425_param_1\converged\';
% 
% list_simulations = {
%     'caseSA_0.bkp';
%     'caseSA_1.bkp';
%     'caseSA_2.bkp';
%     'caseSA_3.bkp';
%     'caseSA_4.bkp';
%     'caseSA_5.bkp';
%     'caseSA_6.bkp';
%     'caseSA_7.bkp';
%     'caseSA_8.bkp';
%     'caseSA_9.bkp';
%     'caseSA_10.bkp';
% };
%
% simFolder = 'D:\523506\A\Create_251120_SA\Simulation\sensitivity_20251121\grid_20251121_173949_param_0\converged\';
% 
% list_simulations = {
%     'caseSA_0.bkp';
%     'caseSA_1.bkp';
%     'caseSA_2.bkp';
%     'caseSA_3.bkp';
%     'caseSA_5.bkp';
%     'caseSA_6.bkp';
%     'caseSA_7.bkp';
%     'caseSA_8.bkp';
%     'caseSA_9.bkp';
%     'caseSA_10.bkp';
% };

nSim = length(list_simulations);
outputFile = fullfile(simFolder,'SAF_Sensitivity_Analysis_251122_4.xlsx');
% outputFile = fullfile(simFolder,'SAF_Sensitivity_Analysis_251122_noise_true.xlsx');
% outputFile = fullfile(simFolder,'SAF_Sensitivity_Analysis_251122_param_1.xlsx');
% outputFile = fullfile(simFolder,'SAF_Sensitivity_Analysis_251122_param_0.xlsx');

%% TEA
TEA_all = struct();

for sim = 1:nSim

    pfad = fullfile(simFolder, list_simulations{sim});
    fprintf('\nProcessing simulation (TEA): %s\n', pfad);

    h = actxserver('Apwn.Document');
    invoke(h,'InitFromArchive2', pfad);
    h.Engine.Reinit;
    h.Engine.Run2;

    processData = accessData_SAF([], h);
    unitCount   = size(processData.Units,1);

    if ~isfield(processData,'LCA'), processData.LCA = struct(); end
    f = {'elec','steam','ref','WF','NG'};
    for ii = 1:length(f)
        if ~isfield(processData.LCA,f{ii})
            processData.LCA.(f{ii}) = [];
        end
    end

    COST_PARAMETER
    LCA_PARAMETER

    processData = CAPEX_SAF(processData,unitCount,COST,h);
    processData = OPEX_SAF(processData,unitCount,COST,h);
    processData = cash_flow_generic_SAF(processData,COST);
    processData = LCA_analysis_SAF(processData,LCA);

    fun = @(x)msp_SAF(processData,COST,x);
    msp = fsolve(fun,3);
    processData_msp = cash_flow_generic_SAF_opt(processData,COST,msp);
    TEA_all(sim).MSP = msp;

    try TEA_all(sim).SAF_Production = h.Tree.FindNode("\Data\Streams\326\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).SAF_Production = NaN; end
    try TEA_all(sim).Gasoline_Production = h.Tree.FindNode("\Data\Streams\324\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).Gasoline_Production = NaN; end
    try TEA_all(sim).NG_Feed = h.Tree.FindNode("\Data\Streams\409\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).NG_Feed = NaN; end

    % H2 consumption
    try
        H2_in  = h.Tree.FindNode("\Data\Streams\206\Output\MASSFLOW\MIXED\H2").value;
        H2_out = h.Tree.FindNode("\Data\Streams\215\Output\MASSFLOW\MIXED\H2").value;
        TEA_all(sim).H2_Consumption = H2_in - H2_out;
    catch
        TEA_all(sim).H2_Consumption = NaN;
    end

    % CO2 capture
    try TEA_all(sim).CO2_Capture = h.Tree.FindNode("\Data\Streams\426\Output\MASSFLMX\$TOTAL").value;
    catch, TEA_all(sim).CO2_Capture = NaN;
    end

    % LG generation in HDP (stream 215)
    try
        compList = {'CO2','CH4','C2H4','C2H6','C3H6','C3H8','C4H10','NH3'};
        sumLG = 0;
        for ci = 1:length(compList)
            try
                nodePath = sprintf("\\Data\\Streams\\215\\Output\\MASSFLOW\\MIXED\\%s", compList{ci});
                v = h.Tree.FindNode(nodePath).value;
                if isnumeric(v), sumLG = sumLG + v; end
            catch
            end
        end
        TEA_all(sim).LG_HDP = sumLG;
    catch
        TEA_all(sim).LG_HDP = NaN;
    end

    % Others
    try TEA_all(sim).CH4_HDP = h.Tree.FindNode("\Data\Streams\215\Output\MASSFLOW\MIXED\CH4").value; catch, TEA_all(sim).CH4_HDP = NaN; end
    try TEA_all(sim).LG_416 = h.Tree.FindNode("\Data\Streams\416\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).LG_416 = NaN; end
    try TEA_all(sim).CH4_416 = h.Tree.FindNode("\Data\Streams\416\Output\MASSFLOW\MIXED\CH4").value; catch, TEA_all(sim).CH4_416 = NaN; end
    try TEA_all(sim).LG_SMR = h.Tree.FindNode("\Data\Streams\407\Output\MASSFLMX\$TOTAL").value; catch, TEA_all(sim).LG_SMR = NaN; end
    try TEA_all(sim).CH4_SMR = h.Tree.FindNode("\Data\Streams\407\Output\MASSFLOW\MIXED\CH4").value; catch, TEA_all(sim).CH4_SMR = NaN; end

    % EQUIPMENT COST
    EC = processData_msp.CapitalCost;
    eq_name = {};
    eq_cost = [];

    % flash
    flash_cost = 0;
    if isfield(EC,'flash2'), flash_cost = flash_cost + sum(EC.flash2(:)); end
    if isfield(EC,'flash3'), flash_cost = flash_cost + sum(EC.flash3(:)); end
    eq_name{end+1,1} = 'flash';
    eq_cost(end+1,1) = flash_cost;

    % dryer & separator
    dryer_cost = 0; separator_cost = 0;
    if isfield(EC,'separator')
        val = EC.separator;
        if size(val,2)>=1, dryer_cost    = sum(val(:,1)); end
        if size(val,2)>=2, separator_cost = sum(val(:,2)); end
    end
    eq_name(end+1,1) = {'dryer'};     eq_cost(end+1,1) = dryer_cost;
    eq_name(end+1,1) = {'separator'}; eq_cost(end+1,1) = separator_cost;

    % remaining
    allFields = fieldnames(EC);
    for i = 1:length(allFields)
        fn = allFields{i};
        if any(strcmp(fn,{'flash2','flash3','separator','dryer'})), continue; end
        eq_name{end+1,1} = lower(fn);
        eq_cost(end+1,1) = sum(EC.(fn)(:));
    end

    total_equipment = sum(eq_cost);
    eq_name{end+1,1} = 'total equipment cost';
    eq_cost(end+1,1) = total_equipment;

    TEA_all(sim).Equipment = table(eq_name,eq_cost,'VariableNames',{'Component','Cost'});

    % CAPEX
    fields = {'C_alloc','C_cont','C_site','C_startup','C_land','C_comp','C_royal','C_serv'};
    names = {'Alloc','Cont','Site','Startup','Land','Comp','Royal','Serv'};
    costs = zeros(length(fields),1);

    for i = 1:length(fields)
        if isfield(processData_msp.EconomicData,fields{i})
            costs(i) = sum(processData_msp.EconomicData.(fields{i})(:));
        end
    end

    cap_name = [{'Equipment Cost'}; names'; {'Total CAPEX'}];
    cap_cost = [total_equipment; costs; total_equipment+sum(costs)];

    TEA_all(sim).CAPEX = table(cap_name,cap_cost,'VariableNames',{'Component','Cost'});

    % Feed and Utility
    op_fields = fieldnames(processData_msp.OperatingCost);
    op_cost = zeros(length(op_fields),1);
    for i = 1:length(op_fields)
        op_cost(i) = sum(processData_msp.OperatingCost.(op_fields{i})(:));
    end

    biomass_cost = 0; ng_cost = 0;
    if isfield(processData_msp.EconomicData,'C_feed')
        fd = processData_msp.EconomicData.C_feed;
        if size(fd,2)>=2
            biomass_cost = sum(fd(:,1));
            ng_cost      = sum(fd(:,2));
        end
    end

    fu_name = [op_fields; {'Biomass'}; {'Natural Gas'}];
    fu_cost = [op_cost; biomass_cost; ng_cost];
    fu_name = [fu_name; {'Total FeedUtility'}];
    fu_cost = [fu_cost; sum(fu_cost)];

    TEA_all(sim).FeedUtility = table(fu_name,fu_cost,'VariableNames',{'Component','Cost'});

    % OPEX
    if isfield(processData_msp.EconomicData,'C_catalyst')
        processData_msp.EconomicData.C_catalyst_opex = (1/3)*processData_msp.EconomicData.C_catalyst;
    else
        processData_msp.EconomicData.C_catalyst_opex = 0;
    end

    if isfield(processData_msp.OperatingCost,'Capture')
        processData_msp.EconomicData.C_cap_op = sum(processData_msp.OperatingCost.Capture(:));
    else
        processData_msp.EconomicData.C_cap_op = 0;
    end

    opf = {'GE','C_M','C_D','C_O','C_prop','C_OH','C_catalyst_opex','C_feed','C_util','C_cap_op'};
    opn = {'General','Maint','Dep','Oper','Tax','OH','Catalyst','Feed','Util','Capture'};
    opc = zeros(length(opf),1);

    for i = 1:length(opf)
        if isfield(processData_msp.EconomicData,opf{i})
            opc(i) = sum(processData_msp.EconomicData.(opf{i})(:));
        end
    end

    opn = [opn,{'Total OPEX'}];
    opc = [opc; sum(opc)];

    TEA_all(sim).OPEX = table(opn',opc,'VariableNames',{'Component','Cost'});

    release(h);

end

categories = {'Equipment','CAPEX','FeedUtility','OPEX'};
combinedTEA = table();

for c = 1:length(categories)
    cat = categories{c};

    allComp = {};
    for sim = 1:nSim
        allComp = [allComp; TEA_all(sim).(cat).Component];
    end

    allComp = unique(allComp,'stable');
    T = table(allComp,'VariableNames',{'Category'});

    for sim = 1:nSim
        [~,col] = fileparts(list_simulations{sim});
        costSim = zeros(height(T),1);
        [found,idx] = ismember(T.Category,TEA_all(sim).(cat).Component);
        costSim(found) = TEA_all(sim).(cat).Cost(idx(found));
        T.(col) = costSim;
    end

    T.ComponentGroup = repmat({cat},height(T),1);
    combinedTEA = [combinedTEA; T];
end

allCols = combinedTEA.Properties.VariableNames;
fileCols = setdiff(allCols,{'ComponentGroup','Category'},'stable');
combinedTEA = combinedTEA(:,[{'ComponentGroup','Category'}, fileCols]);

mspRow = cell(1,width(combinedTEA)); 
mspRow{1} = 'MSP [$/kg]';

for sim = 1:nSim
    [~,col] = fileparts(list_simulations{sim});
    idx = find(strcmp(combinedTEA.Properties.VariableNames,col));
    if ~isempty(idx)
        mspRow{idx} = TEA_all(sim).MSP;
    end
end

mspTable = cell2table(mspRow,'VariableNames',combinedTEA.Properties.VariableNames);

vars = {
    'SAF production [kg/hr]','SAF_Production';
    'Gasoline production [kg/hr]','Gasoline_Production';
    'NG feed [kg/hr]','NG_Feed';
    'H2 consumption [kg/hr]','H2_Consumption';
    'CO2 capture flow [kg/hr]','CO2_Capture';
    'LG generation in HDP [kg/hr]','LG_HDP';
    'CH4 in HDP [kg/hr]','CH4_HDP';
    'LG in 416 [kg/hr]','LG_416';
    'CH4 in 416 [kg/hr]','CH4_416';
    'LG to SMR [kg/hr]','LG_SMR';
    'CH4 to SMR [kg/hr]','CH4_SMR';
};

extraRows = {};

for r = 1:size(vars,1)
    row = cell(1,width(combinedTEA));
    row{1} = vars{r,1};
    for sim = 1:nSim
        [~,col] = fileparts(list_simulations{sim});
        idx = find(strcmp(combinedTEA.Properties.VariableNames,col));
        if ~isempty(idx)
            if isfield(TEA_all(sim),vars{r,2})
                row{idx} = TEA_all(sim).(vars{r,2});
            else
                row{idx} = NaN;
            end
        end
    end
    extraRows = [extraRows; row];
end

extraTable = cell2table(extraRows,'VariableNames',combinedTEA.Properties.VariableNames);
combinedTEA_out = [mspTable; extraTable; combinedTEA];

%% LCA
Results = table();   % will accumulate 1 row per simulation

maxElecLength  = 17; 
maxSteamLength = 17; 
maxRefLength   = 17;
maxWWTLength   = 5;  
maxWFLength    = 5;  
maxNGLength    = 5;
maxAllocLength = 5;  
maxFeedLength  = 5;  
maxEmiLength   = 5;

scenarioFields = {'elec_GRID','elec_FUTURE','elec_NUCLEAR','elec_WIND','elec_PHOTO',...
                  'steam','ref','WWT','Feed','emi'};

for idxSim = 1:nSim

    pfad = fullfile(simFolder, list_simulations{idxSim});
    fprintf('\nProcessing simulation (LCA): %s\n', pfad);

    h = actxserver('Apwn.Document');
    invoke(h,'InitFromArchive2', pfad);
    h.Engine.Reinit;
    h.Engine.Run2;

    processData = struct();
    processData.LCA = struct();
    f = {'elec','steam','ref','WF','NG'};
    for ii = 1:length(f)
        processData.LCA.(f{ii}) = [];
    end

    processData = accessData_SAF(processData, h);
    unitCount   = size(processData.Units,1);

    COST_PARAMETER;
    LCA_PARAMETER;

    processData = CAPEX_SAF(processData, unitCount, COST, h);
    processData = OPEX_SAF(processData, unitCount, COST, h);
    processData = cash_flow_generic_SAF(processData, COST);
    processData = LCA_analysis_SAF(processData, LCA);

    fun_msp = @(x)msp_SAF(processData, COST, x);
    msp_LCA = fsolve(fun_msp, 3);
    processData_msp = cash_flow_generic_SAF_opt(processData, COST, msp_LCA);

    Read_results;

    row = table();

    row.Case = string(list_simulations{idxSim});

    % Allocation Gasoline
    if isfield(processData_msp,'LCA_alloc_Gasoline')
        allocGas = processData_msp.LCA_alloc_Gasoline(:)';
    else
        allocGas = [];
    end
    allocGas = [allocGas zeros(1,maxAllocLength - length(allocGas))];

    for j = 1:maxAllocLength
        row.(sprintf('Alloc_Gasoline_%d', j)) = allocGas(j);
    end

    % Allocation SAF
    if isfield(processData_msp,'LCA_alloc_SAF')
        allocSAF = processData_msp.LCA_alloc_SAF(:)';
    else
        allocSAF = [];
    end
    allocSAF = [allocSAF zeros(1,maxAllocLength - length(allocSAF))];

    for j = 1:maxAllocLength
        row.(sprintf('Alloc_SAF_%d', j)) = allocSAF(j);
    end

    % Electricity LCA
    elecTypes = {'Grid','Future','Nuclear','Wind','PV'};
    for iType = 1:length(elecTypes)

        typeName = elecTypes{iType};
        vec = [];

        if ~isempty(processData_msp.LCA.elec)
            if iscell(processData_msp.LCA.elec)
                if length(processData_msp.LCA.elec) >= iType
                    vec = processData_msp.LCA.elec{iType};
                end
            else
                if size(processData_msp.LCA.elec,2) >= iType
                    vec = processData_msp.LCA.elec(:,iType)';
                else
                    vec = processData_msp.LCA.elec(:)';
                end
            end
        end

        % pastikan tidak kosong → isi 0
        if isempty(vec)
            vec = 0;
        end

        vec = [vec zeros(1, maxElecLength - length(vec))];

        for j = 1:maxElecLength
            row.(sprintf('Elec_%s_%d', typeName, j)) = vec(j);
        end
    end

    % Steam LCA
    steamVec = [];
    if isfield(processData_msp.LCA,'steam') && ~isempty(processData_msp.LCA.steam)
        steamVec = processData_msp.LCA.steam(:)';
    end
    if isempty(steamVec)
        steamVec = 0;
    end
    steamVec = [steamVec zeros(1, maxSteamLength - length(steamVec))];

    for j = 1:maxSteamLength
        row.(sprintf('Steam_%d', j)) = steamVec(j);
    end

    % Refrigerant LCA
    refVec = [];
    if isfield(processData_msp.LCA,'ref') && ~isempty(processData_msp.LCA.ref)
        refVec = processData_msp.LCA.ref(:)';
    end
    if isempty(refVec)
        refVec = 0;
    end
    refVec = [refVec zeros(1, maxRefLength - length(refVec))];

    for j = 1:maxRefLength
        row.(sprintf('Ref_%d', j)) = refVec(j);
    end

    % WWT
    wwtVec = [];
    if isfield(processData_msp,'Output') && ...
       isfield(processData_msp.Output,'WASTE_W') && ...
       isfield(processData_msp.Output.WASTE_W,'Massflow')

        wwtVec = processData_msp.Output.WASTE_W.Massflow(:)';
    end
    if isempty(wwtVec)
        wwtVec = 0;
    end
    wwtVec = [wwtVec zeros(1, maxWWTLength - length(wwtVec))];

    for j = 1:maxWWTLength
        row.(sprintf('WWT_%d', j)) = wwtVec(j);
    end

    % Biomass (WF)
    wfVec = [];
    if isfield(processData_msp.LCA,'WF') && ~isempty(processData_msp.LCA.WF)
        wfVec = processData_msp.LCA.WF(:)';
    end
    if isempty(wfVec)
        wfVec = 0;
    end
    wfVec = [wfVec zeros(1, maxWFLength - length(wfVec))];

    for j = 1:maxWFLength
        row.(sprintf('Biomass_%d', j)) = wfVec(j);
    end

    % Natural Gas (NG)
    ngVec = [];
    if isfield(processData_msp.LCA,'NG') && ~isempty(processData_msp.LCA.NG)
        ngVec = processData_msp.LCA.NG(:)';
    end
    if isempty(ngVec)
        ngVec = 0;
    end
    ngVec = [ngVec zeros(1, maxNGLength - length(ngVec))];

    for j = 1:maxNGLength
        row.(sprintf('NG_%d', j)) = ngVec(j);
    end

    % LCA Scenario
    if isfield(processData_msp.LCA,'scenario')
        sc = processData_msp.LCA.scenario;

        for f = 1:length(scenarioFields)
            fieldName = scenarioFields{f};

            if isfield(sc, fieldName)
                data = sc.(fieldName);

                if isnumeric(data)
                    switch fieldName
                        case {'elec_GRID','elec_FUTURE','elec_NUCLEAR','elec_WIND','elec_PHOTO'}
                            maxLen = maxElecLength;
                        case 'steam'
                            maxLen = maxSteamLength;
                        case 'ref'
                            maxLen = maxRefLength;
                        case 'WWT'
                            maxLen = maxWWTLength;
                        case 'Feed'
                            maxLen = maxFeedLength;
                        case 'emi'
                            maxLen = maxEmiLength;
                        otherwise
                            maxLen = length(data);
                    end

                    % kalau data kosong → vector nol
                    if isempty(data)
                        vec = zeros(1,maxLen);
                    else
                        vec = [data(:)' zeros(1, maxLen - length(data))];
                    end

                    for j = 1:maxLen
                        row.(sprintf('Scenario_%s_%d', fieldName, j)) = vec(j);
                    end

                elseif isstruct(data)
                    subFields = fieldnames(data);
                    for sf = 1:length(subFields)
                        subVec = data.(subFields{sf})(:)';
                        if isempty(subVec)
                            subVec = 0;
                        end
                        for j = 1:length(subVec)
                            row.(sprintf('Scenario_%s_%s_%d', fieldName, subFields{sf}, j)) = subVec(j);
                        end
                    end
                end
            end
        end
    end

    release(h);

    % === MERGE KE Results: TANPA NaN, KOSONG → 0 ===
    if isempty(Results)
        Results = row;
    else
        % kolom baru di row → tambahkan ke Results, isi 0
        newColsInRow = setdiff(row.Properties.VariableNames, Results.Properties.VariableNames);
        for nc = 1:length(newColsInRow)
            Results.(newColsInRow{nc}) = zeros(height(Results),1);
        end

        % kolom yang ada di Results tapi belum ada di row → isi 0
        newColsInResults = setdiff(Results.Properties.VariableNames, row.Properties.VariableNames);
        for nc = 1:length(newColsInResults)
            varName = newColsInResults{nc};
            if strcmp(varName,'Case')
                % Case sudah ada di row, harusnya nggak masuk sini, tapi jaga-jaga
                if ~isfield(row,varName)
                    row.(varName) = "";
                end
            else
                row.(varName) = 0;
            end
        end

        % samakan urutan kolom
        row = row(:, Results.Properties.VariableNames);

        Results = [Results; row];
    end

end 

% TO EXCEL

fprintf('\n=== Writing TEA + LCA results to Excel ===\n');

try
    writetable(combinedTEA_out, outputFile, 'Sheet', 'TEA', 'WriteRowNames', false);
catch ME
    warning('Failed to write TEA sheet normally: %s', ME.message);
    try
        writecell([combinedTEA_out.Properties.VariableNames; table2cell(combinedTEA_out)], ...
                  outputFile, 'Sheet', 'TEA');
    catch ME2
        warning('Fallback TEA write also failed: %s', ME2.message);
    end
end

try
    writetable(Results, outputFile, 'Sheet', 'LCA', 'WriteRowNames', false);
catch ME
    warning('Failed to write LCA sheet: %s', ME.message);
end

fprintf('Results saved to Excel: %s\n', outputFile);
