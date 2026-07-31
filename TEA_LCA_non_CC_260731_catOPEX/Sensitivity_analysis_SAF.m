simFolder = "C:\Users\user\Desktop\SAF_HDO\res_20260403_non_CC\";

list_simulations = {
    '260401_pyrolysis_oil_CC_case_j_HOadjustment3_non_CC.bkp';
    '260401_pyrolysis_oil_CC_case_k_6_non_CC.bkp';
    '260120_pyrolysis_oil_CC_case_i_non_CC.bkp';
    };

nSim = length(list_simulations);
outputFile = fullfile(simFolder,'SAF_Sensitivity_Analysis_260403_non_CC_all.xlsx');

% TEA
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

    if isfield(processData,'NPV')
        TEA_all(sim).NPV = processData.NPV;
    else
        TEA_all(sim).NPV = NaN;
    end

    fun = @(x)msp_SAF(processData,COST,x);
    msp = fsolve(fun,3);
    processData_msp = cash_flow_generic_SAF_opt_mjsp(processData,COST,msp);

    TEA_all(sim).MSP = msp;
    TEA_all(sim).mjsp_Sales = processData_msp.EconomicData.mjsp_Sales;
    TEA_all(sim).mjsp_CAPEX = processData_msp.EconomicData.mjsp_CAPEX;
    TEA_all(sim).mjsp_OPEX = processData_msp.EconomicData.mjsp_OPEX;

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
    % try TEA_all(sim).CO2_Capture = h.Tree.FindNode("\Data\Streams\426\Output\MASSFLMX\$TOTAL").value;
    % catch, TEA_all(sim).CO2_Capture = NaN;
    % end

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
    try TEA_all(sim).CH4_403 = h.Tree.FindNode("\Data\Streams\403\Output\MASSFLOW\MIXED\CH4").value; catch, TEA_all(sim).CH4_SMR = NaN; end

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
    fields = {'C_alloc','C_WC','C_cont','C_site','C_startup','C_land','C_comp','C_royal','C_serv'};
    names  = {'Allocation','Working Capital','Contingency','Site Prep.','Startup','Land','C_Comp','Royalties','Services'};
    costs  = zeros(length(fields),1);

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

    % if isfield(processData_msp.OperatingCost,'Capture')
    %     processData_msp.EconomicData.C_cap_op = sum(processData_msp.OperatingCost.Capture(:));
    % else
    %     processData_msp.EconomicData.C_cap_op = 0;
    % end

    % opf = {'GE','C_M','C_D','C_O','C_prop','C_OH','C_catalyst_opex','C_feed','C_util','C_cap_op'};
    % opn = {'General Expenses','Maintenance','Depreciation','Operation','Property Taxes & Insurance','Operating Overhead','Catalyst OPEX','Feed Cost','Utility Cost','Capture OPEX'};
    opf = {'GE','C_M','C_D','C_O','C_prop','C_OH','C_catalyst_opex','C_feed','C_util'};
    opn = {'General Expenses','Maintenance','Depreciation','Operation','Property Taxes & Insurance','Operating Overhead','Catalyst OPEX','Feed Cost','Utility Cost'};
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

npvRow = cell(1,width(combinedTEA));
npvRow{1} = 'NPV [$]';

for sim = 1:nSim
    [~,col] = fileparts(list_simulations{sim});
    idx = find(strcmp(combinedTEA.Properties.VariableNames,col));
    if ~isempty(idx)
        mspRow{idx} = TEA_all(sim).MSP;
        if isfield(TEA_all(sim),'NPV')
            npvRow{idx} = TEA_all(sim).NPV;
        else
            npvRow{idx} = NaN;
        end
    end
end

mspTable = cell2table(mspRow,'VariableNames',combinedTEA.Properties.VariableNames);
npvTable = cell2table(npvRow,'VariableNames',combinedTEA.Properties.VariableNames);

vars = {
    'SAF production [kg/hr]','SAF_Production';
    'Gasoline production [kg/hr]','Gasoline_Production';
    'NG feed [kg/hr]','NG_Feed';
    'H2 consumption [kg/hr]','H2_Consumption';
    % 'CO2 capture flow [kg/hr]','CO2_Capture';
    'LG generation in HDP [kg/hr]','LG_HDP';
    'CH4 in HDP [kg/hr]','CH4_HDP';
    'LG in 416 [kg/hr]','LG_416';
    'CH4 in 416 [kg/hr]','CH4_416';
    'LG to SMR [kg/hr]','LG_SMR';
    'CH4 to SMR [kg/hr]','CH4_SMR';
    'CH4 off gas FP [kg/hr]','CH4_403';
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

%%
breakdownVars = {
    'Sales in MSP breakdown', 'mjsp_Sales';
    'CAPEX in MSP breakdown', 'mjsp_CAPEX';
    'OPEX in MSP breakdown', 'mjsp_OPEX';
    };
breakdownRows = {};
for r = 1:size(breakdownVars,1)
    row = cell(1,width(combinedTEA));
    row{1} = breakdownVars{r,1};
    for sim = 1:nSim
        [~,col] = fileparts(list_simulations{sim});
        idx = find(strcmp(combinedTEA.Properties.VariableNames,col));
        if ~isempty(idx)
            if isfield(TEA_all(sim),breakdownVars{r,2})
                row{idx} = sum( TEA_all(sim).(breakdownVars{r,2}) );
            else
                row{idx} = NaN;
            end
        end
    end
    breakdownRows = [breakdownRows; row];
end

breakdownTable = cell2table(breakdownRows,'VariableNames',combinedTEA.Properties.VariableNames);

combinedTEA_out = [mspTable; npvTable; extraTable; combinedTEA; breakdownTable];

%% LCA
Results = table();

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
%%
    % Allocation Gasoline
    if isfield(processData_msp,'LCA_alloc_Gasoline')
        allocGas = processData_msp.LCA_alloc_Gasoline(:)';
    else
        allocGas = NaN;
    end
    row = addVectorDynamic(row,'Alloc_Gasoline',allocGas);

    % Allocation SAF
    if isfield(processData_msp,'LCA_alloc_SAF')
        allocSAF = processData_msp.LCA_alloc_SAF(:)';
    else
        allocSAF = NaN;
    end
    row = addVectorDynamic(row,'Alloc_SAF',allocSAF);

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

        if isempty(vec), vec = NaN; end
        prefix = sprintf('Elec_%s', typeName);
        row = addVectorDynamic(row, prefix, vec);
    end

    % Steam LCA
    steamVec = [];
    if isfield(processData_msp.LCA,'steam') && ~isempty(processData_msp.LCA.steam)
        steamVec = processData_msp.LCA.steam(:)';
    end
    if isempty(steamVec), steamVec = NaN; end
    row = addVectorDynamic(row,'Steam',steamVec);

    % Refrigerant LCA
    refVec = [];
    if isfield(processData_msp.LCA,'ref') && ~isempty(processData_msp.LCA.ref)
        refVec = processData_msp.LCA.ref(:)';
    end
    if isempty(refVec), refVec = NaN; end
    row = addVectorDynamic(row,'Ref',refVec);

    % WWT
    wwtVec = [];
    if isfield(processData_msp,'Output') && ...
            isfield(processData_msp.Output,'WASTE_W') && ...
            isfield(processData_msp.Output.WASTE_W,'Massflow')

        wwtVec = processData_msp.Output.WASTE_W.Massflow(:)';
    end
    if isempty(wwtVec), wwtVec = NaN; end
    row = addVectorDynamic(row,'WWT',wwtVec);

    % Biomass (WF - Wood Feedstock)
    wfVec = [];
    if isfield(processData_msp.LCA,'WF') && ~isempty(processData_msp.LCA.WF)
        wfVec = processData_msp.LCA.WF(:)';
    end
    if isempty(wfVec), wfVec = NaN; end
    row = addVectorDynamic(row,'Biomass',wfVec);

    % Natural Gas (NG)
    ngVec = [];
    if isfield(processData_msp.LCA,'NG') && ~isempty(processData_msp.LCA.NG)
        ngVec = processData_msp.LCA.NG(:)';
    end
    if isempty(ngVec), ngVec = NaN; end
    row = addVectorDynamic(row,'NG',ngVec);

    %% LCA Scenario
    if isfield(processData_msp.LCA,'scenario')
        sc = processData_msp.LCA.scenario;
        scFields = fieldnames(sc);

        for f = 1:length(scFields)
            fieldName = scFields{f};
            data = sc.(fieldName);

            if isnumeric(data)
                vec = data(:)';
                if isempty(vec), vec = NaN; end
                prefix = sprintf('Scenario_%s', fieldName);
                row = addVectorDynamic(row, prefix, vec);

            elseif isstruct(data)
                subFields = fieldnames(data);
                for sf = 1:length(subFields)
                    subVec = data.(subFields{sf})(:)';
                    if isempty(subVec), subVec = NaN; end
                    prefix = sprintf('Scenario_%s_%s', fieldName, subFields{sf});
                    row = addVectorDynamic(row, prefix, subVec);
                end
            end
        end
    end

    release(h);


    if isempty(Results)
        Results = row;
    else
        newColsInRow = setdiff(row.Properties.VariableNames, Results.Properties.VariableNames);
        for nc = 1:length(newColsInRow)
            Results.(newColsInRow{nc}) = NaN(height(Results),1);
        end

        newColsInResults = setdiff(Results.Properties.VariableNames, row.Properties.VariableNames);
        for nc = 1:length(newColsInResults)
            row.(newColsInResults{nc}) = NaN;
        end

        row = row(:, Results.Properties.VariableNames);

        Results = [Results; row];
    end

end

%% TO EXCEL
fprintf('\n=== TEA + LCA results to Excel ===\n');

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

function row = addVectorDynamic(row, prefix, vec)
if nargin < 3 || isempty(vec)
    vec = NaN;
end

vec = vec(:)';

for j = 1:length(vec)
    colName = sprintf('%s_%d', prefix, j);
    row.(colName) = vec(j);
end
end