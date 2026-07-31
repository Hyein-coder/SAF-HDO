%% =========================================
%% Main Script: Full LCA + Scenario Detailed Export (with struct support)
%% =========================================

clear; clc;

% ----------------------------
% Folder dan list file backup
% ----------------------------
folder = 'D:\523506\A\Create_251120_SA\Simulation\';

listFiles = {
    '251120_pyrolysis_oil_CC_case_a_2_run.bkp';
    '251120_pyrolysis_oil_CC_case_f_2_run.bkp';
    '251120_pyrolysis_oil_CC_case_i_2_run.bkp';
};

Results = table();

%% ----------------------------
% Tentukan panjang maksimal tiap tipe LCA & scenario (estimasi)
%% ----------------------------
maxElecLength  = 12;
maxSteamLength = 10;
maxRefLength   = 10;
maxWWTLength   = 5;
maxWFLength    = 5;
maxNGLength    = 5;
maxAllocLength = 5;
maxFeedLength  = 5;
maxEmiLength   = 5;

scenarioFields = {'elec_GRID','elec_FUTURE','elec_NUCLEAR','elec_WIND','elec_PHOTO', ...
                  'steam','ref','WWT','Feed','emi'};

% ----------------------------
% Loop setiap file
% ----------------------------
for idx = 1:length(listFiles)

    fprintf('\n=========================================\n');
    fprintf('Running simulation: %s\n', listFiles{idx});
    fprintf('=========================================\n');

    pfad = fullfile(folder, listFiles{idx});
    h = actxserver('Apwn.Document');
    invoke(h, 'InitFromArchive2', pfad);

    % Jalankan simulasi
    h.Engine.Reinit;
    h.Engine.Run2;

    % ----------------------------
    % Reset struct
    % ----------------------------
    processData = struct();
    processData.LCA = struct();
    processData.LCA.elec  = [];
    processData.LCA.steam = [];
    processData.LCA.ref   = [];
    processData.LCA.WF    = [];
    processData.LCA.NG    = [];

    % ----------------------------
    % Ambil data dari Aspen
    % ----------------------------
    processData = accessData_SAF(processData, h);
    unitCount = size(processData.Units,1);

    % ----------------------------
    % Parameter
    % ----------------------------
    COST_PARAMETER;
    LCA_PARAMETER;

    % ----------------------------
    % Economic calculations
    % ----------------------------
    processData = CAPEX_SAF(processData, unitCount, COST, h);
    processData = OPEX_SAF(processData, unitCount, COST, h);
    processData = cash_flow_generic_SAF(processData, COST);
    processData = LCA_analysis_SAF(processData, LCA);

    % ----------------------------
    % MSP Calculation
    % ----------------------------
    fun = @(x)msp_SAF(processData, COST, x);
    x0 = 3;  
    msp = fsolve(fun, x0);
    processData_msp = cash_flow_generic_SAF_opt(processData, COST, msp);

    % ----------------------------
    % Read additional results
    % ----------------------------
    Read_results;

    % ----------------------------
    % Store results row
    % ----------------------------
    row = table();
    row.Case = listFiles(idx);

    %% ----------------------------
    % LCA allocation
    %% ----------------------------
    % Gasoline
    if isfield(processData_msp,'LCA_alloc_Gasoline')
        allocGas = processData_msp.LCA_alloc_Gasoline(:)';
    else
        allocGas = [];
    end
    allocGas = [allocGas zeros(1,maxAllocLength-length(allocGas))];
    for j = 1:maxAllocLength
        row.(sprintf('Alloc_Gasoline_%d',j)) = allocGas(j);
    end

    % SAF
    if isfield(processData_msp,'LCA_alloc_SAF')
        allocSAF = processData_msp.LCA_alloc_SAF(:)';
    else
        allocSAF = [];
    end
    allocSAF = [allocSAF zeros(1,maxAllocLength-length(allocSAF))];
    for j = 1:maxAllocLength
        row.(sprintf('Alloc_SAF_%d',j)) = allocSAF(j);
    end

    %% ----------------------------
    % LCA main data (elec, steam, ref, WWT, WF, NG)
    %% ----------------------------
    elecTypes = {'Grid','Future','Nuclear','Wind','PV'};
    for iType = 1:length(elecTypes)
        typeName = elecTypes{iType};
        if ~isempty(processData_msp.LCA.elec)
            if iscell(processData_msp.LCA.elec)
                vec = processData_msp.LCA.elec{iType};
            else
                vec = processData_msp.LCA.elec;
            end
        else
            vec = [];
        end
        vec = [vec(:)' zeros(1,maxElecLength-length(vec))];
        for j = 1:maxElecLength
            row.(sprintf('Elec_%s_%d', typeName,j)) = vec(j);
        end
    end

    % Steam
    steamVec = [];
    if isfield(processData_msp.LCA,'steam') && ~isempty(processData_msp.LCA.steam)
        steamVec = processData_msp.LCA.steam(:)';
    end
    steamVec = [steamVec zeros(1,maxSteamLength-length(steamVec))];
    for j = 1:maxSteamLength
        row.(sprintf('Steam_%d',j)) = steamVec(j);
    end

    % Refrigerant
    refVec = [];
    if isfield(processData_msp.LCA,'ref') && ~isempty(processData_msp.LCA.ref)
        refVec = processData_msp.LCA.ref(:)';
    end
    refVec = [refVec zeros(1,maxRefLength-length(refVec))];
    for j = 1:maxRefLength
        row.(sprintf('Ref_%d',j)) = refVec(j);
    end

    % Waste water treatment
    wwtVec = [];
    if isfield(processData_msp,'Output') && isfield(processData_msp.Output,'WASTE_W')
        wwtVec = processData_msp.Output.WASTE_W.Massflow(:)';
    end
    wwtVec = [wwtVec zeros(1,maxWWTLength-length(wwtVec))];
    for j = 1:maxWWTLength
        row.(sprintf('WWT_%d',j)) = wwtVec(j);
    end

    % Biomass (WF)
    wfVec = [];
    if isfield(processData_msp.LCA,'WF') && ~isempty(processData_msp.LCA.WF)
        wfVec = processData_msp.LCA.WF(:)';
    end
    wfVec = [wfVec zeros(1,maxWFLength-length(wfVec))];
    for j = 1:maxWFLength
        row.(sprintf('Biomass_%d',j)) = wfVec(j);
    end

    % Natural Gas (NG)
    ngVec = [];
    if isfield(processData_msp.LCA,'NG') && ~isempty(processData_msp.LCA.NG)
        ngVec = processData_msp.LCA.NG(:)';
    end
    ngVec = [ngVec zeros(1,maxNGLength-length(ngVec))];
    for j = 1:maxNGLength
        row.(sprintf('NG_%d',j)) = ngVec(j);
    end

    %% ----------------------------
    % Scenario data (support struct)
    %% ----------------------------
    if isfield(processData_msp.LCA,'scenario')
        sc = processData_msp.LCA.scenario;
        for f = 1:length(scenarioFields)
            fieldName = scenarioFields{f};
            if isfield(sc, fieldName)
                data = sc.(fieldName);
                if isnumeric(data)
                    vec = data(:)';
                    % tentukan panjang maksimal sesuai tipe
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
                            maxLen = length(vec);
                    end
                    vec = [vec zeros(1,maxLen-length(vec))];
                    for j = 1:maxLen
                        row.(sprintf('Scenario_%s_%d', fieldName,j)) = vec(j);
                    end
                elseif isstruct(data)
                    subFields = fieldnames(data);
                    for sf = 1:length(subFields)
                        subName = subFields{sf};
                        subVec = data.(subName);
                        if isnumeric(subVec)
                            subVec = subVec(:)';
                            subMaxLen = max(length(subVec),1); % minimal 1
                            for j = 1:subMaxLen
                                row.(sprintf('Scenario_%s_%s_%d', fieldName, subName, j)) = subVec(j);
                            end
                        end
                    end
                end
            end
        end
    end

    % Append row
    Results = [Results; row];

    % Release Aspen COM object
    release(h);

end

%% ----------------------------
% Save results to Excel
%% ----------------------------
excelPath = fullfile(folder, 'LCA_full_scenario_results.xlsx');
writetable(Results, excelPath, 'Sheet', 'Summary');
fprintf('\nData lengkap LCA + scenario berhasil disimpan ke Excel: %s\n\n', excelPath);
