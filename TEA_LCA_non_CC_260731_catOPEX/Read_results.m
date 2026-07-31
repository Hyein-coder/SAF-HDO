%% 1. Equipment Cost
% % fields_capex = fieldnames(processData_msp.CapitalCost);
% % equipment_name = fields_capex;
% % equipment_cost = zeros(length(fields_capex), 1);
% %
% % for i = 1:length(fields_capex)
% %     value = processData_msp.CapitalCost.(fields_capex{i});
% %     equipment_cost(i) = sum(value(:));
% % end
% %
% % total_equipment = sum(equipment_cost);
% %
% % equipment_name_total = [equipment_name; {'Total Equipment Cost'}];
% % equipment_cost_total = [equipment_cost; total_equipment];
% % EquipmentCost = [equipment_name_total, num2cell(equipment_cost_total)];
% %
% % assignin('base', 'EquipmentCost', EquipmentCost);

EC = processData_msp.CapitalCost;

equipment_name = {};
equipment_cost = [];

flash_cost = 0;

if isfield(EC, 'flash2') && ~isempty(EC.flash2)
    val = EC.flash2;
    if isnumeric(val)
        flash_cost = flash_cost + sum(val(:));
    elseif isstruct(val)
        for k = 1:numel(val)
            flds = fieldnames(val(k));
            for f = 1:numel(flds)
                if isnumeric(val(k).(flds{f}))
                    flash_cost = flash_cost + sum(val(k).(flds{f})(:));
                end
            end
        end
    end
end

if isfield(EC, 'flash3') && ~isempty(EC.flash3)
    val = EC.flash3;
    if isnumeric(val)
        flash_cost = flash_cost + sum(val(:));
    elseif isstruct(val)
        for k = 1:numel(val)
            flds = fieldnames(val(k));
            for f = 1:numel(flds)
                if isnumeric(val(k).(flds{f}))
                    flash_cost = flash_cost + sum(val(k).(flds{f})(:));
                end
            end
        end
    end
end

equipment_name{end+1,1} = 'flash';
equipment_cost(end+1,1) = flash_cost;

dryer_cost = 0;
separator_cost = 0;

if isfield(EC, 'separator') && ~isempty(EC.separator)
    val = EC.separator;
    
    if isnumeric(val)
        if size(val,2) >= 1
            dryer_cost = sum(val(:,1));
        end
        if size(val,2) >= 2
            separator_cost = sum(val(:,2));
        end
        
    elseif isstruct(val)
        % iterasi tiap struct
        for k = 1:numel(val)
            flds = fieldnames(val(k));
            for f = 1:numel(flds)
                if isnumeric(val(k).(flds{f}))
                    if f == 1
                        dryer_cost = dryer_cost + sum(val(k).(flds{f})(:));
                    elseif f == 2
                        separator_cost = separator_cost + sum(val(k).(flds{f})(:));
                    end
                end
            end
        end
    end
end

equipment_name{end+1,1} = 'dryer';
equipment_cost(end+1,1) = dryer_cost;

equipment_name{end+1,1} = 'separator';
equipment_cost(end+1,1) = separator_cost;

all_fields = fieldnames(EC);
for i = 1:length(all_fields)
    fname = all_fields{i};
    if any(strcmp(fname, {'flash2','flash3','separator','dryer'}))
        continue;
    end
    value = sum(EC.(fname)(:));
    equipment_name{end+1,1} = lower(fname);
    equipment_cost(end+1,1) = value;
end

total_equipment = sum(equipment_cost);
equipment_name = [equipment_name; {'total equipment cost'}];
equipment_cost = [equipment_cost; total_equipment];

EquipmentCost = [equipment_name, num2cell(equipment_cost)];
assignin('base', 'EquipmentCost', EquipmentCost);

%% 2. CAPEX
% econ_fields = {'C_alloc', 'C_cont', 'C_site', 'C_startup', ...
%     'C_land', 'C_comp', 'C_royal', 'C_serv', 'C_catalyst'};
% econ_name = {'Alloc', 'Contingency', 'SitePrep', ...
%     'Startup', 'Land', 'C_comp', 'Royalties', 'Services', 'Catalyst'};

econ_fields = {'C_alloc', 'C_WC', 'C_cont', 'C_site', 'C_startup', ...
    'C_land', 'C_comp', 'C_royal', 'C_serv'};
econ_name = {'Alloc', 'WorkingCapital', 'Contingency', 'SitePrep', ...
    'Startup', 'Land', 'C_comp', 'Royalties', 'Services'};
econ_cost = zeros(length(econ_fields),1);

for i = 1:length(econ_fields)
    if isfield(processData_msp.EconomicData, econ_fields{i})
        value = processData_msp.EconomicData.(econ_fields{i});
        econ_cost(i) = sum(value(:));
    end
end

capex_name = [{'Equipment Cost'}; econ_name'; {'Total CAPEX'}];
capex_cost = [total_equipment; econ_cost; total_equipment + sum(econ_cost)];
TPI = processData_msp.EconomicData.C_TPI;

CAPEX = [capex_name, num2cell(capex_cost)];
assignin('base', 'CAPEX', CAPEX);

%% 3. Feedstock & Utility
if isfield(processData_msp, 'OperatingCost')
    op_fields = fieldnames(processData_msp.OperatingCost);
    op_name = cellstr(string(op_fields));
    op_cost = zeros(length(op_fields),1);
    for i = 1:length(op_fields)
        value = processData_msp.OperatingCost.(op_fields{i});
        op_cost(i) = sum(value(:));
    end
else
    op_name = {};
    op_cost = [];
end

biomass_cost = 0;
ng_cost = 0;
if isfield(processData_msp.EconomicData, 'C_feed')
    feed_data = processData_msp.EconomicData.C_feed;
    if size(feed_data,2) >= 2
        biomass_cost = sum(feed_data(:,1));
        ng_cost = sum(feed_data(:,2));
    end
end

feed_util_name = [op_name; {'Biomass'}; {'Natural Gas'}; {'Total FeedUtility'}];
feed_util_cost = [op_cost; biomass_cost; ng_cost; sum(op_cost) + biomass_cost + ng_cost];
FeedUtilityCost = [feed_util_name, num2cell(feed_util_cost)];

assignin('base', 'FeedUtilityCost', FeedUtilityCost);

%% 4. OPEX
% feed_util_name = FeedUtilityCost(1:end-1,1);
% feed_util_cost = cell2mat(FeedUtilityCost(1:end-1,2));

if isfield(processData_msp.EconomicData, 'C_catalyst')
    processData_msp.EconomicData.C_catalyst_opex = (1/3) * processData_msp.EconomicData.C_catalyst;
else
    processData_msp.EconomicData.C_catalyst_opex = 0;
end

% processData_msp.EconomicData.C_cap_op = processData_msp.OperatingCost.Capture;

% opex_fields = {'GE', 'C_M', 'C_D', 'C_O', 'C_prop', 'C_OH', 'C_catalyst_opex'};
% opex_name = {'General Expenses', 'Maintenance', 'Depreciation', 'Operation', ...
%              'Property Taxes & Insurance', 'Operating Overhead', 'Catalyst OPEX'};

opex_fields = {'GE', 'C_M', 'C_D', 'C_O', 'C_prop', 'C_OH', ...
    'C_catalyst_opex', 'C_feed', 'C_util', 'C_cap_op'};
opex_name = {'General Expenses', 'Maintenance', 'Depreciation', 'Operation', ...
    'Property Taxes & Insurance', 'Operating Overhead', ...
    'Catalyst OPEX', 'Feed Cost', 'Utility Cost', 'Capture OPEX'};

opex_cost = zeros(length(opex_fields),1);

for i = 1:length(opex_fields)
    if isfield(processData_msp.EconomicData, opex_fields{i})
        value = processData_msp.EconomicData.(opex_fields{i});
        opex_cost(i) = sum(value(:));
    end
end

% all_name = [feed_util_name; opex_name'; {'Total OPEX'}];
% all_cost = [feed_util_cost; opex_cost; sum(feed_util_cost) + sum(opex_cost)];

all_name = [opex_name'; {'Total OPEX'}];
all_cost = [opex_cost; sum(opex_cost)];
opex_cashflow = processData_msp.EconomicData.C;

OPEX = [all_name, num2cell(all_cost)];
assignin('base', 'OPEX', OPEX);