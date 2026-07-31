% %% 1. Equipment Cost
% 
% fields_capex = fieldnames(processData_msp.CapitalCost);
% equipment_name = fields_capex;  % cell array
% equipment_cost = zeros(length(fields_capex), 1);
% 
% for i = 1:length(fields_capex)
%     value = processData_msp.CapitalCost.(fields_capex{i});
%     equipment_cost(i) = sum(value(:));  % total cost
% end
% 
% total_equipment = sum(equipment_cost);
% equipment_name_total = [equipment_name; {'Total'}];
% equipment_cost_total = [equipment_cost; total_equipment];
% EquipmentCost = table(equipment_name_total, equipment_cost_total, ...
%     'VariableNames', {'Equipment', 'Cost_USD'});
% assignin('base', 'EquipmentCost', EquipmentCost);
% 
% %% 2. CAPEX 
% 
% econ_fields = {'C_alloc', 'C_WC', 'C_cont', 'C_site', 'C_startup', ...
%                'C_land', 'C_comp', 'C_royal', 'C_serv'};
% econ_name = {'Alloc', 'WorkingCapital', 'Contingency', 'SitePrep', 'Startup', ...
%              'Land', 'C_comp', 'Royalties', 'Services'};
% econ_cost = zeros(length(econ_fields),1);
% 
% for i = 1:length(econ_fields)
%     value = processData_msp.EconomicData.(econ_fields{i});
%     econ_cost(i) = sum(value(:));
% end
% 
% capex_name = [{'Equipment Cost'}; econ_name'];
% capex_cost = [total_equipment; econ_cost];
% total_capex = sum(capex_cost);
% capex_name_total = [capex_name; {'Total'}];
% capex_cost_total = [capex_cost; total_capex];
% 
% CAPEX = table(capex_name_total, capex_cost_total, ...
%     'VariableNames', {'Item', 'Cost_USD'});
% assignin('base', 'CAPEX', CAPEX);
% 
% %% 3. Feedstock and Utility Cost 
% op_fields = fieldnames(processData_msp.OperatingCost);
% op_name = cellstr(string(op_fields));  
% op_cost = zeros(length(op_fields),1);
% 
% for i = 1:length(op_fields)
%     value = processData_msp.OperatingCost.(op_fields{i});
%     op_cost(i) = sum(value(:));
% end
% 
% biomass_cost = sum(processData_msp.EconomicData.C_feed(:,1));
% ng_cost = sum(processData_msp.EconomicData.C_feed(:,2));
% 
% feed_util_name = [op_name; {'Biomass'}; {'Natural Gas'}];
% feed_util_cost = [op_cost; biomass_cost; ng_cost];
% 
% total_feed_util = sum(feed_util_cost);
% feed_util_name_total = [feed_util_name; {'Total'}];
% feed_util_cost_total = [feed_util_cost; total_feed_util];
% 
% FeedUtilityCost = table(feed_util_name_total, feed_util_cost_total, ...
%     'VariableNames', {'Item', 'Cost_USD'});
% assignin('base', 'FeedUtilityCost', FeedUtilityCost);
% 
% %% 4. OPEX Table
% feed_util_name = FeedUtilityCost.Item(1:end-1); 
% feed_util_cost = FeedUtilityCost.Cost_USD(1:end-1);
% 
% opex_fields = {'GE', 'C_M', 'C_D', 'C_O', 'C_prop', 'C_OH'};
% opex_name = {'General Expenses', 'Maintenance', 'Depreciation', 'Operation', ...
%              'Prop. Taxes & Insur.', 'Operating Overhead'};
% opex_cost = zeros(length(opex_fields),1);
% 
% for i = 1:length(opex_fields)
%     value = processData_msp.EconomicData.(opex_fields{i});
%     opex_cost(i) = sum(value(:));
% end
% 
% all_name = [feed_util_name; opex_name'];
% all_cost = [feed_util_cost; opex_cost];
% 
% total_opex = sum(all_cost);
% all_name_total = [all_name; {'Total'}];
% all_cost_total = [all_cost; total_opex];
% 
% OPEX = table(all_name_total, all_cost_total, ...
%     'VariableNames', {'Item', 'Cost_USD'});
% assignin('base', 'OPEX', OPEX);
