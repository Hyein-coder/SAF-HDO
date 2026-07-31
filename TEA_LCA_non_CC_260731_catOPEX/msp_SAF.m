function y = msp_SAF(processData, COST, x)
processData = cash_flow_generic_SAF_opt(processData, COST, x);
% processData = cash_flow_generic_SAF_opt_mjsp(processData, COST, x);

y_NPV = processData.NPV;

y = -y_NPV;
end