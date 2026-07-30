function npv = msp_npv_SA(processData, COST, x)
% MSP_NPV_SA  NPV of the project when SAF is sold at price x ($/kg).
%   Root of npv(x)=0 is the Minimum Selling Price (MSP). Used with fzero,
%   which needs only base MATLAB (no Optimization Toolbox).
pd  = cash_flow_SA(processData, COST, x);
npv = pd.NPV;
end
