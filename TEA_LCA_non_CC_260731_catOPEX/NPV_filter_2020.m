function [cost,V] = NPV_filter_2020(SFlow,UPF)
if (abs(SFlow) > 0.000001)
    
    % Function which calculates capital flash cost
    % Input in SI
    %
    % INPUT
    %------------------------------------------------
    % Name           | Description                                | Size |
    % SFlow          | Solid mass flow of stream                  |[1x1] |
    % UPF            | Upscale factor                             |[1x1] |
    %------------------------------------------------
    %
    % OUTPUT
    %------------------------------------------------
    % Name  | Description                                         | Size |
    % cost  | upscaled capital cost of flash in dollar            |[1x1] |
    % V     | Settling area of the filter                         |[1x1]|
    %------------------------------------------------
    
    %assumptions: Horizontal fabrication, L/D=4
    
%     VesselPressure = VesselPressure/10^5; %Pa to bar
    SFlow_kg = UPF * SFlow * (24) ; % [kg/hr] to [kg/day]
    
    %Cost calculation
    UF = 596.2/394; % CEPCI for 2020.
    FBM = 3.291;    % bare-module factor, from p.490
    SettArea = SFlow_kg ./ 681 ;      % X [ft2] = C1 [kg/day] / (1500 lb/day-ft2) (estimation)
    FOB = exp(11.432-0.1905*log(SettArea)+0.0554*(log(SettArea)^2)) ;    % FOB of filter, carbon steel
    BMC = UF*FBM*FOB ;      % Need to check conversion from FOB to BMC 
    cost = BMC ; 
    V = SettArea;
    
else
    cost = 0.0 ;
    V = 0;
end

end