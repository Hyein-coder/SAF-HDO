function [cost,V] = NPV_clarifier_2020(SFlow,UPF)
if (abs(SFlow) > 0.000001)
    
    % Function which calculates capital thickener cost (22/01/18)
    % Input in SI
    % Cost model based on Seider et al.(2009), Table 16.32 (pp.553-556)
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
    % Name  | Description                                        | Size |
    % cost  | upscaled capital cost of flash in dollar           |[1x1] |
    % V     | Settling area of the thickener                     |[1x1]|
    %------------------------------------------------
    
    %assumptions: Horizontal fabrication, L/D=4
    
%     VesselPressure = VesselPressure/10^5; %Pa to bar
    SFlow_ton = UPF * SFlow * (0.001) * (24) ; % [kg/hr] to [ton/day]
    
    %Cost calculation
    UF = 596.2/394; % CEPCI for 2020.
    FBM = 3.291;    % bare-module factor, from p.490
    SettArea = SFlow_ton ;      % X [ft2] = C1 [ton/day] (estimation)
    FOB = 2400*(SettArea^0.58) ;    % FOB of thickener, carbon steel
    BMC = UF*FBM*FOB ;      % Need to check conversion from FOB to BMC 
    cost = BMC ; 
    V = SettArea;
else
    cost = 0.0 ;
    V = 0;
end

end