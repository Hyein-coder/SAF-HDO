function [cost,V] = NPV_Pyrolyzer_NREL(LFlow,MBiomass,VesselPressure,UPF,V)
if (abs(LFlow) > 0.000001)
    
    % Function which calculates capital flash cost
    % Input in SI
    %
    % INPUT
    %------------------------------------------------
    % Name           | Description                                | Size |
    % LFlow          | Liquid mass flow leaving the flash         |[1x1] |
    % MBiomass       | Biomass mass flow entering the reactor     |[1x1] |
    % VesselPressure | Pressure in flash                          |[1x1] |
    % UPF            | Upscale factor                             |[1x1] |
    %------------------------------------------------
    %
    % OUTPUT
    %------------------------------------------------
    % Name  | Description                                         | Size |
    % cost  | upscaled capital cost of flash in dollar           |[1x1] |
    %------------------------------------------------
    
    %assumptions: Horizontal fabrication, L/D=4
    
    Mraw = MBiomass * UPF;

%     UPF
    %Cost calculation
    M0 = 400; % mtpd
    C0= 38000000; 
    beta = 0.7;
    
    %Calculating whether Size is within Costing Range, otherwise dividing unit
    %into several units
    Mmax = 1000;    % According to NREL
    if Mraw <= Mmax
        Amount = 1;
        M = Mraw;
    else
        Amount = ceil(Mraw/Mmax);
        M = Mraw/Amount;
    end

    BC = C0*((M/M0)^beta);
        
    BMC = BC; %updated bare module cost
    cost = BMC*Amount;
    
else
    cost = 0.0 ;
end

end