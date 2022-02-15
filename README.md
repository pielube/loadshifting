# Load Shifting library

By Pietro Lubello (Uni. Firenze) and Sylvain Quoilin (Uni. Liège)

Suite of tools to model load shifting and demand side management at the residential level.

This repository contains a framework for the simulation of electrical and thermal demands of Belgian households.  
So far two models are being used:
- [StRoBe](https://github.com/open-ideas/StROBe)  
    For modelling the occupancy, use of appliances, lighting, internal heat gains and domestic hot water redrawals.
- [RAMP-mobility](https://github.com/RAMP-project/RAMP-mobility)  
    For the modelling of the EV charging.  

Alongside the models, some elements are directly included within the framework. It is the case of the thermal building model, the HP-based heating system and the electric boiler.

## License

The LoadShifting library is a free software licensed under the “European Union Public Licence" EUPL v1.2. It 
can be redistributed and/or modified under the terms of this license.