# Shifting strategies
## EV

How EV charging profile are obtained:
-	charging profile obtained from RAMP mobility
-	charging profile vector multiplied per occupancy vector of one of the household members (hereafter main driver)

**Main issue**: some discharging events might happen when main driver is at home, which does not make sense.

Considering the main issue, right now the shifting strategy is the following:
-	obtaining time windows in which main driver is at home
-	obtaining time windows in which the EV is charging and the corresponding consumptions
-	matching between at home time windows and charging time windows (multiple charging windows might be contained in one at home time window)
-	shifting is done skipping from one at-home time window to the following
-	per each time window we define:
    - initial Level of Charge (LOC), always equal to zero
    - maximum LOC: total consumption of charging events in the considered time window
    - minimum LOC: between zero and maximum LOC, ramping in correspondence of charging events.
-	if shifting based on PV: charging happens when PV is available (inv2load) or when LOC is below LOC_min (grid2load). LOC might be below LOC_min no because LOC diminishes, but because LOC_min increases.
-	if shifting based on tariffs: charging happens when energy prices are lower than threshold price or when LOC is lower than LOC_min

**Main issue**: not possible to store more energy in the battery than what was originally stored in the considered time window, regardless of the original state of charge of the battery.