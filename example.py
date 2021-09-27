import os
import sys

strobeDir = os.path.dirname(os.path.realpath(__file__)) # get path where this file is (StROBe path)
sys.path.append(os.path.join(strobeDir, 'Corpus')) 

from strobe import feeder as fee
import strobe

# Create and simulate a single household, with given type of members, and given year
family = strobe.Household("Example household", members=['FTE', 'Unemployed'])
family.simulate(year=2013, ndays=365)

family.__dict__ # list all elements of family for inspection


# Simulate households for an entire feeder

cleanup = True #choose whether individual household results will be deleted or not
# create folder where simulations will be stored for feeder
dataDir = "Simulations"

absdatadir = os.path.abspath(dataDir)
if not os.path.exists(absdatadir):
    os.mkdir(absdatadir)
    
# Test feeder with 5 households, temperatures given in Kelvin
fee.IDEAS_Feeder(name='Household',nBui=5, path=absdatadir, sh_K=True)

# cleanup pickled household files from new folder (only keep text files with results)
if cleanup:
    for file in os.listdir(absdatadir):
        print(file)
        if file.endswith(('.p')):
            os.remove(os.path.join(absdatadir, file))
