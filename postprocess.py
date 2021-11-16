
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import calendar




def cloudplots(ntestcases,nsim):
    
    """
    Function to create one figure per case with cloud plots

    inputs:
    ntestcases  int - number of test cases to be plotted
    nsim        int - number of simulations per each case

    outputs:
    None

    """
        
    for j in range(ntestcases):
                
        ncase = j+1
        pathtofolder = r'.\simulations\case_{0}'.format(ncase)
    
        # Opening result files
        # Aggregating to have one single demand curve per case
        # Storing all demand curve in one single dataframe (df_tot)
        
        runname = 'run_1'
        filename = runname+'.pkl'
        path = os.path.join(pathtofolder,filename)
        df_tot = pd.read_pickle(path)
        df_tot = df_tot.sum(axis=1)
        df_tot = df_tot.to_frame()
        df_tot.columns=[runname]
        
        for i in range(nsim-1):
            nrun = i+2
            runname = 'run_{0}'.format(nrun)
            filename = runname+'.pkl'
            path = os.path.join(pathtofolder,filename)
            df = pd.read_pickle(path)
            df = df.sum(axis=1)
            df = df.to_frame()
            df.columns=[runname]
            df_tot = df_tot.join(df)
    
        # Calculating mean demand
        
        df_mean = df_tot.mean(axis=1)
        df_mean = df_mean.to_frame()
     
        # Calculating most representative curve
        # as the curve minimizing its R2 wrt the mean curve    
     
        bestr2 = -float('inf')
        bestr2_index = 0
    
        for j in range(nsim):
            r2 = (r2_score(df_mean[0], df_tot.iloc[:,j]))
            print(r2)
            if r2 > bestr2:
                bestr2 = r2
                bestr2_index = j
    
        # Plotting one day of the year
        
        rng = pd.date_range(start='2016-08-06',end='2016-08-07',freq='min')
        ax = df_tot.loc[rng].plot(figsize=(8,4),color='#b0c4de',legend=False)
        bestr2_col = df_tot.columns[bestr2_index]
        df_mean.loc[rng].plot(ax=ax,legend=False)
        df_tot.loc[rng,bestr2_col].plot(ax=ax,color ='red',legend=False)
    
        
        figtitle = 'Case {0}'.format(ncase)
        ax.set_title(figtitle)
        ax.xaxis.set_label_text('Time [min]')
        ax.yaxis.set_label_text('Power [W]')
        
    
        # Saving figure
        
        fig = ax.get_figure()
        
        newpath = r'.\simulations\plots' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        figname = 'case_{0}.png'.format(ncase)
        
        figpath = os.path.join(newpath,figname)
        fig.savefig(figpath,format='png',bbox_inches='tight')
        
        plt.close(fig)
        


def prosumpyfeeder(ntestcases,nsim):

    """
    Function to adapt demand curves to be used by prosumpy

    inputs:
    ntestcases  int - number of test cases to be plotted
    nsim        int - number of simulations per each case

    outputs:
    None

    """
    
    for j in range(ntestcases):
                
        ncase = j+1
        pathtofolder = r'.\simulations\case_{0}'.format(ncase)
        
        for i in range(nsim):
            
            # Reading results
            nrun = i+1
            runname = 'run_{0}'.format(nrun)
            filename = runname+'.pkl'
            path = os.path.join(pathtofolder,filename)
            df = pd.read_pickle(path)
            # Aggregating electricity consumptions in one demand
            df = df.sum(axis=1)
            # Resampling at 15 min
            df = df.to_frame()
            df = df.resample('15Min').mean()
            # Extracting ref year used in the simulation
            df.index = pd.to_datetime(df.index)
            year = df.index.year[0]
            # If ref year is leap remove 29 febr
            leapyear = calendar.isleap(year)
            if leapyear:
                start_leap = str(year)+'-02-29 00:00:00'
                stop_leap = str(year)+'-02-29 23:45:00'
                daterange_leap = pd.date_range(start_leap,stop_leap,freq='15min')
                df = df.drop(daterange_leap)
            # Remove last row if is from next year
            nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00')
            df = df.drop(nye)
            # New reference year 2015, to be used with TMY from pvlib
            start_ref = '2015-01-01 00:00:00'
            end_ref = '2015-12-31 23:45:00'
            daterange_ref = pd.date_range(start_ref,end_ref,freq='15min')
            df = df.set_index(daterange_ref)
            # Saving
            newpath = r'.\simulations\prosumpy_case_'+str(ncase) 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            filename = 'run_{0}.pkl'.format(nrun)
            filename = os.path.join(newpath,filename)
            df.to_pickle(filename)
        

ntestcases = 1
nsim = 10

# cloudplots(ntestcases,nsim)
prosumpyfeeder(ntestcases,nsim)





















