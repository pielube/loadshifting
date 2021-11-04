
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


ntestcases = 1

for j in range(ntestcases):
    
    nsim = 10
    
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
    
    rng = pd.date_range(start='2016-01-02',end='2016-01-03',freq='min')
    ax = df_tot.loc[rng].plot(figsize=(10,5),color='#b0c4de',legend=False)
    bestr2_col = df_tot.columns[bestr2_index]
    df_mean.loc[rng].plot(ax=ax,legend=False)
    df_tot.loc[rng,bestr2_col].plot(ax=ax,legend=False,color ='red')

    
    figtitle = 'Case {0}'.format(ncase)
    ax.set_title(figtitle)
    ax.xaxis.set_label_text('Time [min]')
    ax.yaxis.set_label_text('Power [W]')

    # Saving figure
    
    fig = ax.get_figure()
    
    newpath = r'.\plots' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    figname = 'case_{0}.pdf'.format(ncase)
    
    figpath = os.path.join(newpath,figname)
    fig.savefig(figpath,format='pdf',bbox_inches='tight')
    
    plt.close(fig)
    
























