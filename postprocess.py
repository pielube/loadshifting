
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt

ntestcases = 36

for j in range(ntestcases):
    
    nsim = 10
    
    ncase = j+1
    pathtofolder = r'.\simulations\case_{0}'.format(ncase)
    
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
    
    df_mean = df_tot.mean(axis=1)
    df_mean = df_mean.to_frame()
    
    rng = pd.date_range(start='2016-01-02',end='2016-01-03',freq='min')
    ax = df_tot.loc[rng].plot(figsize=(10,5),color='#b0c4de',legend=False)
    df_mean.loc[rng].plot(ax=ax,legend=False)
    
    figtitle = 'Case {0}'.format(ncase)
    ax.set_title(figtitle)
    ax.xaxis.set_label_text('Time [min]')
    ax.yaxis.set_label_text('Power [W]')
    
    fig = ax.get_figure()
    
    newpath = r'.\plots' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    figname = 'case_{0}.pdf'.format(ncase)
    
    figpath = os.path.join(newpath,figname)
    fig.savefig(figpath,format='pdf',bbox_inches='tight')
    
    plt.close(fig)


















