
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os

nsim = 10
pathtofolder = r'.\simulations\case_1'
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
ax = df_tot.loc[rng].plot(color='#b0c4de',legend=False)
df_mean.loc[rng].plot(ax=ax,legend=False)