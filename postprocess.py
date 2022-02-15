
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# import calendar




# def cloudplots(ntestcases,nsim):
    
#     """
#     Function to create one figure per case with cloud plots

#     inputs:
#     ntestcases  int - number of test cases to be plotted
#     nsim        int - number of simulations per each case

#     outputs:
#     None

#     """
        
#     for j in range(ntestcases):
                
#         ncase = j+1
#         pathtofolder = r'.\simulations\case_{0}'.format(ncase)
    
#         # Opening result files
#         # Aggregating to have one single demand curve per case
#         # Storing all demand curve in one single dataframe (df_tot)
        
#         runname = 'run_1'
#         filename = runname+'.pkl'
#         path = os.path.join(pathtofolder,filename)
#         df_tot = pd.read_pickle(path)
#         df_tot = df_tot.sum(axis=1)
#         df_tot = df_tot.to_frame()
#         df_tot.columns=[runname]
        
#         for i in range(nsim-1):
#             nrun = i+2
#             runname = 'run_{0}'.format(nrun)
#             filename = runname+'.pkl'
#             path = os.path.join(pathtofolder,filename)
#             df = pd.read_pickle(path)
#             df = df.sum(axis=1)
#             df = df.to_frame()
#             df.columns=[runname]
#             df_tot = df_tot.join(df)
    
#         # Calculating mean demand
        
#         df_mean = df_tot.mean(axis=1)
#         df_mean = df_mean.to_frame()
     
#         # Calculating most representative curve
#         # as the curve minimizing its R2 wrt the mean curve    
     
#         bestr2 = -float('inf')
#         bestr2_index = 0
    
#         for j in range(nsim):
#             r2 = (r2_score(df_mean[0], df_tot.iloc[:,j]))
#             print(r2)
#             if r2 > bestr2:
#                 bestr2 = r2
#                 bestr2_index = j
    
#         # Plotting one day of the year
        
#         rng = pd.date_range(start='2016-08-06',end='2016-08-07',freq='min')
#         ax = df_tot.loc[rng].plot(figsize=(8,4),color='#b0c4de',legend=False)
#         bestr2_col = df_tot.columns[bestr2_index]
#         df_mean.loc[rng].plot(ax=ax,legend=False)
#         df_tot.loc[rng,bestr2_col].plot(ax=ax,color ='red',legend=False)
    
        
#         figtitle = 'Case {0}'.format(ncase)
#         ax.set_title(figtitle)
#         ax.xaxis.set_label_text('Time [min]')
#         ax.yaxis.set_label_text('Power [W]')
        
    
#         # Saving figure
        
#         fig = ax.get_figure()
        
#         newpath = r'.\simulations\plots' 
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         figname = 'case_{0}.png'.format(ncase)
        
#         figpath = os.path.join(newpath,figname)
#         fig.savefig(figpath,format='png',bbox_inches='tight')
        
#         plt.close(fig)
        


# def prosumpyfeeder(ntestcases,nsim):

#     """
#     Function to adapt demand curves to be used by prosumpy

#     inputs:
#     ntestcases  int - number of test cases to be plotted
#     nsim        int - number of simulations per each case

#     outputs:
#     None

#     """
    
#     for j in range(ntestcases):
                
#         ncase = j+1
#         pathtofolder = r'.\simulations\case_{0}'.format(ncase)
        
#         for i in range(nsim):
            
#             # Reading results
#             nrun = i+1
#             runname = 'run_{0}'.format(nrun)
#             filename = runname+'.pkl'
#             path = os.path.join(pathtofolder,filename)
#             df = pd.read_pickle(path)
#             # Aggregating electricity consumptions in one demand
#             df = df.sum(axis=1)
#             # Resampling at 15 min
#             df = df.to_frame()
#             df = df.resample('15Min').mean()
#             # Extracting ref year used in the simulation
#             df.index = pd.to_datetime(df.index)
#             year = df.index.year[0]
#             # If ref year is leap remove 29 febr
#             leapyear = calendar.isleap(year)
#             if leapyear:
#                 start_leap = str(year)+'-02-29 00:00:00'
#                 stop_leap = str(year)+'-02-29 23:45:00'
#                 daterange_leap = pd.date_range(start_leap,stop_leap,freq='15min')
#                 df = df.drop(daterange_leap)
#             # Remove last row if is from next year
#             nye = pd.Timestamp(str(year+1)+'-01-01 00:00:00')
#             df = df.drop(nye)
#             # New reference year 2015, to be used with TMY from pvlib
#             start_ref = '2015-01-01 00:00:00'
#             end_ref = '2015-12-31 23:45:00'
#             daterange_ref = pd.date_range(start_ref,end_ref,freq='15min')
#             df = df.set_index(daterange_ref)
#             # Saving
#             newpath = r'.\simulations\prosumpy_case_'+str(ncase) 
#             if not os.path.exists(newpath):
#                 os.makedirs(newpath)
#             filename = 'run_{0}.pkl'.format(nrun)
#             filename = os.path.join(newpath,filename)
#             df.to_pickle(filename)
        


import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'


house = '4f'
index = 5

# # Demands
# name = house+'.pkl'
# path = r'./simulations'
# file = os.path.join(path,name)
# demands = pd.read_pickle(file)
# demand = demands[index]

# Occupancy
name = house+'_occ.pkl'
path = r'./simulations'
file = os.path.join(path,name)
occs = pd.read_pickle(file)

# Demands
name = 'test.pkl'
path = r'./simulations/results'
file = os.path.join(path,name)
demand = pd.read_pickle(file)
# demand = demands[index]

# PV
pvpeak = 10.
pvfile = r'./simulations/pv.pkl'
pvadim = pd.read_pickle(pvfile)
pv = pvadim * pvpeak # kW

# Plot
rng = pd.date_range(start='2015-07-19',end='2015-07-24',freq='15min')

df =  demand.loc[rng]/1000.
ymax = max(np.max(pv.iloc[:,0][rng]),np.max(df.sum(axis=1)))*1.2
ymax = int(ymax)+1









# Adimissibile time windows according to occupancy
occ = np.zeros(len(occs[index][0]))
for i in range(len(occs[index])):
    occs[index][i] = [1 if a==1 else 0 for a in occs[index][i]]
    occ += occs[index][i]
    
occ = [1 if a >=1 else 0 for a in occ]    
occ = occ[:-1].copy()
occupancy = np.zeros(len(demand['StaticLoad']))
for i in range(len(occ)):
    for j in range(10):
        occupancy[i*10+j] = occ[i]
occupancy[-1] = occupancy[-2]

occupancy=pd.DataFrame(data=occupancy,index=demand.index)
occupancy=occupancy.loc[rng]





# cols = list(df.columns)
# shiftcols = ['TumbleDryerShift','DishWasherShift','WashingMachineShift']
# nonshiftcols = [x for x in cols if x not in shiftcols]
# #colors = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#FF6692','#B6E880','#FF97FF','#FECB52']
# coldict = {'StaticLoad': '#636EFA',
#            'TumbleDryer':'#EF553B',
#            'DishWasher':'#00CC96',
#            'WashingMachine':'#AB63FA',
#            'DomesticHotWater':'#FFA15A',
#            'HeatPumpPower':'#19D3F3',
#            'EVCharging':'#FF6692',
#            'WashingMachineShift':'#AB63FA',
#            'TumbleDryerShift':'#EF553B',
#            'DishWasherShift':'#00CC96'}
# traces = []

# # PV trace
# marker=dict(color='goldenrod')
# trace = go.Scatter(x=pv.loc[rng].index,
#                    y=pv.loc[rng][0],
#                    name='PV',
#                    marker=marker,
#                    fill='tonexty',
#                    fillcolor='rgba(218, 165, 32, 0.15)',
#                    yaxis='y3')#,legendgroup=1)
# traces.append(trace)

# # Demand traces
# for col in nonshiftcols:
    
#     color = coldict[col]
#     marker=dict(color=color)
#     pattern = None
#     trace = go.Bar(x=df.index,
#                    y=df[col],
#                    name=col,
#                    marker=marker,
#                    marker_pattern_shape=pattern)#,egendgroup=2)
#     traces.append(trace)
    
# # Shifted demand traces
# for col in shiftcols:
    
#     color = coldict[col]
#     marker=dict(color=color)
#     pattern = '/'
#     # base = True
#     trace = go.Bar(x=df.index,
#                    y=df[col],
#                    name=col,
#                    # base=base,
#                    marker=marker,
#                    marker_pattern_shape=pattern,yaxis='y2')#,legendgroup=3)
#     traces.append(trace) 
    

    
# layout = go.Layout(yaxis2=dict(overlaying='y'),
#                    yaxis3=dict(overlaying='y'),
#                    barmode='stack',
#                    legend={'traceorder':'grouped'})

# fig = go.Figure(data=traces,
#                 layout=layout)

# fig.update_yaxes(range = [0,ymax])

# fig.add_vrect(x0="2015-07-19 15:00:00", x1="2015-07-19 17:00:00", 
#               # annotation_text="Admissible start time", 
#               # annotation_position="top left",
#               fillcolor="green", 
#               opacity=0.15, 
#               line_width=0)

# fig.show()




cols = list(df.columns)
preshift = ['TumbleDryer','DishWasher','WashingMachine']
postshift = ['TumbleDryerShift','DishWasherShift','WashingMachineShift']

cols_static = [x for x in cols if x not in preshift+postshift]

coldict = {'StaticLoad': '#636EFA',
           'TumbleDryer':'#EF553B',
           'DishWasher':'#00CC96',
           'WashingMachine':'#AB63FA',
           'DomesticHotWater':'#FFA15A',
           'HeatPumpPower':'#19D3F3',
           'EVCharging':'#FF6692',
           'WashingMachineShift':'#AB63FA',
           'TumbleDryerShift':'#EF553B',
           'DishWasherShift':'#00CC96'}
traces = []

# PV trace
marker=dict(color='goldenrod')
trace = go.Scatter(x=pv.loc[rng].index,
                   y=pv.loc[rng][0],
                   name='PV',
                   marker=marker,
                   fill='tonexty',
                   fillcolor='rgba(218, 165, 32, 0.15)',
                   yaxis='y2',legendgroup=1)
traces.append(trace)

# Static + won't be shifted demand traces
for col in cols_static:
    
    color = coldict[col]
    marker=dict(color=color)
    pattern = None
    trace = go.Bar(x=df.index,
                   y=df[col],
                   name=col,
                   marker=marker,
                   marker_pattern_shape=pattern,
                   legendgroup=2)
    traces.append(trace)
    
# Will be shifted demand traces: pre-shift
for col in preshift:
    
    color = coldict[col]
    marker=dict(color=color)
    #base = True
    trace = go.Bar(x=df.index,
                   y=df[col],
                   name=col,
                   # base=base,
                   marker=marker,
                   legendgroup=3)
    traces.append(trace)

# Will be shifted demand traces: post-shift
for col in postshift:
    
    color = coldict[col]
    marker=dict(color=color)
    pattern = '/'
    #base = True
    trace = go.Bar(x=df.index,
                   y=df[col],
                   name=col,
                   # base=base,
                   marker=marker,
                   marker_pattern_shape=pattern,
                   legendgroup=4)
    traces.append(trace) 
    

    
layout = go.Layout(yaxis2=dict(overlaying='y'),
                   barmode='stack',
                   legend={'traceorder':'grouped'})

fig = go.Figure(data=traces,
                layout=layout)

fig.update_yaxes(range = [0,ymax])

fig.add_vrect(x0="2015-07-19 15:00:00", x1="2015-07-19 17:00:00", 
              fillcolor="green", 
              opacity=0.15, 
              line_width=0)

fig.show()








