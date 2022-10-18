#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:29:18 2022

@author: sylvain
"""

import plotly.graph_objects as go
from defaults import defaultcolors


def make_demand_plot(idx,data,PV = None,title='Consumption'):
    '''
    Uses to plotly to generate a stacked consumption plot

    Parameters
    ----------
    idx : DateTime
        Index of the time period to be plotted.
    data : pandas.DataFrame
        Dataframe with the columns to be plotted. Its index should include idx.
    title : str, optional
        Title of the plot. The default is 'Consumption'.

    Returns
    -------
    Plotly figure.

    '''
    fig = go.Figure()
    cols = data.columns.tolist()
    if 'BatteryGeneration' in cols:
        cols.remove('BatteryGeneration')

    if PV is not None:
        fig.add_trace(go.Scatter(
                name = 'PV geneartion',
                x = idx,
                y = PV.loc[idx],
                stackgroup='three',
                fillcolor='rgba(255, 255, 126,0.5)',
                mode='none'               # this remove the lines
                          ))        

    for key in cols:
        fig.add_trace(go.Scatter(
            name = key,
            x = idx,
            y = data.loc[idx,key],
            stackgroup='one',
            fillcolor = defaultcolors[key],
            mode='none'               # this remove the lines
           ))
    

    fig.update_layout(title = title,
                      xaxis_title = 'Dates',
                      yaxis_title = 'Puissance (kW)'
                      )
    
    return fig

def make_pflow_plot(idx,pflows,title='Power flows'):
    '''
    Uses to plotly to generate a generation/consumption plot

    Parameters
    ----------
    idx : DateTime
        Index of the time period to be plotted.
    pflows : pandas.DataFrame
        Dataframe with the columns to be plotted. Its index should include idx.
    title : str, optional
        Title of the plot. The default is 'Consumption'.

    Returns
    -------
    Plotly figure.

    '''
    fig = go.Figure()

    fig.add_trace(go.Scatter(
         name = 'PV geneartion',
         x = idx,
         y = pflows.loc[idx,'pv'],
         stackgroup='one',
         fillcolor='rgba(255, 255, 126,0.5)',
         mode='none'               # this remove the lines
        ))
    fig.add_trace(go.Scatter(
         name = 'Battery Generation',
         x = idx,
         y = -pflows.loc[idx,'BatteryGeneration'],
         stackgroup='one',
         mode='none'               # this remove the lines
        ))    
    fig.add_trace(go.Scatter(
         name = 'Battery Consumption',
         x = idx,
         y = -pflows.loc[idx,'BatteryConsumption'],
         stackgroup='two',
         mode='none'               # this remove the lines
        ))    
    fig.add_trace(go.Scatter(
         name = 'Original Demand',
         x = idx,
         y = pflows.loc[idx,'demand_noshift'],
         line=dict(color='grey', width=1, dash='dash')
        ))  
    fig.add_trace(go.Scatter(
         name = 'Shifted Demand',
         x = idx,
         y = pflows.loc[idx,'demand_shifted_nobatt'],
         line=dict(color='black', width=1)
        ))  

    fig.update_layout(title = title,
                      xaxis_title = 'Dates',
                      yaxis_title = 'Puissance (kW)'
                      )
    
    return fig