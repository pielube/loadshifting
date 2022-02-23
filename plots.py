#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:29:18 2022

@author: sylvain
"""

import plotly.graph_objects as go


def make_demand_plot(idx,data,title='Consumption'):
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
        fig.add_trace(go.Scatter(
                name = 'BatteryGeneration',
                x = idx,
                y = data.loc[idx,'BatteryGeneration'],
                stackgroup='two',
                mode='none'               # this remove the lines
               ))
        cols.remove('BatteryGeneration')
    for key in cols:
        fig.add_trace(go.Scatter(
            name = key,
            x = idx,
            y = data.loc[idx,key],
            stackgroup='one',
            mode='none'               # this remove the lines
           ))

    fig.update_layout(title = 'Consommation avec déplacement de charge',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Puissance (kW)'
                      )
    
    return fig