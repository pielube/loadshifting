#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:09:09 2021

@author: sylvain
"""
import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_bootstrap_components as dbc    # for the css
from dash.dependencies import Input, Output, State

import os
import numpy as np
import pandas as pd
import datetime, calendar
import strobe
import json
import pickle

def simulate_load():
    
    # Reading JSON
    with open('inputs/loadshift_inputs.json') as f:
        inputs = json.load(f)
    # Ambient data
    temp, irr = strobe.ambientdata()
    
    # Strobe
    res_el,res_DHW,Tbath,res_Qgains,textoutput = strobe.simulate_scenarios(1,inputs)
    
    # House heating model
    ressize = np.size(res_el['pstatic'])
    phi_c = res_Qgains['Qrad']+res_Qgains['Qcon']
    timersetting = strobe.HeatingTimer(inputs)
    phi_h_space,Tem_test = strobe.HouseThermalModel(inputs,ressize,temp,irr,phi_c,timersetting)
    thermal_load = int(sum(phi_h_space)/1000./60.)
    print(' - Thermal demand for space heating is ',thermal_load,' kWh')
    
    # Heat pump electric load
    phi_hp = strobe.ElLoadHP(temp,phi_h_space)
    
    # Electric boiler and hot water tank
    phi_a = strobe.HotWaterTankModel(inputs,res_DHW['mDHW'],Tbath)
    
    # Creating dataframe with the time-series results
    df = pd.DataFrame(data=res_el)
    df['elboiler'] = phi_a
    df['heatpump'] = phi_hp
    time = list(range(0,np.size(res_el['pstatic'])))
    time = [datetime.datetime(2020,1,1) + datetime.timedelta(minutes=each) for each in time]
    df.index = time
    
    # Generating the dictionnary with the aggregated results
    results = {}
    results['thermal_load'] = thermal_load
    results['textoutput'] = textoutput
    results['timeseries'] = df
    results['inputs'] = inputs
    
    return results

# List of bootstrap themes for dash: https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
    
app.layout = html.Div(id = 'parent', children = [
        html.H1(id = 'H1', children = 'Load generator display', style = {'textAlign':'center',\
                                                'marginTop':40,'marginBottom':40}),

        html.H2(id='text1', children='Simulation parameters:'),

        dcc.Checklist(
                        options=[
                            {'label': 'Washing Machine', 'value': 'wp'},
                            {'label': 'Tumble Dryer', 'value': 'td'},
                            {'label': 'Dish Washer', 'value': 'dw'},
                            {'label': 'Electric boiler', 'value': 'eb'},
                            {'label': 'Heat pump', 'value': 'hp'},
                            {'label': 'Electric Vehicle', 'value': 'ev'}
                        ],
                        value=['wp', 'td','dw','eb','hp'],
                        labelStyle={'display': 'inline-block'}
                    ),
        html.H2(id='text_household_composition', children='Composition du ménage'),
        dcc.Dropdown( id = 'dropdown_FTE',
                        options = [ {'label':'0 FTE', 'value': 0}, 
                                   {'label':'1 FTE', 'value': 1},
                                   {'label':'2 FTE', 'value': 2},
                                   {'label':'3 FTE', 'value': 3}],
                        value = 2),
        dcc.Dropdown( id = 'dropdown_Unemployed',
                options = [ {'label':'0 Unemployed', 'value': 0}, 
                           {'label':'1 Unemployed', 'value': 1},
                           {'label':'2 Unemployed', 'value': 2},
                           {'label':'3 Unemployed', 'value': 3}],
                value = 0),
        dcc.Dropdown( id = 'dropdown_School',
                        options = [ {'label':'0 School', 'value': 0}, 
                                   {'label':'1 School', 'value': 1},
                                   {'label':'2 School', 'value': 2},
                                   {'label':'3 School', 'value': 3}],
                        value = 1),        
        dcc.Dropdown( id = 'dropdown_Retired',
                         options = [ {'label':'0 Retired', 'value': 0}, 
                                    {'label':'1 Retired', 'value': 1},
                                    {'label':'2 Retired', 'value': 2},
                                    {'label':'3 Retired', 'value': 3}],
                         value = 1), 
        
        html.H2(id = 'H2', children = 'Heat pump parameters', style = {'textAlign':'left'}),        
        html.Div(id='text_house_type', children='House type:'),
        dcc.Dropdown( id = 'dropdown2',
                        options = [ {'label':'4 Façades', 'value': '4'}, 
                                   {'label':'3 Façades', 'value': '3'},
                                   {'label':'2 Façades', 'value': '2'},
                                   {'label':'Appartement', 'value': 'flat'}],
                        value = '4'),
        html.Div(id='text_hp_power', children='Heat pumpe thermal power (W):'),
        html.Div(dcc.Input(id='input_hp_pwoer', type='text',value=5000)),
        
        html.H2(id = 'title_boiler', children = 'Electrical boiler paramters', style = {'textAlign':'left'}),        
        html.Div(id='text_volume', children='Volume (l):'),
        html.Div(dcc.Input(id='input_boiler_volume', type='text',value=200)),
        html.Div(id='text_boiler_temperature', children='Set point temperature (°C):'),
        html.Div(dcc.Input(id='input_boiler_temperature', type='text',value=53)),

        html.H2(id='text2', children='Time parameters:'),
        dcc.Dropdown( id = 'dropdown',
        options = [ {'label':calendar.month_name[x], 'value': calendar.month_name[x]} for x in range(1 ,13)]),
        html.Div(dcc.Input(id='textbox', type='text',value=2020)),
        html.Button('Submit', id='simulate', n_clicks=0),
        html.Div(id='text', children='Enter a value and press submit'),
        dcc.Graph(id = 'plot'),
        html.Div(id='outputtext', children='')
    ])    
    
@app.callback(Output(component_id='plot', component_property= 'figure'),
              Output(component_id='text', component_property= "children"),
              Output(component_id='outputtext', component_property= "children"),
              [Input(component_id='simulate', component_property= 'n_clicks'),
               Input(component_id='textbox', component_property= 'value')],
              [State(component_id='dropdown', component_property= 'value')])
def simulate_button(N,textvalue,value):
    '''
    We need as many arguments to the function as there are inputs and states
    Inputs trigger a callback 
    States are used as parameters but do not trigger a callback

    '''
    if value is None:
        todisplay = 'Not showing any data'
        n_month=0
    else:
        todisplay = 'Showing data for ' + value + ' (' + str(N) + ')'
        n_month = list(calendar.month_name).index(value)
    

    print(todisplay)
    
    if os.path.isfile('data.p'):
        #results = pd.read_pickle('data.p')           # temporary file to speed up things
        results = pickle.load(open('data.p','rb'))
    else:
        results = simulate_load()
        pickle.dump(results,open(b"data.p","wb"))
    load = results['timeseries']
    # fig = go.Figure([go.Scatter(x = load.index, y = load['{}'.format(value)],\
    #                   line = dict(color = 'firebrick', width = 4))
    #                   ])    

#    fig = px.area(load, load.index)
    idx = load.index[(load.index.month==n_month) & (load.index.year==int(textvalue))]
    fig = go.Figure()
    for key in load:
        fig.add_trace(go.Scatter(
            name = key,
            x = idx,
            y = load.loc[idx,key],
            stackgroup='one',
            mode='none'               # this remove the lines
           ))

    fig.update_layout(title = 'Power consumption',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Load in W'
                      )
    totext = []
    for x in results['textoutput']:
        totext.append(x)
        totext.append(html.Br())   
#    totext = '<br>'.join(results['textoutput'])
    
    return fig,todisplay,totext     # Number of returns must be equal to the number of outputs

if __name__ == '__main__': 
    app.run_server()