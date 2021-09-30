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
import calendar
import strobe,ramp
import json
import pickle

from typing import Dict, Any      # for the dictionnary hashing funciton
import hashlib  # for the dictionnary hashing funciton

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def simulate_load(inputs):
    
    # Strobe
    result,textoutput = strobe.simulate_scenarios(1, inputs)
    
    n_scen = 0 # Working only with the first scenario
    
    # RAMP-mobility
    if inputs['EV']:
        result_ramp = ramp.EVCharging(inputs, result['occupancy'][n_scen])
    else:
        result_ramp=pd.DataFrame()
    
    # Creating dataframe with the results 
    n_steps = np.size(result['StaticLoad'][n_scen,:])
    index = pd.date_range(start='2016-01-01 00:00', periods=n_steps, freq='1min')
    df = pd.DataFrame(index=index,columns=['StaticLoad','TumbleDryer','DishWasher','WashingMachine','ElectricalBoiler','HeatPumpPower','EVCharging'])
    
    result_ramp.loc[df.index[-1],'EVCharging']=0
    #df.index.union(result_ramp.index)        # too slow
    
    for key in df.columns:
        if key in result:
            df[key] = result[key][n_scen,:]
        elif key in result_ramp:
            df[key] = result_ramp[key]* 1000
            textoutput.append(' - Total EV load: ' + str(int(result_ramp[key].sum()/60)) + ' kWh')
        else:
            df[key] = 0
    
    # Generating the dictionnary with the aggregated results
    results = {}
    results['textoutput'] = textoutput
    results['timeseries'] = df
    results['inputs'] = inputs
    
    return results

# List of bootstrap themes for dash: https://www.bootstrapcdn.com/bootswatch/
#app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app = dash.Dash()
    
app.layout = html.Div(id = 'parent', children = [
        html.H1(id = 'H1', children = 'Générateur de courbes de demande', style = {'textAlign':'center',\
                                                'marginTop':40,'marginBottom':40}),

        html.H2(id='text1', children='Paramètres:'),

        dcc.Checklist(
                        id = 'checklist_apps',
                        options=[
                            {'label': 'Machine à laver', 'value': 'wm'},
                            {'label': 'Séchoir', 'value': 'td'},
                            {'label': 'Lave vaisselle', 'value': 'dw'},
                            {'label': 'Chauffe-eau électrique', 'value': 'eb'},
                            {'label': 'Pompe à Chaleur', 'value': 'hp'},
                            {'label': 'Véhicule électrique', 'value': 'ev'}
                        ],
                        value=['wm', 'td','dw','eb','hp'],
                        labelStyle={'display': 'inline-block'}
                    ),
        html.H2(id='text_household_composition', children='Composition du ménage'),
        dcc.Dropdown( id = 'dropdown_FTE',
                        options = [ {'label':'0 Travailleur temps plein', 'value': 0}, 
                                   {'label':'1 Travailleur temps plein', 'value': 1},
                                   {'label':'2 Travailleurs temps plein', 'value': 2},
                                   {'label':'3 Travailleurs temps plein', 'value': 3}],
                        value = 2),
        dcc.Dropdown( id = 'dropdown_Unemployed',
                options = [ {'label':'0 Inactif', 'value': 0}, 
                           {'label':'1 Inactif', 'value': 1},
                           {'label':'2 Inactifs', 'value': 2},
                           {'label':'3 Inactifs', 'value': 3}],
                value = 0),
        dcc.Dropdown( id = 'dropdown_School',
                        options = [ {'label':'0 Enfant', 'value': 0}, 
                                   {'label':'1 Enfant', 'value': 1},
                                   {'label':'2 Enfants', 'value': 2},
                                   {'label':'3 Enfants', 'value': 3}],
                        value = 1),        
        dcc.Dropdown( id = 'dropdown_Retired',
                         options = [ {'label':'0 Retraité', 'value': 0}, 
                                    {'label':'1 Retraité', 'value': 1},
                                    {'label':'2 Retraités', 'value': 2},
                                    {'label':'3 Retraités', 'value': 3}],
                         value = 1), 
        
        html.H2(id = 'H2', children = 'Paramètres de la pompe à chaleur', style = {'textAlign':'left'}),        
        html.Div(id='text_house_type', children='Type de logement:'),
        dcc.Dropdown( id = 'dropdown_house',
                        options = [ {'label':'4 Façades', 'value': '4'}, 
                                   {'label':'3 Façades', 'value': '3'},
                                   {'label':'2 Façades', 'value': '2'},
                                   {'label':'Appartement', 'value': 'flat'}],
                        value = '4'),
        html.Div(id='text_hp_power', children='Puissance thermique (W):'),
        html.Div(dcc.Input(id='input_hp_power', type='text',value=5000)),
        
        html.H2(id = 'title_boiler', children = 'Paramètres de chauffe-eau électrique', style = {'textAlign':'left'}),        
        html.Div(id='text_volume', children='Volume (l):'),
        html.Div(dcc.Input(id='input_boiler_volume', type='text',value=200)),
        html.Div(id='text_boiler_temperature', children='Temperature nominale (°C):'),
        html.Div(dcc.Input(id='input_boiler_temperature', type='text',value=53)),

        html.H2(id='text2', children='Simulation:'),
        dcc.Dropdown( id = 'dropdown_month',
                     options = [ {'label':calendar.month_name[x], 'value': calendar.month_name[x]} for x in range(1 ,13)]),
        html.Button('Simuler', id='simulate', n_clicks=0),
        html.Div(id='text', children='Entrez un mois de l année et simulez'),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
        dcc.Graph(id = 'plot')
    ])    
    
@app.callback(Output(component_id='plot', component_property= 'figure'),
              Output(component_id='text', component_property= "children"),
              Output("loading-output-1", "children"),
              [Input(component_id='simulate', component_property= 'n_clicks')],
              [State(component_id='checklist_apps', component_property= 'value'),
               State(component_id='dropdown_FTE', component_property= 'value'),
               State(component_id='dropdown_Unemployed', component_property= 'value'),
               State(component_id='dropdown_School', component_property= 'value'),
               State(component_id='dropdown_Retired', component_property= 'value'),
               State(component_id='dropdown_house', component_property= 'value'),
               State(component_id='input_hp_power', component_property= 'value'),
               State(component_id='input_boiler_volume', component_property= 'value'),
               State(component_id='input_boiler_temperature', component_property= 'value'),               
               State(component_id='dropdown_month', component_property= 'value')])
def simulate_button(N,checklist_apps,dropdown_FTE,dropdown_Unemployed,dropdown_School,dropdown_Retired,dropdown_house,input_hp_power,input_boiler_volume,input_boiler_temperature,month):
    '''
    We need as many arguments to the function as there are inputs and states
    Inputs trigger a callback 
    States are used as parameters but do not trigger a callback

    '''
    if month is None:
        todisplay = 'Not showing any data'
        n_month=0
    else:
        todisplay = 'Showing data for ' + month + ' (' + str(N) + ')'
        n_month = list(calendar.month_name).index(month)
    print(todisplay)
    
    # Reading JSON
    with open('inputs/loadshift_inputs.json') as f:
        inputs = json.load(f)

    #update inputs with user-defined values
    inputs['appliances'] = []
    if 'wm' in checklist_apps:
        inputs['appliances'].append("WashingMachine")
    if 'td' in checklist_apps:
        inputs['appliances'].append("TumbleDryer")        
    if 'dw' in checklist_apps:
        inputs['appliances'].append("DishWasher")    
    if 'hp' in checklist_apps:
        inputs['HeatPump'] = True
    else:
        inputs['HeatPump'] = False
    if 'eb' in checklist_apps:
        inputs['ElectricBoiler'] = True
    else:
        inputs['ElectricBoiler'] = False
    if 'ev' in checklist_apps:
        inputs['EV'] = True
    else:
        inputs['EV'] = False
        
    inputs['members'] = []
    inputs['members'] += ['FTE' for i in range(dropdown_FTE)]
    inputs['members'] += ['Retired' for i in range(dropdown_Retired)]
    inputs['members'] += ['Unemployed' for i in range(dropdown_Unemployed)]
    inputs['members'] += ['School' for i in range(dropdown_School)]
    
    inputs['HeatPumpThermalPower'] = int(input_hp_power)
    inputs['Vcyl'] = int(input_boiler_volume)
    inputs['Ttarget'] = int(input_boiler_temperature)
    
    map_building_types = {
        '4': "Detached",
        '3': "Semi-detached",
        '2': "Terraced",
        'flat': "Improved terraced"
        }
    inputs['dwelling_type'] = map_building_types[dropdown_house]
    
    # generating hash for the current config:
    filename = 'cache/' + dict_hash(inputs)
    
    if os.path.isfile(filename):
        #results = pd.read_pickle('data.p')           # temporary file to speed up things
        results = pickle.load(open(filename,'rb'))
        print('Reading previous (cached) simulation data')
    else:
        results = simulate_load(inputs)
        pickle.dump(results,open(filename,"wb"))
    load = results['timeseries']
    # fig = go.Figure([go.Scatter(x = load.index, y = load['{}'.format(value)],\
    #                   line = dict(color = 'firebrick', width = 4))
    #                   ])    

#    fig = px.area(load, load.index)
    idx = load.index[(load.index.month==n_month) & (load.index.year==2016)]
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