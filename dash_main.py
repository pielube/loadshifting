import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.figure_factory as ff
import pandas as pd
import os,json

from dash_components import household_components,heating_components,ev_components,pv_components
import defaults
from demands import compute_demand

#%% Build the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY], title="Load Shifting")
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dcc.Markdown("""
                # Analyse de déplacements de charge au niveau résidentiel
                
                Outil de simulation développé dans le cadre du marché public Région Wallone ''estimation
                individuelle du taux d’autoconsommation'' (cahier spécial des charges N° O4.04.03 -20-3933)
                       
                Par [Sylvain Quoilin](http://www.squoilin.eu/) et Pietro Lubello (Université de Liège).
                """)
            ], width=True),
            dbc.Col([
                html.Img(src="assets/uliege_logo_cmjn_mono.svg", alt="Uliege Logo", height="50px"),
            ], width=1)
        ], align="end"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                
                dbc.Button(
                    "Charactéristiques du ménage",
                    id="household_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            household_components,
                        )
                    ),
                    id="household_collapse",
                    is_open=False
                ),
                html.Hr(),
                                
                dbc.Button(
                    "Chauffage et ECS",
                    id="heating_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            heating_components,
                        )
                    ),
                    id="heating_collapse",
                    is_open=False
                ),
                html.Hr(),                
                
                dbc.Button(
                    "Véhicule électrique",
                    id="ev_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            ev_components,
                        )
                    ),
                    id="ev_collapse",
                    is_open=False
                ),
                html.Hr(),
                
                dbc.Button(
                    "Installation PV et batterie",
                    id="pv_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            pv_components,
                        )
                    ),
                    id="pv_collapse",
                    is_open=False
                ),
                html.Hr(),
                
                dbc.Button(
                    "Show Raw Coordinates (*.dat format)",
                    id="coordinates_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Markdown(id="coordinates_output")
                        )
                    ),
                    id="coordinates_collapse",
                    is_open=False
                ),
                html.Hr(),
                dbc.Button(
                    "Simulation",
                    id="analyze", color="primary", style={"margin": "5px"}),
                html.Div(id="text_week", children='Afficher la semaine n°:'),
                dcc.Slider(
                    id="week",
                    min=1,
                    max=52,
                    step=1,
                    value=1,
                    marks={
                            1: '1',
                            10: '10',
                            20: '20',
                            30: '30',
                            40: '40',
                            52: '52'},
                    tooltip={"placement": "top", "always_visible": False},
                    disabled = True
                ),
                html.Div(id="text_week2", children=''),
                html.Br(),
                html.Hr(),
                dcc.Markdown("##### Résultats de simulation"),
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id="simulation_output")
                ),

            ], width=3),
            dbc.Col([
                dcc.Graph(id='display1', style={'height': '50vh'}),
                dcc.Graph(id='display2', style={'height': '50vh'}),
            ], width=9, align="start")
        ]),
        html.Hr(),
        dcc.Markdown("""
        Residential load shifting potential analysis. Powered by [LoadShifting Library](https://github.com/pielube/loadshifting). Build beautiful UIs for your scientific computing apps with [Plot.ly](https://plotly.com/) and [Dash](https://plotly.com/dash/)!
        """),
    ],
    fluid=True
)

#%%  Callbacks for menu expansion
                     
### Callback to make household menu expand
@app.callback(
    Output("household_collapse", "is_open"),
    [Input("household_button", "n_clicks")],
    [State("household_collapse", "is_open")]
)
def toggle_household_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open                     
                     
 
### Callback to make household menu expand
@app.callback(
    Output("heating_collapse", "is_open"),
    [Input("heating_button", "n_clicks")],
    [State("heating_collapse", "is_open")]
)
def toggle_heating_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open    
               


### Callback to make EV parameters menu expand
@app.callback(
    Output("ev_collapse", "is_open"),
    [Input("ev_button", "n_clicks")],
    [State("ev_collapse", "is_open")]
)
def toggle_ev_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make EV parameters menu expand
@app.callback(
    Output("pv_collapse", "is_open"),
    [Input("pv_button", "n_clicks")],
    [State("pv_collapse", "is_open")]
)
def toggle_pv_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open



### Callback to make coordinates menu expand
@app.callback(
    Output("coordinates_collapse", "is_open"),
    [Input("coordinates_button", "n_clicks")],
    [State("coordinates_collapse", "is_open")]
)
def toggle_coordinates_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


#%%  Callbacks for activation/deactivation of specific inputs

### Callback to activate the HP inputs
@app.callback(
    Output("input_hp_power", "disabled"),
    Output('input_hp_power','value'),
    [Input("yesno_hp", "value")]
)
def disable_hp_inputs(yesno_hp):
    if 'auto_hp' in yesno_hp:
        return True,""
    return False,defaults.hp_thermal_power

### Callback to activate the HP inputs
@app.callback(
    Output("input_pv_power", "disabled"),
    Output('input_pv_power','value'),
    [Input("yesno_pv", "value")]
)
def disable_pv_inputs(yesno_pv):
    if 'auto_pv' in yesno_pv:
        return True,""
    return False,defaults.pv_power



#%%  Callbacks (misc)



def make_table(dataframe):
    return dbc.Table.from_dataframe(
        dataframe,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={

        }
    )


last_analyze_timestamp = None
n_clicks_last = 0

#%% The callback to the main simulation

# List of states to be considered in the functions:
statelist = ['week','dropdown_house']


@app.callback(
    Output("display1", "figure"),
    Output("week", "disabled"),
    Output("simulation_output", "children"),
#    Output("coordinates_output", "children"),
    [
        Input('analyze', 'n_clicks'),
#        Input("height_slider_input", "value"),
#        Input("streamline_density_slider_input", "value"),
    ],
    [State(component_id=state, component_property= 'value') for state in statelist],
)
def display_graph(n_clicks,week,dropdown_house):
    '''
    We need as many arguments to the function as there are inputs and states
    Inputs trigger a callback 
    States are used as parameters but do not trigger a callback
    '''
    global n_clicks_last
    if n_clicks is None:
        n_clicks = 0
    analyze_button_pressed = n_clicks > n_clicks_last
    n_clicks_last = n_clicks
    
    path = r'./inputs'
        
    """
    Loading inputs
    """
    filename = dropdown_house +'.json'
    file = os.path.join(path,filename)
    with open(file) as f:
      inputs = json.load(f)
    N = 1
      
    demands = compute_demand(inputs,N)
    global load
    load = demands['results'][0]

    # #update inputs with user-defined values
    # inputs['appliances']['apps'] = []
    # if 'wm' in checklist_apps:
    #     inputs['appliances']['apps'].append("WashingMachine")
    # if 'td' in checklist_apps:
    #     inputs['appliances']['apps'].append("TumbleDryer")        
    # if 'dw' in checklist_apps:
    #     inputs['appliances']['apps'].append("DishWasher")    
    # if 'hp' in checklist_apps:
    #     inputs['HP']['loadshift'] = True
    # else:
    #     inputs['HP']['loadshift'] = False
    # if 'eb' in checklist_apps:
    #     inputs['DHW']['loadshift'] = True
    # else:
    #     inputs['DHW']['loadshift'] = False
    # if 'ev' in checklist_apps:
    #     inputs['EV']['loadshift'] = True
    # else:
    #     inputs['EV']['loadshift'] = False
        
    # inputs['members'] = []
    # inputs['members'] += ['FTE' for i in range(dropdown_FTE)]
    # inputs['members'] += ['Retired' for i in range(dropdown_Retired)]
    # inputs['members'] += ['Unemployed' for i in range(dropdown_Unemployed)]
    # inputs['members'] += ['School' for i in range(dropdown_School)]
    
    # # If given, HP size imposed, otherwise automatic sizing
    # inputs['HP']['HeatPumpThermalPower'] = int(input_hp_power)
    
    # inputs['DHW']['Vcyl'] = int(input_boiler_volume)
    # inputs['DHW']['Ttarget'] = int(input_boiler_temperature)
    

    n_middle = int(len(load)/2)
    year = load.index.isocalendar().year[n_middle]
    idx = load.index[(load.index.isocalendar().week==week) & (load.index.isocalendar().year==year)]
    fig = go.Figure()
    for key in load:
        fig.add_trace(go.Scatter(
            name = key,
            x = idx,
            y = load.loc[idx,key],
            stackgroup='one',
            mode='none'               # this remove the lines
           ))

    fig.update_layout(title = 'Consommation',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Puissance en W'
                      )
    totext = []
    for x in ['coucou','tout le monde']:
        totext.append(x)
        totext.append(html.Br())   
    
    return fig,False,totext     # Number of returns must be equal to the number of outputs


if __name__ == '__main__':
    app.run_server(debug=False)



### Update the plot with a different week. TODO: understand why this does not work
@app.callback(
    Output("display1", "figure"),
    [Input("week", "value")]
)
def update_plot(week):
    n_middle = int(len(load)/2)
    idx = load.index[(load.index.isocalendar().week==week) & (load.index.isocalendar().year==load.index.isocalendar().year[n_middle])]
    fig = go.Figure()
    for key in load:
        fig.add_trace(go.Scatter(
            name = key,
            x = idx,
            y = load.loc[idx,key],
            stackgroup='one',
            mode='none'               # this remove the lines
           ))

    fig.update_layout(title = 'Consommation',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Puissance en W'
                      )
    return fig


