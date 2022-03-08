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
from plots import make_demand_plot
from simulation import shift_load,load_config

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

#%% Build the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY], title="Load Shifting")
server = app.server

# Load config
conf = load_config('default')

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
                html.Br(),
                html.Div(id="text_inputs_changed", children=''),
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



#%%  Callbacks relative to the modification of particular input
inputlist = ['dropdown_house','checklist_apps','dropdown_flex_appliances','checklist_hp','yesno_hp','input_hp_power',
             'checklist_dhw','input_boiler_volume','input_boiler_temperature','checklist_ev','checklist_pv','yesno_pv',
             'checklist_bat','input_bat_capacity','input_bat_power']

@app.callback(
    Output("text_inputs_changed", "children"),
    [Input(component_id=state, component_property= 'value') for state in inputlist],
    prevent_initial_call=True
)
def change_config(dropdown_house,checklist_apps,dropdown_flex_appliances,checklist_hp,yesno_hp,input_hp_power,
             checklist_dhw,input_boiler_volume,input_boiler_temperature,checklist_ev,checklist_pv,yesno_pv,
             checklist_bat,input_bat_capacity,input_bat_power):
    global conf
    list_modified = []
    
    if dropdown_house != conf['config']['house']:
        conf['config']['house'] = dropdown_house
        list_modified.append('dropdown_house')
        
    apps = {'td':'TumbleDryer','wm':'WashingMachine','dw':'DishWasher'}
    for key in apps:
        if key in checklist_apps and apps[key] not in conf['config']['columns']:
            conf['config']['columns'].append(apps[key])
            list_modified.append(apps[key])
        elif key not in checklist_apps and apps[key] in conf['config']['columns']:
            conf['config']['columns'].remove(apps[key])
            list_modified.append(apps[key])
        if dropdown_flex_appliances=='shiftable':
            if key in checklist_apps and apps[key] not in conf['config']['TechsShift']:
                conf['config']['TechsShift'].append(apps[key])
                list_modified.append(apps[key])    
            elif key not in checklist_apps and apps[key] in conf['config']['TechsShift']:
                conf['config']['TechsShift'].remove(apps[key])
                list_modified.append(apps[key])
        else:
            if key in conf['config']['TechsShift']:
                conf['config']['TechsShift'].remove(apps[key])
                list_modified.append(apps[key])
            
    if len(list_modified) > 0:
        out = "Modified inputs: " + str(list_modified)
    else:
        out = ''
    
    return out



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
statelist = ['dropdown_house']


@app.callback(
    Output("display1", "figure"),
    Output("display2", "figure"),
    Output("week", "disabled"),
    Output("simulation_output", "children"),
#    Output("coordinates_output", "children"),
    [
        Input('analyze', 'n_clicks'),
        Input("week", "value"),
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
    global n_clicks_last,demand_15min, demand_shifted,pflows,totext,conf
    if n_clicks is None:
        n_clicks = 0
    analyze_button_pressed = n_clicks > n_clicks_last
    n_clicks_last = n_clicks
    
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = ''
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger!='week':

        results,demand_15min,demand_shifted,pflows = shift_load(conf['config'],conf['pvbatt_param'],conf['econ_param'],conf['tariffs'],conf['housetypes'],conf['N'])
        
        print(json.dumps(results, indent=4))
        
        if 'pv' in pflows and not (pflows['pv']==0).all():
            pv = pflows['pv']
        else:
            pv = None
        
    
        n_middle = int(len(demand_15min)/2)
        year = demand_15min.index.isocalendar().year[n_middle]
        idx = demand_15min.index[(demand_15min.index.isocalendar().week==week) & (demand_15min.index.isocalendar().year==year)]
        
        """
        Figures
        """    
        fig = make_demand_plot(idx,demand_15min,PV = pv,title='Consommation sans déplacement de charge')
        fig2 = make_demand_plot(idx,demand_shifted,PV = pv,title='Consommation avec déplacement de charge')
    
        """
        Text output
        """   
        totext = []
        for key in results:
            totext.append(key + ": " + str(results[key]))
            totext.append(html.Br())   
        
        return fig,fig2,False,totext     # Number of returns must be equal to the number of outputs

    else:
        n_middle = int(len(demand_15min)/2)
        idx = demand_15min.index[(demand_15min.index.isocalendar().week==week) & (demand_15min.index.isocalendar().year==demand_15min.index.isocalendar().year[n_middle])]
        if 'pv' in pflows and not (pflows['pv']==0).all():
            pv = pflows['pv']
        else:
            pv = None    
        fig = make_demand_plot(idx,demand_15min,PV = pv,title='Consommation sans déplacement de charge')
        fig2 = make_demand_plot(idx,demand_shifted,PV = pv,title='Consommation avec déplacement de charge')
        return fig,fig2,False,totext
    

# The following function, although it works with some versions of Dash, is not allowed because it adds a callback on the display outputs, which is not allowed.
# @app.callback(
#     Output("display1", "figure"),
#     Output("display2", "figure"),
#     [Input("week", "value")],
#     prevent_initial_call=True
# )
# def update_plot(week):
#     n_middle = int(len(demand_15min)/2)
#     idx = demand_15min.index[(demand_15min.index.isocalendar().week==week) & (demand_15min.index.isocalendar().year==demand_15min.index.isocalendar().year[n_middle])]

#     fig = make_demand_plot(idx,demand_15min,title='Consommation sans déplacement de charge')
#     fig2 = make_demand_plot(idx,demand_shifted,title='Consommation avec déplacement de charge')
    
#     return fig,fig2





#%%

if __name__ == '__main__':
    app.run_server(debug=False)




