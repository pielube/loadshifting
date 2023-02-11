from dash import Dash, dcc, html, Input, Output, State, callback,ctx
import dash_bootstrap_components as dbc
import os

from dash_components import household_components,heating_components,ev_components,pv_components
import defaults
from plots import make_demand_plot,make_pflow_plot
from simulation import shift_load,load_cases
from readinputs import read_config

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


#%% Build the app
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY], title="Load Shifting")
server = app.server

from flask_cors import CORS
CORS(server)

config_path = os.path.join(__location__,'inputs/config.xlsx')

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')])

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
            dbc.Col([dcc.Markdown('Scénario a simuler:') ]),
            dbc.Col([
                dcc.Dropdown( id = 'dropdown_cases',
                             options = [ {'label': 'Non prédéfini', 'value': 'default'} ] + [ {'label':x, 'value': x} for x in load_cases()],
                             value = 'default') 
            ], width=True),
            dbc.Col([
                dbc.Button(
                    "Rafraîchir",
                    id="refresh_cases"
                )
            ], width=True),
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
                    "Stratégie de pilotage",
                    id="coordinates_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            html.Div(id="text_raw_outputs", children='')
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
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id="simulation_output")
                ),
                html.Hr(),

            ], width=3),
            dbc.Col([
                dcc.Graph(id='display1', style={'height': '50vh'}),
                dcc.Graph(id='display2', style={'height': '50vh'}),
                dcc.Graph(id='display3', style={'height': '50vh'}),
                dcc.Markdown(id='titre0',children="### Résultats de simulation"),
                html.Div(id='results',children=''),
                html.Br(),
                dcc.Markdown(id='titre1',children="### Inputs utilisés:"),
                html.Div(id='results2',children='')
            ], width=9, align="start"),
        ]),
        html.Hr(),
        dcc.Markdown("""
        Residential load shifting potential analysis. Powered by [LoadShifting Library](https://github.com/pielube/loadshifting). Build beautiful UIs for your scientific computing apps with [Plot.ly](https://plotly.com/) and [Dash](https://plotly.com/dash/)!
        """),
    ],
    fluid=True
)
                                 
             
                     
#%%  Callbacks for menu expansion on the main page
  
### Callback to refresh the list of scenarios
@callback(
    Output("dropdown_cases", "options"),
    [Input("refresh_cases", "n_clicks")]
)
def refresh_dropdown_cases(n_clicks):
    
    return [ {'label': 'Non prédéfini', 'value': 'default'} ] + [ {'label':x, 'value': x} for x in load_cases()]


                   
### Callback to make household menu expand
@callback(
    Output("household_collapse", "is_open"),
    [Input("household_button", "n_clicks"),
     Input('dropdown_cases','value')],
    [State("household_collapse", "is_open")]
)
def toggle_household_collapse(n_clicks, dropdown_value, is_open):
    if ctx.triggered_id == "household_button":
        if dropdown_value == 'default':
            if n_clicks:
                return not is_open
            return is_open       
        else:
            return False
    else:
        if dropdown_value == 'default':
            return is_open
        else:
            return False
        
### Callback to make heating menu expand
@callback(
    Output("heating_collapse", "is_open"),
    [Input("heating_button", "n_clicks"),
     Input('dropdown_cases','value')],
    [State("heating_collapse", "is_open")]
)
def toggle_heating_collapse(n_clicks, dropdown_value, is_open):
    if ctx.triggered_id == "heating_button":
        if dropdown_value == 'default':
            if n_clicks:
                return not is_open
            return is_open       
        else:
            return False
    else:
        if dropdown_value == 'default':
            return is_open
        else:
            return False
        
### Callback to make ev menu expand
@callback(
    Output("ev_collapse", "is_open"),
    [Input("ev_button", "n_clicks"),
     Input('dropdown_cases','value')],
    [State("ev_collapse", "is_open")]
)
def toggle_ev_collapse(n_clicks, dropdown_value, is_open):
    if ctx.triggered_id == "ev_button":
        if dropdown_value == 'default':
            if n_clicks:
                return not is_open
            return is_open       
        else:
            return False
    else:
        if dropdown_value == 'default':
            return is_open
        else:
            return False

### Callback to make pv menu expand
@callback(
    Output("pv_collapse", "is_open"),
    [Input("pv_button", "n_clicks"),
     Input('dropdown_cases','value')],
    [State("pv_collapse", "is_open")]
)
def toggle_pv_collapse(n_clicks, dropdown_value, is_open):
    if ctx.triggered_id == "pv_button":
        if dropdown_value == 'default':
            if n_clicks:
                return not is_open
            return is_open       
        else:
            return False
    else:
        if dropdown_value == 'default':
            return is_open
        else:
            return False                     

### Callback to make coordinates menu expand
@callback(
    Output("coordinates_collapse", "is_open"),
    [Input("coordinates_button", "n_clicks"),
     Input('dropdown_cases','value')],
    [State("coordinates_collapse", "is_open")]
)
def toggle_coordinates_collapse(n_clicks, dropdown_value, is_open):
    if ctx.triggered_id == "coordinates_button":
        if dropdown_value == 'default':
            if n_clicks:
                return not is_open
            return is_open       
        else:
            return False
    else:
        if dropdown_value == 'default':
            return is_open
        else:
            return False


#%%  Callbacks for activation/deactivation of specific inputs

### Callback to activate the HP inputs
@callback(
    Output("input_hp_power", "disabled"),
    Output('input_hp_power','value'),
    [Input("yesno_hp", "value")]
)
def disable_hp_inputs(yesno_hp):
    if 'auto_hp' in yesno_hp:
        return True,defaults.hp_thermal_power
    return False,defaults.hp_thermal_power

### Callback to activate the HP inputs
@callback(
    Output("input_pv_power", "disabled"),
    Output('input_pv_power','value'),
    [Input("yesno_pv", "value")]
)
def disable_pv_inputs(yesno_pv):
    if 'auto_pv' in yesno_pv:
        return True,""
    return False,defaults.pv_power


#%%
def update_config(conf,dropdown_house,checklist_apps,dropdown_flex_appliances,checklist_hp,yesno_hp,input_hp_power,
             checklist_dhw,input_boiler_volume,input_boiler_temperature,checklist_ev,checklist_pv,yesno_pv,
             input_pv_power,checklist_bat,input_bat_capacity,input_bat_power):

    
    #house type:
    conf['dwelling']['type'] = dropdown_house
    
    #wet appliances:
    apps = {'td':'tumble_dryer','wm':'washing_machine','dw':'dish_washer'}
    for key in apps:
        if key in checklist_apps:
            conf['dwelling'][key] = True
        else:
            conf['dwelling'][key] = False
    
    # type of control for the wet appliances:
    conf['cont']['wetapp'] = dropdown_flex_appliances
        
    # heat pump:
    conf['hp']['yesno'] = 'hp_in' in checklist_hp
    conf['hp']['loadshift'] = 'hp_flex' in checklist_hp
    conf['hp']['automatic_sizing'] = 'auto_hp' in yesno_hp
    if not conf['hp']['automatic_sizing']:
        conf['hp']['pnom'] = float(input_hp_power)

    # DHW:
    conf['dhw']['yesno'] = 'dhw_in' in checklist_dhw
    conf['dhw']['loadshift'] = 'dhw_flex' in checklist_dhw
    conf['dhw']['vol'] = float(input_boiler_volume)
    conf['dhw']['set_point'] = float(input_boiler_temperature)
        
    # EV:
    conf['ev']['yesno'] = 'ev_in' in checklist_ev
    conf['ev']['loadshift'] = 'ev_flex' in checklist_ev      
        
    # PV:
    conf['pv']['yesno'] = 'pv_in' in checklist_pv
    conf['pv']['automatic_sizing'] = 'auto_pv' in yesno_pv
    if not conf['pv']['automatic_sizing']:
        conf['pv']['ppeak'] = float(input_pv_power)
    
    # Battery:
    conf['batt']['yesno'] = 'bat_in' in checklist_bat
    conf['batt']['pnom'] =  float(input_bat_power)
    conf['batt']['capacity'] = float(input_bat_capacity)
    

#%%  Callbacks (misc)
def make_table(conf,config_full):
    '''
    build a dash table from the configuration dataframe with the simulation
    inputs
    '''
    # update the config_full dataframe for the boolean variables:
    for key1 in conf:
        for key2 in conf[key1]:
            varname = key1 + '_' + key2
            if varname in config_full.index:
                if isinstance(conf[key1][key2],bool):
                    if conf[key1][key2]:
                        config_full.loc[varname,'Valeur'] = 'Oui'
                    else:
                        config_full.loc[varname,'Valeur'] = 'Non'
                else:
                    config_full.loc[varname,'Valeur'] = conf[key1][key2]    
    return dbc.Table.from_dataframe(
        config_full,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={}
    )


#%% The callback to the main simulation

# List of states to be considered in the functions:
statelist = ['dropdown_house','checklist_apps','dropdown_flex_appliances','checklist_hp','yesno_hp','input_hp_power',
             'checklist_dhw','input_boiler_volume','input_boiler_temperature','checklist_ev','checklist_pv','yesno_pv',
             'input_pv_power','checklist_bat','input_bat_capacity','input_bat_power']


@callback(
    Output("display1", "figure"),
    Output("display2", "figure"),
    Output("display3", "figure"),
    Output("week", "disabled"),
    Output("simulation_output", "children"),
    Output("results", "children"),
    Output("results2", "children"),
#    Output("coordinates_output", "children"),
    [
        Input('analyze', 'n_clicks'),
        Input("week", "value"),
#        Input("height_slider_input", "value"),
#        Input("streamline_density_slider_input", "value"),
    ],
    [State(component_id=state, component_property= 'value') for state in statelist],
)
def display_graph(n_clicks,week,
                  dropdown_house,checklist_apps,dropdown_flex_appliances,checklist_hp,yesno_hp,input_hp_power,
                  checklist_dhw,input_boiler_volume,input_boiler_temperature,checklist_ev,checklist_pv,yesno_pv,
                  input_pv_power,checklist_bat,input_bat_capacity,input_bat_power):
    '''
    We need as many arguments to the function as there are inputs and states
    Inputs trigger a callback 
    States are used as parameters but do not trigger a callback
    '''
        
    conf,prices,config_full = read_config(config_path)
    update_config(conf,dropdown_house,checklist_apps,dropdown_flex_appliances,checklist_hp,yesno_hp,input_hp_power,
                  checklist_dhw,input_boiler_volume,input_boiler_temperature,checklist_ev,checklist_pv,yesno_pv,
                  input_pv_power,checklist_bat,input_bat_capacity,input_bat_power)
    
    # make a table with all simulation inputs:
    inputs_table = make_table(conf,config_full)

    results,demand_15min,demand_shifted,pflows,input_data = shift_load(conf,prices)
    
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
    fig3 = make_pflow_plot(idx,pflows)

    """
    Text output
    """   
    # first check that all results are numerical
    results.fillna(-99999,inplace=True)
    
    totext = []
    totext.append('### Résultats de simulation')
    totext.append('#### Installation')
    totext.append("Système PV: {:.2f} kWc".format(results['Valeur']['PVCapacity']))
    totext.append("Demande maximale: {:.2f} kW".format(results['Valeur']['peakdem']))
    totext.append("Consommation totale: {:.0f} kWh".format(results['Valeur']['annual_load']))
    totext.append("Electricité produite: {:.0f} kWh".format(results['Valeur']['el_prod']))
    totext.append("Electricité autoconsommée: {:.0f} kWh".format(results['Valeur']['el_selfcons']))
    totext.append("Electricité achetée au réseau: {:.0f} kWh".format(results['Valeur']['el_boughtfromgrid']))
    totext.append("SSR: {:.1f} %".format(results['Valeur']['selfsuffrate']*100))
    totext.append("SCR: {:.1f} %".format(results['Valeur']['selfconsrate']*100))
    totext.append("Quantité de charge déplacée: {:.0f} kWh".format(results['Valeur']['el_shifted']))
    totext.append('#### Paramètres économiques')
    totext.append("LCOE: {:.0f} €/MWh".format(results['Valeur']['LCOE']))
    totext.append("Temps de retour sur investissement: {:.1f} ans".format(results['Valeur']['pbp_all']))
    totext.append("**Facture d'électricité: {:.2f} EUR/an**".format(results['Valeur']['el_bill']))
    totext.append("Bénéfices de la revente au réseau: {:.0f} €".format(results['Valeur']['el_stg']))
    
    # for key in results:
    #     totext.append(key + ": " + str(results[key])) 
        
    maintext = "Facture d'électricité: {:.2f} EUR/an".format(results['Valeur']['el_bill'])
    if defaults.verbose > 0:
        for txt in totext:
            print(txt)
    
    results_table = dbc.Table.from_dataframe(
        results,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={} )

    return fig,fig2,fig3,False,maintext,results_table,inputs_table    # Number of returns must be equal to the number of outputs
    



#%%

if __name__ == '__main__':
    app.run_server(debug=True)




