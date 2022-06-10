import dash_core_components as dcc
import dash_html_components as html

import defaults

# Household characteristics:
household_components = [
    html.Div(id='text_house_type', children='Type de logement:'),
    dcc.Dropdown( id = 'dropdown_house',
                    options = [ {'label':'4 Façades', 'value': '4f'}, 
                               {'label':'3 Façades', 'value': '3f'},
                               {'label':'2 Façades', 'value': '2f'},
                               {'label':'Appartement', 'value': '1f'}],
                    value = '4f'),

    html.Hr(),
    html.Div(id='text_appliances', children='Equipements électrodomestiques'),
    dcc.Checklist(
                    id = 'checklist_apps',
                    options=[
                        {'label': 'Lave-linge', 'value': 'wm'},
                        {'label': 'Sèche-linge', 'value': 'td'},
                        {'label': 'Lave-vaisselle', 'value': 'dw'}
                    ],
                    value=['wm', 'td','dw'],
                    labelStyle={'display': 'block'}
                ),

    html.Div(id='text_flex_appliances', children='Participation des électrodomestiques'),
    dcc.Dropdown( id = 'dropdown_flex_appliances',
                    options = [ {'label':'Charge déplaçable', 'value': 'shiftable'}, 
                               {'label':'Charge non déplaçable', 'value': 'non-shiftable'}],
                    value = 'shiftable'),


]


# Heating characteristics:
heating_components = [
    html.H4(id = 'title_hp', children = 'Pompe à chaleur', style = {'textAlign':'left'}),  
    dcc.Checklist(
                    id = 'checklist_hp',
                    options=[
                        {'label': ' Présence dans le ménage', 'value': 'hp_in'},
                        {'label': ' Charge déplaçable', 'value': 'hp_flex'},
                    ],
                    value=['hp_in', 'hp_flex'],
                    labelStyle={'display': 'block'}
                ),      
    html.Div(id='text_hp_power', children='Puissance thermique (W):'),
    dcc.Checklist(
                    id = 'yesno_hp',
                    options=[
                        {'label': ' Dimensionnement automatique', 'value': 'auto_hp'},
                    ],
                    value=['auto_hp'],
                    labelStyle={'display': 'block'}
                ),     
    html.Div(dcc.Input(id='input_hp_power', type='text',value=defaults.hp_thermal_power,disabled=True)),
    html.Hr(),
    
    html.H4(id = 'title_dhw', children = 'Chauffe-eau électrique', style = {'textAlign':'left'}),     
    dcc.Checklist(
                    id = 'checklist_dhw',
                    options=[
                        {'label': ' Présence dans le ménage', 'value': 'dhw_in'},
                        {'label': ' Charge déplaçable', 'value': 'dhw_flex'},
                    ],
                    value=['dhw_in', 'dhw_flex'],
                    labelStyle={'display': 'block'}
                ),   
    html.Div(id='text_volume', children='Volume (l):'),
    html.Div(dcc.Input(id='input_boiler_volume', type='text',value=defaults.Vol_DHW)),
    html.Div(id='text_boiler_temperature', children='Temperature nominale (°C):'),
    html.Div(dcc.Input(id='input_boiler_temperature', type='text',value=defaults.T_sp_DHW)),

]


### Electric vehicles
ev_components = [
    dcc.Checklist(
        id = 'checklist_ev',
        options=[
            {'label': " Présence d'un véhicule électrique", 'value': 'ev_in'},
            {'label': ' Charge déplaçable', 'value': 'ev_flex'},
        ],
        value=['ev_in', 'ev_flex'],
        labelStyle={'display': 'block'}
    ),      

]



# PV and battery characteristics:
pv_components = [
    html.H4(id = 'title_pv', children = 'Photovoltaïque', style = {'textAlign':'left'}),  
    dcc.Checklist(
                    id = 'checklist_pv',
                    options=[
                        {'label': ' Installation PV', 'value': 'pv_in'},
                    ],
                    value=['pv_in'],
                    labelStyle={'display': 'block'}
                ),      
    html.Div(id='text_pv_power', children='Puissance crète (kWp):'),
    dcc.Checklist(
                    id = 'yesno_pv',
                    options=[
                        {'label': ' Dimensionnement automatique', 'value': 'auto_pv'},
                    ],
                    value=['auto_pv'],
                    labelStyle={'display': 'block'}
                ),     
    html.Div(dcc.Input(id='input_pv_power', type='text',value=defaults.pv_power,disabled=True)),
    html.Hr(),
    
    html.H4(id = 'title_bat', children = 'Batterie', style = {'textAlign':'left'}),     
    dcc.Checklist(
                    id = 'checklist_bat',
                    options=[
                        {'label': " Présence d'une batterie", 'value': 'bat_in'},
                    ],
                    value=['bat_in'],
                    labelStyle={'display': 'block'}
                ),   
    html.Div(id='text_bat_cap', children='Capacité (kWh):'),
    html.Div(dcc.Input(id='input_bat_capacity', type='text',value=defaults.bat_cap)),
    html.Div(id='text_bat_power', children='Puissance max (W):'),
    html.Div(dcc.Input(id='input_bat_power', type='text',value=defaults.bat_power)),

]



