from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import json

# load risk factors impact to the life expectancy by country, age, sex
risk_impact = pd.read_csv(
    os.path.join('data','risk_impact.csv')
)

# load calculated life expectancy by country, age, sex
life_expectancy = pd.read_csv(
    os.path.join('data','life_expectancy_calculated.csv')
)

#load risk factors names manageable
risks_names_manageable = pd.read_csv(
    os.path.join('data', 'risks_names_manageable.csv')
)

risks_names_manageable = list(np.concatenate(risks_names_manageable.values))


# load risk_ierarchy and names to id mapping
rei_ierarchy = pd.read_csv(
    os.path.join('data','rei_ierarchy.csv')
)

risk_id_to_parent_id = {int(k):int(v) for k,v in rei_ierarchy[['rei_id','parent_id']].values}

code_book = pd.read_csv(
    os.path.join('data','code_book.csv')
).iloc[1:, 1:]

sex_name_to_id = {
    key: int(value) 
    for key, value in code_book[['sex_label', 'sex_id']].dropna().values[1:]
}

risks_name_to_id = {
    key: int(value) 
    for key, value in code_book[['rei_name', 'rei_id']].dropna().values
}

location_ids = [str(x) for x in risk_impact.location_id.unique()]

location_name_to_id = {
    key: int(value) 
    for key, value in code_book.query(
        'location_id in @location_ids'
    )[['location_name', 'location_id']].dropna().values
}

# load color map
rei_color_map = pd.read_csv(
    os.path.join('data', 'rei_color_map.csv')
)

rei_color_map = {k[1][0]: k[1][1] for k in rei_color_map.iterrows()}

color_mapping = {
    'Default life expectancy': 'rgb(50, 145, 168)',
    'Estimated life extension': 'rgb(49, 212, 117)'
}

def prepare_data(
    location_name: str,
    age: int,
    sex_name: int,
    risk_factors_names: list,
    risk_impact: pd.DataFrame,
    life_expectancy: pd.DataFrame,
    risk_id_to_parent_id: dict,
    location_name_to_id: dict,
    sex_name_to_id: dict,
    risks_name_to_id: dict,
    dietary_risks: str='Groupped',
    round_n_decimals: int=2,
) -> pd.DataFrame:

    location_id = location_name_to_id[location_name]

    risk_factors_id = [risks_name_to_id[x] for x in risk_factors_names]

    sex_id = sex_name_to_id[sex_name]

    risk_impact_filtered = risk_impact.query(
        f'location_id == {location_id}'
        f' and sex_id == {sex_id}'
        ' and rei_id in @risk_factors_id'
    )

    life_expectancy_filtered = life_expectancy.query(
        f'location_id == {location_id}'
        f' and sex_id == {sex_id}'
        f' and age == {age}'
    )[['val', 'upper', 'lower']]

    risk_impact_filtered['rei_parent_id'] = risk_impact_filtered.copy()['rei_id'].map(risk_id_to_parent_id)

    risk_impact_filtered['rei_name'] = (
        risk_impact_filtered.copy()['rei_id']
        .map({k:v for v,k in risks_name_to_id.items()})
    )

    risk_impact_filtered['rei_parent_name'] = (
        risk_impact_filtered.copy()['rei_parent_id']
        .map({k:v for v,k in risks_name_to_id.items()})
    )

    risk_impact_filtered_cur_age = risk_impact_filtered.query(
        f'age == {age}'
    )

    extension_change_by_age = (
        risk_impact_filtered
        .groupby(by=['age'])
        [['val', 'lower', 'upper',]]
        .sum()
        .reset_index()
    )

    if dietary_risks == 'Groupped':
        groupped_dietary_risks = (
            risk_impact_filtered.query('rei_name.str.contains("Diet")')
            .groupby(by=['age', 'rei_parent_name'])
            [['val', 'lower', 'upper']]
            .sum()
            .reset_index()
        )

        groupped_dietary_risks.columns = ['age', 'rei_name', 'val', 'lower', 'upper']

        risk_impact_filtered_dietary_groupped = (
            risk_impact_filtered
            .query('rei_name.str.contains("Diet") == False')
            [['age', 'rei_name', 'val', 'lower', 'upper']]
            .append(groupped_dietary_risks)
        ).sort_values(by=['age', 'val'], ascending=False)

        risk_impact_filtered_dietary_groupped_cur_age = (
            risk_impact_filtered_dietary_groupped
            .query(f'age == {age}')
        )
    else:
        risk_impact_filtered_dietary_groupped = risk_impact_filtered.copy()
        risk_impact_filtered_dietary_groupped_cur_age = risk_impact_filtered_cur_age.copy()

    report = pd.DataFrame(
        {
            'Default life expectancy': life_expectancy_filtered[['val', 'upper', 'lower']].values[0] + age,
            'Estimated life extension': risk_impact_filtered_dietary_groupped_cur_age[['val', 'upper', 'lower',]].sum(),
        },
        index=['val', 'upper', 'lower',]
    )

    report['Extended life expectancy'] = (
        report['Default life expectancy']
        +
        report['Estimated life extension']
    )
    life_expectancy_extension = round(report.loc["val","Estimated life extension"], round_n_decimals)

    suptitle = (
        f' ###### For **{sex_name}**, age: '
        f'**{age}**, '
        f'location: **{location_name}**'
        f' estimated increasing of life expectancy is **{life_expectancy_extension}** years'
    )

    return (
        risk_impact_filtered_cur_age.round(decimals=round_n_decimals),
        risk_impact_filtered,
        life_expectancy_filtered.round(decimals=round_n_decimals),
        round(risk_impact_filtered_cur_age.val.sum(), round_n_decimals),
        extension_change_by_age,
        risk_impact_filtered_dietary_groupped,
        
        risk_impact_filtered_dietary_groupped_cur_age
        .sort_values(by='val', ascending=False)
        .round(decimals=round_n_decimals),

        report.round(decimals=round_n_decimals),
        suptitle
    )


def life_expectancy_treemap_plotter(
    risk_impact_filtered,
    rei_color_map,
) -> go.Figure:
    fig = px.treemap(
        risk_impact_filtered, path=[px.Constant("All selected risks"), 'rei_parent_name', 'rei_name'],
        values='val',
        color='rei_name',
        color_discrete_map=rei_color_map,
        template='plotly_white'
    )

    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

    fig.data[0].textinfo = "label+value+percent root"

    fig.update_layout(height=400, width=300)

    return fig


def life_expectancy_extension_all_ages_plotter(
    extension_change_by_age,
    age,
    risk_impact_filtered_dietary_groupped,
    risk_impact_filtered_dietary_groupped_cur_age,
    rei_color_map,
    width: float=0.7
) -> go.Figure:
    extension_change_by_age = extension_change_by_age.query('age >= @age')
    risk_impact_filtered_dietary_groupped = risk_impact_filtered_dietary_groupped.query('age >= @age')
    risk_impact_filtered_dietary_groupped_cur_age = risk_impact_filtered_dietary_groupped_cur_age.query('age >= @age')

    max_extension_by_age = risk_impact_filtered_dietary_groupped.groupby(by=['age']).val.sum().max()

    fig = make_subplots()

    fig.add_trace(
        go.Scatter(
            x=extension_change_by_age.age.values,
            y=extension_change_by_age.val.values,
            line=dict(color=color_mapping['Estimated life extension']),
            name='Total',
            showlegend=True
        ),
    )
    
    risk_impact_filtered_dietary_groupped

    for i, cur_age in enumerate(risk_impact_filtered_dietary_groupped.age.unique()):

        for rei_name in risk_impact_filtered_dietary_groupped_cur_age.rei_name:
            try:
                extension = risk_impact_filtered_dietary_groupped.query('age == @cur_age and rei_name == @rei_name').val.values[0]
            except:
                extension = 0

            fig.add_trace(
                go.Bar(
                    x=[cur_age],
                    y=[extension],
                    orientation='v',
                    marker_color=rei_color_map[rei_name],
                    width=width,
                    name=rei_name,
                    legendgroup=rei_name,
                    showlegend=True if i == 0 else False
                ),
            )

    fig.update_layout(
        barmode='stack',
        template='plotly_white',
        xaxis1=dict(
            title='Age(years)',           
            tickvals=list(
                range(risk_impact_filtered_dietary_groupped.age.min(), risk_impact_filtered_dietary_groupped.age.max(), 5))
            ),
        yaxis=dict(
            title='Life expectancy extension (years)',
            range=(0, max_extension_by_age + 0.4)
        ),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


def life_expectancy_extension_plotter(
    risk_impact_filtered_dietary_groupped_cur_age,
    report,
    rei_color_map,
    age,
    width: float=0.6
) -> go.Figure:
    default_le = report.loc['val', 'Default life expectancy']
    default_le_lower = report.loc['lower', 'Default life expectancy']
    default_le_upper = report.loc['upper', 'Default life expectancy']

    extended_le = report.loc['val', 'Extended life expectancy']
    extended_le_lower = report.loc['lower', 'Extended life expectancy']
    extended_le_upper = report.loc['upper', 'Extended life expectancy']

    fig = make_subplots(
        y_title='Age',
        subplot_titles=[],
    )

    fig.add_trace(
            go.Bar(
                x=['Extended life expectancy'],
                y=[default_le],
                width=width,
                marker_color=color_mapping['Default life expectancy'],
                name='Default life expectancy',
                showlegend=False,
                legendgroup='Estimated life expectancy without risk factors'
            ),
        )

    for rei_name in risk_impact_filtered_dietary_groupped_cur_age.rei_name.values[:-1]:

        fig.add_trace(
            go.Bar(
                x=['Extended life expectancy'],
                y=[risk_impact_filtered_dietary_groupped_cur_age.query('rei_name == @rei_name').val.values[0]],
                width=width,
                marker_color=rei_color_map[rei_name],
                name=rei_name,
            ),
        )

    rei_name = risk_impact_filtered_dietary_groupped_cur_age.rei_name.values[-1]

    fig.add_trace(
            go.Bar(
                x=['Extended life expectancy'],
                y=[risk_impact_filtered_dietary_groupped_cur_age.query('rei_name == @rei_name').val.values[0]],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[(
                        extended_le_upper
                        - 
                        extended_le
                    )],
                    arrayminus=[(
                        extended_le
                        - 
                        extended_le_lower
                    )]
                ),
                width=width,
                name='Estimated life expectancy excluded risk factors',
                legendgroup='Estimated life expectancy excluded risk factors',
                showlegend=False,
            ),
        )
    fig.add_trace(
            go.Bar(
                x=['Default life expectancy'],
                y=[report.loc['val', 'Default life expectancy']],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[(
                        default_le_upper
                        - 
                        default_le
                    )],
                    arrayminus=[(
                        default_le
                        - 
                        default_le_lower
                    )]
                ),
                width=width,
                marker_color=color_mapping['Default life expectancy'],
                name='Default life expectancy with risk factors',
                showlegend=False,
            ),
        )

    fig.add_annotation(
        y=extended_le,
        x='Extended life expectancy',
        text=(
            f'<b>{round(extended_le, 1)}</b>,<br>'
            f' 95% CI {round(extended_le_lower, 1), round(extended_le_upper, 1)}'
        ),
        showarrow=True,
        arrowwidth=1,
        arrowside='end',
        arrowhead=5,
        ay=-40,
        align='center',
        standoff=20,    
    )

    fig.add_annotation(
        y=default_le,
        x='Default life expectancy',
        text=(
            f'<b>{round(default_le, 1)}</b>,<br>'
            f' 95% CI {round(default_le_lower, 1), round(default_le_upper, 1)}'
        ),
        showarrow=True,
        arrowwidth=1,
        arrowside='end',
        arrowhead=5,
        ay=-40,
        align='center',
        standoff=20,  
    )


    fig.update_layout(
        height=400,
        width=600,
        barmode='stack',
        template='plotly_white',
        yaxis=dict(
            range=(age, int(report.max().max() // 1) + 10),
            tickvals=list(range(age, int(report.max().max() // 1) + 2, 2))
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(title='Exluded risk factors:', traceorder='normal', y=0.9)
    )

    return fig


app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

# create filters
controls_risk = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Excluded risk factors"),
                html.Br(),
                dcc.Checklist(
                    id="risks_names_manageable",
                    options=risks_names_manageable,
                    value=risks_names_manageable
                ),
            ]
        ),
    ],
    body=True,
)

controls_location_sex_age_dietary = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Location"),
                dcc.Dropdown(
                    id="location_name",
                    options=list(location_name_to_id.keys()),
                    value='United States of America',
                    searchable=True,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Sex"),
                dcc.Dropdown(
                    id="sex_name",
                    options=['Male', 'Female'],
                    value='Male',
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Age years"),
                html.Br(),
                dcc.Dropdown(
                    id="age",
                    options=list(range(0, 110, 1)),
                    value=42,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Dietary risks"),
                html.Br(),
                dcc.Dropdown(
                    id="dietary_risks",
                    options=['Groupped', 'Detailed'],
                    value='Groupped',
                    searchable=True,
                ),
            ]
        ),
    ],
    body=True,
)

# setup layout
app.layout = dbc.Container(
    [
        html.Br(),
        html.H2(
            children='Estimation the increase of life expectancy with the exclusion of risk factors'
        ),
        html.Br(),
        dcc.Markdown(
            id='suptitle',
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        controls_location_sex_age_dietary,
                        controls_risk
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="life_expectancy_treemap",
                                                config={
                                                    'displayModeBar': False,                                
                                                },
                                                animate=True,
                                        )
                                    ]
                                ),                                
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="life_expectancy_extension",
                                                config={
                                                    'displayModeBar': False,                                
                                                },
                                                animate=True,
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Br(),
                        html.H6(
                            children='Change of extension life expectancy by age',
                        ),
                        dcc.Graph(
                            id="life_expectancy_extension_all_ages",
                            config={
                                'displayModeBar': False
                            },
                            animate=True,
                        )
                    ],
                    md=9
                )
            ]
        ),
        html.Div([dcc.Store(id='preprocessed_data')])
    ]
)

# setup the callback function for update plots
@app.callback(    
    Output(component_id='suptitle', component_property='children'),
    Output(component_id='preprocessed_data', component_property='data'),
    Input(component_id='location_name', component_property='value'),
    Input(component_id='sex_name', component_property='value'),
    Input(component_id='age', component_property='value'),
    Input(component_id='risks_names_manageable', component_property='value'),
    Input(component_id='dietary_risks', component_property='value'),
)
def update_data(
    location_name,
    sex_name,
    age,
    risks_names_manageable,
    dietary_risks,
):

    (
        risk_impact_filtered,
        risk_impact_filtered_all_ages,
        life_expectancy_filtered,
        total_extension,
        extension_change_by_age,
        risk_impact_filtered_dietary_groupped,
        risk_impact_filtered_dietary_groupped_cur_age,
        report,
        suptitle,
    ) = prepare_data(
        location_name=location_name,
        age=age,
        sex_name=sex_name,
        risk_factors_names=risks_names_manageable,
        risk_impact=risk_impact,
        life_expectancy=life_expectancy,
        risk_id_to_parent_id=risk_id_to_parent_id,
        location_name_to_id=location_name_to_id,
        sex_name_to_id=sex_name_to_id,
        risks_name_to_id=risks_name_to_id,
        round_n_decimals=1,
        dietary_risks=dietary_risks
    )

    preprocessed_data = {
        'risk_impact_filtered': risk_impact_filtered.to_json(orient='split', date_format='iso'),
        'risk_impact_filtered_all_ages': risk_impact_filtered_all_ages.to_json(orient='split', date_format='iso'),
        'life_expectancy_filtered': life_expectancy_filtered.to_json(orient='split', date_format='iso'),
        'extension_change_by_age': extension_change_by_age.to_json(orient='split', date_format='iso'),
        'risk_impact_filtered_dietary_groupped': risk_impact_filtered_dietary_groupped.to_json(orient='split', date_format='iso'),
        'risk_impact_filtered_dietary_groupped_cur_age': risk_impact_filtered_dietary_groupped_cur_age.to_json(orient='split', date_format='iso'),
        'report': report.to_json(orient='split', date_format='iso'),
    }


    return suptitle, json.dumps(preprocessed_data)

@app.callback(  
    Output(component_id='life_expectancy_treemap', component_property='figure'),
    Input(component_id='preprocessed_data', component_property='data'),
)
def update_life_expectancy_treemap(preprocessed_data):
    preprocessed_data = json.loads(preprocessed_data)
    risk_impact_filtered = pd.read_json(
        preprocessed_data['risk_impact_filtered'],
        orient='split'
    )

    life_expectancy_treemap = life_expectancy_treemap_plotter(
        risk_impact_filtered=risk_impact_filtered,
        rei_color_map=rei_color_map
    )

    return life_expectancy_treemap

@app.callback(  
    Output(component_id='life_expectancy_extension', component_property='figure'), 
    Input(component_id='preprocessed_data', component_property='data'),
    Input(component_id='age', component_property='value'),
)
def update_life_expectancy_extension(
    preprocessed_data,
    age,
):
    preprocessed_data = json.loads(preprocessed_data)
    
    risk_impact_filtered_dietary_groupped_cur_age = pd.read_json(
        preprocessed_data['risk_impact_filtered_dietary_groupped_cur_age'],
        orient='split'
    )

    report = pd.read_json(
        preprocessed_data['report'],
        orient='split'
    )

    life_expectancy_extension = life_expectancy_extension_plotter(
        risk_impact_filtered_dietary_groupped_cur_age=risk_impact_filtered_dietary_groupped_cur_age,
        report=report,
        age=age,
        rei_color_map=rei_color_map,
        width=0.7
    )

    return life_expectancy_extension

@app.callback(   
    Output(component_id='life_expectancy_extension_all_ages', component_property='figure'),
    Input(component_id='preprocessed_data', component_property='data'),
    Input(component_id='age', component_property='value'),
)
def update_life_expectancy_extension_all_ages(
    preprocessed_data,
    age,
):
    preprocessed_data = json.loads(preprocessed_data)

    extension_change_by_age = pd.read_json(
        preprocessed_data['extension_change_by_age'],
        orient='split'
    )

    risk_impact_filtered_dietary_groupped = pd.read_json(
        preprocessed_data['risk_impact_filtered_dietary_groupped'],
        orient='split'
    )

    risk_impact_filtered_dietary_groupped_cur_age = pd.read_json(
        preprocessed_data['risk_impact_filtered_dietary_groupped_cur_age'],
        orient='split'
    )

    report = pd.read_json(
        preprocessed_data['report'],
        orient='split'
    )

    life_expectancy_extension_all_ages = life_expectancy_extension_all_ages_plotter(
        extension_change_by_age=extension_change_by_age,
        age=age,
        risk_impact_filtered_dietary_groupped=risk_impact_filtered_dietary_groupped,
        risk_impact_filtered_dietary_groupped_cur_age=risk_impact_filtered_dietary_groupped_cur_age,
        rei_color_map=rei_color_map,
        width=0.7
    )

    return life_expectancy_extension_all_ages

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, host='0.0.0.0', port=8080)
    