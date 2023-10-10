import numpy as np
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from sklearn.metrics import mean_absolute_error
import math
from sklearn.model_selection import train_test_split

import ipywidgets as widgets
from ipywidgets import interact
import folium 
from colour import Color
from collections import defaultdict

from typing import List, Tuple, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

FONT_SIZE_TICKS = 12
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16

pollutants_list = ['NO', 'NO2', 'NOX', 'PM25', 'PM10']

def create_histogram_plot(df: pd.core.frame.DataFrame, bins: int):
    '''Creates an interactive histogram
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        bins (int): number of bins for the histogram.
    
    '''
    def _interactive_histogram_plot(station, pollutant):
        data = df[df.Station==station]
        x = data[pollutant].values 
        try:
            plt.figure(figsize=(12,6))
            plt.xlabel(f'{pollutant} concentration', fontsize=FONT_SIZE_AXES)
            plt.ylabel('Number of measurements', fontsize=FONT_SIZE_AXES)
            plt.hist(x, bins=bins)
            plt.title(f'Pollutant: {pollutant} - Station: {station}', fontsize=FONT_SIZE_TITLE)
            plt.xticks(fontsize=FONT_SIZE_TICKS)
            plt.yticks(fontsize=FONT_SIZE_TICKS)
            plt.show()
        except ValueError:
            print('Histogram cannot be shown for selected values as there is no data')
    
    # Widget for picking the city
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station'
    )

    # Widget for picking the continuous variable
    pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
    )
    
    # Putting it all together
    interact(_interactive_histogram_plot, station=station_selection, pollutant=pollutant_selection);

def create_correlation_matrix(
    raw_data: pd.core.frame.DataFrame,
    pollutants_list: List[str],
):
    """Creates a correlation matrix of the features.

    Args:
        raw_data (pd.core.frame.DataFrame): The data used.
        pollutants_list (List[str]): List of features to include in the plot.
    """
    plt.figure(figsize=(5,5))
    sns.heatmap(
        raw_data[pollutants_list].corr(),
        square=True,
        annot=True,
        cbar=False,
        cmap="RdBu",
        vmin=-1,
        vmax=1
    )
    plt.title('Correlation Matrix of Variables')

    plt.show()
    

 
def create_time_series_plot(df: pd.core.frame.DataFrame, pollutants_list: List[str], start_date: str, end_date: str):
    '''Creates a time series plot, showing the concentration of pollutants over time.
    The pollutant and the measuring station can be selected with a dropdown menu.
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        start_date (str): minimum date for plotting.
        end_date (str): maximum date for plotting.
    
    '''
    def _interactive_time_series_plot(station, pollutant, date_range):
        data = df[df.Station==station]
        data = data[data.DateTime > date_range[0]]
        data = data[data.DateTime < date_range[1]]
        plt.figure(figsize=(12, 6))
        plt.plot(data["DateTime"],data[pollutant], '-')
        plt.title(f'Temporal change of {pollutant}', fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f'{pollutant} concentration', fontsize=FONT_SIZE_AXES)
        plt.xticks(rotation=20, fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()
    
    # Widget for picking the station
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station'
    )

    # Widget for picking the pollutant
    pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
    )

    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(' %d/%m/%Y '), date) for date in dates]
    index = (0, len(options)-1)
    
    # Slider for picking the dates
    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '500px'}
    )

    # Putting it all together
    interact(_interactive_time_series_plot, station=station_selection, pollutant=pollutant_selection,date_range=selection_range_slider);



def add_extra_features(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''Adds new columns to the dataframe by joining it with another dataframe
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
    
    Returns:
        df (pd.core.frame.DataFrame): The updated dataframe with new columns.
    '''
    stations = pd.read_csv('./data/informacion_estaciones_red_calidad_aire.csv', sep=';')
    stations = stations[['CODIGO_CORTO', 'ESTACION', 'LATITUD_ETRS89', 'LONGITUD_ETRS89']]
    stations = stations.rename(columns={'CODIGO_CORTO': 'Station', 'ESTACION': 'Name', 'LATITUD_ETRS89': 'Latitude', 'LONGITUD_ETRS89': 'Longitude'})

    # This cell will convert the values in the columns 'Latitud' and 'Longitud' to 'float64' (decimal) datatype
    stations['Latitude'] = stations['Latitude'].apply(parse_dms)
    stations['Longitude'] = stations['Longitude'].apply(parse_dms)

    df = pd.merge(df, stations, on='Station', how='inner')

    # This cell will extract information from the 'datetime' column and generate months, day or week and hour columns
    #df['day_of_week'] = pd.DatetimeIndex(df['DateTime']).day_name()
    df['day_of_week'] = pd.DatetimeIndex(df['DateTime']).dayofweek
    df['hour_of_day'] = pd.DatetimeIndex(df['DateTime']).hour
    df.loc[df['hour_of_day']==0,'hour_of_day'] = 24
    return df


def parse_dms(coor: str) -> float:
    ''' Transforms strings of degrees, minutes and seconds to a decimal value
    
    Args:
        coor (str): String containing degrees in DMS format
        
    Returns:
        dec_coord (float): coordinates as a decimal value
    '''
    #https://docs.python.org/3/library/pdb.html
    #from IPython.core.debugger import set_trace
    #set_trace()
    
    coor = coor.replace("º", "°" ) #data contains special characters -> º that can´t be interpreted as "string"

    
    parts = re.split('[^\d\w]+', coor)
    
    degrees = float(parts[0])
    minutes = float(parts[1])
    
    if parts[3].isnumeric(): #sometimes, seconds don't have decimal part
        seconds = float(parts[2]+'.'+parts[3])
        direction = parts[4]
    else :
        seconds = float(parts[2]+'.0')
        direction = parts[3]
        
    dec_coord = degrees + minutes / 60 + seconds / (3600)
    
    if direction == 'S' or direction == 'W' or direction == 'O':
        dec_coord *= -1
    
    return dec_coord


def create_map_with_plots(full_data: pd.core.frame.DataFrame, x_variable: str, y_variable: str='PM25') -> folium.Map:
    '''
    Create a map to visualize geo points. The popup will show a scatterplot with the average daily/hourly emisions.
        
    Args:
        full_data (pd.core.frame.DataFrame): The dataframe with the data.
        x_variable (str): The x variable on the popup plot. can be day_of_week or hour_of_day
        y_variable (str): A pollutant to be shown on y axis
    
    '''
    

    
    data = full_data[['Latitude', 'Longitude', y_variable, 'Station', x_variable]]

    data_grouped = data.groupby(['Station', x_variable]).agg(({y_variable: ['mean', 'std']}))
    ymin = data_grouped[y_variable]['mean'].min()
    ymax = data_grouped[y_variable]['mean'].max()

    grouped_means = defaultdict(dict)
    grouped_stds = defaultdict(dict)
    for index, row in data_grouped.iterrows():
        grouped_means[index[0]][index[1]] = row[0]
        grouped_stds[index[0]][index[1]] = row[1]

    for key in grouped_means:
        if (x_variable == 'day_of_week'):
            keys = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            label = 'daily average'
        else:
            keys = list(grouped_means[key].keys())
            label = 'hourly average'

        values = []
        stds = []
        for subkey in keys:
            values.append(grouped_means[key][subkey])
            stds.append(grouped_stds[key][subkey])
        values = np.array(values)
        stds = np.array(stds)
        plt.plot(keys, values, '-o', label=label)
        plt.fill_between(keys, values - stds, values + stds, alpha=0.2)
        if y_variable == 'PM25':
            plt.plot(keys, [12]*len(keys),'--g', label='recommended level')
        plt.plot(keys, [np.average(values)]*len(keys), '--b', label='annual average')
        
        plt.ylim(ymin, ymax)
        plt.title(f'Station {key} avg. {y_variable} / {x_variable.split("_")[0]}')
        plt.ylabel(f'Avg. {y_variable} concentration')
        plt.xlabel(x_variable[0].upper() + x_variable[1:].replace('_', ' '))
        plt.legend(loc="upper left")
        if (x_variable == 'day_of_week'):
            plt.xticks(rotation=20)
        plt.savefig(f'img/tmp/{key}.png')

        plt.clf()
    
    data_grouped_grid = data.groupby('Station').agg(({y_variable: 'mean', 'Latitude': 'min', 'Longitude': 'min'}))
    
    data_grouped_grid_array = np.array(
        [
            data_grouped_grid['Latitude'].values,
            data_grouped_grid['Longitude'].values,
            data_grouped_grid[y_variable].values,
            data_grouped_grid.index.values
        ]
    ).T

          
    map3 = folium.Map(
        location=[data_grouped_grid_array[0][0], data_grouped_grid_array[0][1]],
        tiles='openstreetmap',
        zoom_start=11,
        width=1000,
        height=500
    )
        
    fg = folium.FeatureGroup(name="My Map")
    for lt, ln, pol, station in data_grouped_grid_array:
        fg.add_child(folium.CircleMarker(location=[lt, ln], radius = 15, popup=f"<img src='img/tmp/{int(station)}.png'>",
        fill_color=color_producer(y_variable, pol), color = '', fill_opacity=0.5))
        map3.add_child(fg)
    return map3

       
def color_producer(pollutant_type, pollutant_value):
    ''' This function returns colors based on the pollutant values to create a color representation of air pollution.    
    
    The color scale  for PM2.5 is taken from purpleair.com and it agrees with international guidelines
    The scale for other pollutants was created based on the limits for other pollutants to approximately
    correspond to the PM2.5 color scale. The values in the scale should not be taken for granted and
    are used just for the visualization purposes.
    
    Args:
        pollutant_type (str): Type of pollutant to get the color for
        pollutant_value (float): Value of polutant concentration
        
    Returns:
        pin_color (str): The color of the bucket
    '''
    all_colors_dict = {
        'PM25': {0: 'green', 12: 'yellow', 35: 'orange', 55.4: 'red', 150: 'black'},
        'PM10': {0: 'green', 20: 'yellow', 60: 'orange', 110: 'red', 250: 'black'},
        'CO': {0: 'green', 4: 'yellow', 10: 'orange', 20: 'red', 50: 'black'},
        'O3': {0: 'green', 60: 'yellow', 100: 'orange', 200: 'red', 300: 'black'},
        'NOX': {0: 'green', 40: 'yellow', 80: 'orange', 160: 'red', 300: 'black'},
        'NO': {0: 'green', 40: 'yellow', 80: 'orange', 160: 'red', 300: 'black'},
        'NO2': {0: 'green', 20: 'yellow', 40: 'orange', 80: 'red', 200: 'black'},
    }
    
    # Select the correct color scale, if it is not available, choose PM2.5
    colors_dict = all_colors_dict.get(pollutant_type, all_colors_dict['PM25'])
    thresholds = sorted(list(colors_dict.keys()))
    
    previous = 0
    for threshold in thresholds:
        if pollutant_value < threshold:
            bucket_size = threshold - previous
            bucket = (pollutant_value - previous) / bucket_size
            colors = list(Color(colors_dict[previous]).range_to(Color(colors_dict[threshold]), 11))
            pin_color = str(colors[int(np.round(bucket*10))])
            return pin_color
        previous = threshold


def calculate_mae_for_nearest_station(df: pd.core.frame.DataFrame, target: str) -> Dict[str, float]:
    '''Create a nearest neighbor model and run it on your test data
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution
    '''
    df2 = df.dropna(inplace=False)
    df2.insert(0, 'time_discriminator', (df2['DateTime'].dt.dayofyear * 100000 + df2['DateTime'].dt.hour * 100).values, True)

    train_df, test_df = train_test_split(df2, test_size=0.2, random_state=57)

    imputer = KNNImputer(n_neighbors=1)
    imputer.fit(train_df[['time_discriminator','Latitude', 'Longitude', target]])

    regression_scores = {}

    y_test = test_df[target].values

    test_df2 = test_df.copy()
    test_df2.loc[test_df.index, target] = float("NAN")

    y_pred = imputer.transform(test_df2[['time_discriminator', 'Latitude', 'Longitude', target]])[:,3]
    
    return {"MAE": mean_absolute_error(y_pred, y_test)}    

