#!/usr/bin/env python
# coding: utf-8

import sys
import os
import matplotlib.pyplot as plt
import glob

from datetime import datetime
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from ipywidgets import interactive, HBox, VBox
# import keras
import pandas as pd
import numpy as np
import random
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from functools import partial

class interactive_data_chooser:
    """
    Class for selecting data graphically and displaying it
    """
    def __init__(self, df, columns):
        # we don't need this dataframe, make a df_copy instead?
        self.outlier_df = pd.DataFrame()

        # self.df = df
        self.df_copy = df.copy()
        self.columns = columns
        self.df_copy["manual_outlier"] = -1
        self.df_copy["model_outlier"] = 0

        self.axis_dropdowns = None
        self.chosen_color_column = self.df_copy["manual_outlier"]
        self.trace1_color = None
        self.trace2_color = None
    
    def activate_plot(self):
        self.df_copy.reset_index(inplace=True,drop=True)
        numeric_df = self.df_copy.select_dtypes(include=np.number)
        numeric_columns = numeric_df.columns

        # Create the scatter plot with markers and lines for z < 1
        trace1 = go.Scatter(x=self.df_copy.loc[self.chosen_color_column < 1, 'x'], 
                                    y=self.df_copy.loc[self.chosen_color_column < 1, 'y1'],
                                    mode='markers+lines', 
                                    selected_marker_color = "orange",
                                    visible=True,
                                    opacity=1.0,
                                    marker=dict(size=10, 
                                                colorscale=["blue", "green"], 
                                                color=self.trace1_color), # color=numeric_df[numeric_columns[0]]),
                                                showlegend=True,
                                                name="non-outlier")

        # Add a second scatter trace with markers only for z = 1
        trace2 = go.Scatter(x=self.df_copy.loc[self.chosen_color_column == 1, 'x'], 
                                    y=self.df_copy.loc[self.chosen_color_column == 1, 'y1'],
                                    mode='markers', 
                                    selected_marker_color = "orange",
                                    visible=True,
                                    opacity=1.0,
                                    marker=dict(size=10, 
                                                colorscale=["blue", "green", "red"], 
                                                color=self.trace2_color), #  numeric_df[numeric_columns[0]]),
                                                marker_symbol="x", 
                                                showlegend=True,
                                                name="outlier")
        
        trace1.hovertemplate = '<b>Trace 1</b><br>X: %{x}<br>Y: %{y}'
        trace2.hovertemplate = '<b>Trace 2</b><br>X: %{x}<br>Y: %{y}'
        
        self.f = go.FigureWidget(data=[trace1, trace2])

        # Customized legend
        self.f.add_trace(go.Scatter(y=[None], mode='markers',
                         marker=dict(symbol='circle', color='blue', size=10),
                         name='Not manually chosen'
                         ))
        self.f.add_trace(go.Scatter(y=[None], mode='markers',
                         marker=dict(symbol='triangle-up', color='green', size=10),
                         name='Not outlier',
                         ))
        self.f.add_trace(go.Scatter(y=[None], mode='markers',
                         marker=dict(symbol='x', color='red', size=10),
                         name='Outlier',
                         ))
        self.f.data[0].showlegend = False
        self.f.data[1].showlegend = False
        
        self.axis_dropdowns = interactive(self.update_axes, yaxis = self.columns, xaxis = self.columns, color = numeric_columns)
        
        self.f.data[0].on_selection(self.selection_fn)
        self.f.data[1].on_selection(self.selection_fn)
        
        # Put everything together
        return VBox((HBox(self.axis_dropdowns.children),self.f))
    
    def update_axes(self, xaxis, yaxis,color):
        scatter = self.f.data[0]
        scatter.x = self.df_copy[xaxis]
        scatter.y = self.df_copy[yaxis]
        scatter.marker.color = self.df_copy[color]
        with self.f.batch_update():
            self.f.layout.xaxis.title = xaxis
            self.f.layout.yaxis.title = yaxis
   
    def update_manual_outlier(self, row):
        row["manual_outlier"] = 1 if self.df_copy[row[0]]["manual_outlier"] != 1 else 0
        return row
    
    # def multiply_rows(row): Use this solution instead of iterrows
        # return row['column1'] * row['column2']

        # my_df['multiplied'] = my_df.apply(multiply_rows,axis=1)

    def update_temp_df_last_sel(self, row, last_selected):
        row["last_selected"] = last_selected
        return row        

    def remove_selected_data_points(self, current_list_x, current_list_y, points):
        current_list_x = np.delete(current_list_x, points.point_inds)
        current_list_y = np.delete(current_list_y, points.point_inds)
        return current_list_x, current_list_y

    def get_x_and_y_values_current_trace(self, trace):
        trace_value = 0 if trace.name == "non-outlier" else 1
        x_values = np.array(self.f.data[trace_value].x)
        y_values = np.array(self.f.data[trace_value].y)
        return x_values, y_values
    
    def get_x_and_y_values_other_trace(self, trace):
        trace_value = 0 if trace.name == "outlier" else 1
        x_values = np.array(self.f.data[trace_value].x)
        y_values = np.array(self.f.data[trace_value].y)
        return x_values, y_values
    
    def append_selected_data_points(self, current_list_x, current_list_y, points):
        appended_list_x = np.append(current_list_x, points.xs)
        appended_list_y = np.append(current_list_y, points.ys)
        return appended_list_x, appended_list_y

    def selection_fn(self,trace,points,selector):
        
        # Store the selected data points in temp_df
        temp_df = self.df_copy[self.df_copy["x"].isin(points.point_inds)]
        self.chosen_color_column = self.axis_dropdowns.children[2].value 
        
        """ temp_df["last_selected"] = temp_df.apply(lambda row: self.update_temp_df_last_sel(row, last_selected), axis=1) """
        # Get the selected points based on x values
        # TODO: Should I change to index instead?
        # TODO: Skip temp_df and change df_copy to df
        # TODO: Ändra namn på trace1 och trace2 samt trace1_color
        for x_value in points.xs:  
            """ temp_df.at[idx, "last_selected"] = last_selected """
            # This is needed for keeping track of the changes
            temp_df.at[x_value, "manual_outlier"] = 1 if self.df_copy.at[x_value, "manual_outlier"] != 1 else 0
            # This is needed for displaying values in the plot
            self.df_copy.at[x_value, "manual_outlier"] = 1 if (self.df_copy.at[x_value, "manual_outlier"] != 1) else 0

        
        # List only values in manual outlier for trace1 to get a correct plot
        self.trace1_color = [x for x in self.df_copy["manual_outlier"] if x != 1]

        # Add selected data points to the other trace and update it
        other_trace_x, other_trace_y = self.get_x_and_y_values_other_trace(trace)
        other_trace_x, other_trace_y = self.append_selected_data_points(other_trace_x, other_trace_y, points)
        other_trace_name = "outlier" if trace.name == "non-outlier" else "non-outlier"

        # If data points in "outlier" have been added to "non-outlier"-trace, then sort on x axis
        if trace.name == "outlier":
            sort_indices = np.argsort(other_trace_x)
            other_trace_x = other_trace_x[sort_indices]
            other_trace_y = other_trace_y[sort_indices]

        self.f.update_traces(x=other_trace_x, y=other_trace_y, selector=dict(name=other_trace_name))

        # Remove selected data points from current trace and update it
        trace_x, trace_y = self.get_x_and_y_values_current_trace(trace)
        trace_x, trace_y = self.remove_selected_data_points(trace_x, trace_y, points)
        self.f.update_traces(x=trace_x, y=trace_y, selector=dict(name=trace.name))
        
        # Update marker symbol in trace1
        symbols = {-1: "circle", 0: "triangle-up"}
        marker_symbols = [symbols[i] for i in self.trace1_color]
        self.f.update_traces(marker_color=self.trace1_color, marker_symbol=marker_symbols, selector=dict(name="non-outlier")) 

def create_fake_df(n):
    """
    Creates a dataframe with n rows and columns "x", "y1" and "y2". 
    The data are integers, 0-100.
    """
    x = []
    y1 = []
    y2 = []

    for _ in range(n):    
        x_int = random.randint(0, 100)
        x.append(x_int)
        y1_int = random.randint(0, 100)
        y1.append(y1_int)
        y2_int = random.randint(0, 100)
        y2.append(y2_int)

    int_dict = {"x": x, "y1": y1, "y2": y2}
    df = pd.DataFrame(int_dict)
    return df

