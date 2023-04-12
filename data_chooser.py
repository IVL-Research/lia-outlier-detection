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
    
    def activate_plot(self):
        """
        Display interactive plot where images (data points in the plot)
        can be selected using box select or lasso select. 
        """
        # TODO: cmin and cmax depending on chosen_color_column (manual_outlier will always be -1 to 1)
        self.df_copy.reset_index(inplace=True,drop=True)
        numeric_df = self.df_copy.select_dtypes(include=np.number)
        numeric_columns = numeric_df.columns
        self.f = go.FigureWidget([go.Scatter(y = self.df_copy[self.columns[0]], x = self.df_copy[self.columns[1]], mode = 'markers',
                                       selected_marker_color = "red", 
                                             marker=dict(color=numeric_df[numeric_columns[0]],
                                                        colorbar=dict(thickness=10), colorscale=["blue", "green", "orange"]))])
        
        scatter = self.f.data[0]
        scatter.marker.opacity = 0.5
        
        self.axis_dropdowns = interactive(self.update_axes, yaxis = self.columns, xaxis = self.columns, color = numeric_columns)
        scatter.on_selection(self.selection_fn)
        
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

    def selection_fn(self,trace,points,selector):
        """
        Keeping track of points manually selected and change values in column ["manual_outlier"].
        Value for points not manually selected is -1. If selected to be an outlier, value is set to 1.
        If selected again not to be an outlier, value is set to 0. Previous value is stored for future 
        possibility to undo selection. 

        Each selection is stored in a temp_df and all temp_df's are stored in self.outlier_df.
        The dataframe drop_duplicates_df is the df which will be used to train the model, where only 
        the last manually made change to a data point is included. 

        The plot is updated after selection.
        """
        temp_df = self.df_copy.loc[points.point_inds]
        
        # If there will be an undo button, we need to keep track of number of points selected each time
        last_selected = len(temp_df)
        
        for i in temp_df.iterrows():
            idx = i[0]
            temp_df.at[idx, "last_selected"] = last_selected
            # This is needed for keeping track of the changes
            temp_df.at[idx, "manual_outlier"] = 1 if self.df_copy.at[idx, "manual_outlier"] != 1 else 0
            # This is needed for displaying values in the plot
            self.df_copy.at[idx, "manual_outlier"] = 1 if self.df_copy.at[idx, "manual_outlier"] != 1 else 0

        self.outlier_df = pd.concat([self.outlier_df, temp_df], ignore_index=False, axis=0)

        no_points = "point" if last_selected == 1 else "points"
        print(f"Selected {last_selected} new {no_points}. Total: {len(self.outlier_df)}")

        drop_duplicates_df = self.outlier_df.drop_duplicates(subset=["x", "y1"], keep="last")
        drop_duplicates_df.sort_values(by=["x"], inplace=True)

        print(f"Unique points selected ({len(drop_duplicates_df)}):")
        for i in drop_duplicates_df.iterrows():
            outlier = "yes" if i[1][3] == 1 else "no"
            print(f"x: {int(i[1][0])}, y1: {int(i[1][1])}, outlier: {outlier}")

        # Update plot with chosen column
        chosen_color_column = self.axis_dropdowns.children[2].value
        trace.update(marker_color=self.df_copy[chosen_color_column])

    def clear_selection(self):
        self.outlier_df = self.outlier_df.iloc[0:0]
    
    def show_selected(self):
        for index, row in self.outlier_df.iterrows():
            plt.figure()
            plt.imshow(plt.imread(row['file']))
            plt.title(f"{row['time']}, wl: {row['wl']}, turb_s: {row['turb_sensor']}, turb_p: {row['turb_post']}")

    # create train model function based on outlier status in self.df

    # visualize result in graph

    # function to mark point as non-outlier DONE

    # button to undo choice

    # button to confirm (then train model), disable if not choosen areas == 1

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

