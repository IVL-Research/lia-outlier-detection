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
        self.outlier_df = pd.DataFrame(df, columns)

        self.df = df
        self.columns = columns
        self.df["manual_outlier"] = -1
        self.df["model_outlier"] = 0
    
    def activate_plot(self):
        """
        Display interactive plot where images (data points in the plot)
        can be selected using box select or lasso select. 
        Selected values are stored in the global dataframe outlier_df
        """
        self.df.reset_index(inplace=True,drop=True)
        numeric_df = self.df.select_dtypes(include=np.number)
        numeric_columns = numeric_df.columns
        self.f = go.FigureWidget([go.Scatter(y = self.df[self.columns[0]], x = self.df[self.columns[1]], mode = 'markers',
                                       selected_marker_color = "red", 
                                             marker=dict(color=numeric_df[numeric_columns[0]],
                                                        colorbar=dict(thickness=10)))])
        # Somewhere in this method: if manual_outlier not "NaN" and manual_outlier not model_outlier:
        # print manual_outlier
        
        scatter = self.f.data[0]
        
        scatter.marker.opacity = 0.5
        
        axis_dropdowns = interactive(self.update_axes, yaxis = self.columns, xaxis = self.columns, color = numeric_columns)
        scatter.on_selection(self.selection_fn)

        # Put everything together
        return VBox((HBox(axis_dropdowns.children),self.f))
    
    def update_axes(self, xaxis, yaxis,color):
        scatter = self.f.data[0]
        scatter.x = self.df[xaxis]
        scatter.y = self.df[yaxis]
        scatter.marker.color = self.df[color]
        with self.f.batch_update():
            self.f.layout.xaxis.title = xaxis
            self.f.layout.yaxis.title = yaxis

        self.outlier_df = pd.DataFrame(columns=self.df.columns.values)

    def selection_fn(self,trace,points,selector):
        """
        Keeping track of points manually selected and change values in self.outlier_df["manual_outlier"].
        Value for points not manually selected is -1. If selected to be an outlier, value is set to 1.
        If selected again not to be an outlier, value is set to 0.

        Previous value is stored in temp_df if user chooses to undo selection. 
        """
        temp_df = self.df.loc[points.point_inds]
        
        last_selected = len(temp_df)
        # self.old_selected += len(temp_df)
        for i in temp_df.iterrows():
            idx = i[0]
            # Remember when combining with model that manual_outlier should override model_outlier
            # in the plot if value is not -1. Do a plot_outlier column.
            temp_df.at[idx, "last_selected"] = last_selected
            temp_df.at[idx, "manual_outlier"] = 1 if self.df.at[idx, "manual_outlier"] != 1 else 0

        self.outlier_df = pd.concat([self.outlier_df, temp_df], ignore_index=False, axis=0)
        print(f"outlier_df: {self.outlier_df}")
        print(f"temp_df: {temp_df}")
        # Kom ihåg att ändra i metodbeskrivningen
        
        print(f"Selected {last_selected} new points. Total: {len(self.outlier_df)}")

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


if __name__ == "__main__":
    df = create_fake_df(500)
    print(df)

    chooser = interactive_data_chooser(df, df.columns)
    chooser.activate_plot()
