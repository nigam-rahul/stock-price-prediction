# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:41:56 2024

@author: Rahul Nigam
"""
import pandas as pd
import numpy as np
import streamlit as st
import calendar

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns

import plotly.express as px
from itertools import cycle


# Importing the data.py module to get the stock price data
import data
 
############# Data cleaning and Transformations ##############

@st.cache_data
def clean_data(df):
        global company
        #company = data.stocks[data.stock_ticker]
        company = data.company_name
        
        df = df[df["Date"].isnull() == False]
        df = df[df["Open"].isnull() == False]
        df = df[df["Close"].isnull() == False]
        df = df[df["High"].isnull() == False]
        df = df[df["Low"].isnull() == False]
        
        global df001
        df001 = df[['Date','Close','Open','High','Low']] #using double square bracket to create dataframe
        df001.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)
        
        # Converting date field from object to Date format
        
        df001['date'] = pd.to_datetime(df001['date'], errors='coerce')
        df001['open'] = df001['open'].astype(float)
        df001['close'] = df001['close'].astype(float)
        df001['high'] = df001['high'].astype(float)
        df001['low'] = df001['low'].astype(float)
        return df001

########################## EDA ##############################

# Plotting raw data

@st.cache_resource
def plot_stock_price():
    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])
    fig = px.line(df001, x=df001.date, y=[df001['open'], df001['close'], df001['high'], df001['low']],labels={'date': 'Date','value':'Stock value'})
    fig.update_layout(title_text='Stock analysis chart', autosize=False, width=800,height=500 , font_size=15, font_color='black',legend_title_text='Stock Parameters',xaxis_rangeslider_visible=True)
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes( showgrid=False)
    return st.plotly_chart(fig)

def process_data(df):
    processed_data=clean_data(df)
    
    price=processed_data[['date','close','high','low']]
    price.columns = ['Date', 'Close', 'High','Low']
        
    # Extracting the Year and Month from the 'Date' column
    price['Year'] = price['Date'].dt.year
    price['Month_no'] = price['Date'].dt.month
    
    price = price.sort_values(by=['Year', 'Month_no'], ascending=[True, True])
    price['Month_Name'] = price['Month_no'].apply(lambda x: calendar.month_abbr[x])
    
    # Grouping by Year and Month, and aggregate to get max,min and mean closing prices
    #global grouped_data
    grouped_data = price.groupby(['Year', 'Month_no']).agg(
        max_close=('Close', 'max'),
        min_close=('Close', 'min'),
        avg_close=('Close','mean'),
        max_high=('High','max'),
        min_high=('High','min'),
        max_low=('Low','max'),
        min_low=('Low','min')
    ).reset_index()
    
    # Sorting by column 'Year' (descending) and then column 'Month_no' (ascending)
    grouped_data=grouped_data.sort_values(by=['Year', 'Month_no'], ascending=[True, True])
    
    # Mapping the month numbers to month names
    grouped_data['Month_Name'] = grouped_data['Month_no'].apply(lambda x: calendar.month_abbr[x])
    return grouped_data

"""### Average Close Price Trend"""
@st.cache_resource
def avg_close_trend(grouped_data):
    
    # Setting plot style
    sns.set_theme(style="whitegrid")
    
    # Creating a figure with subplots average close prices
    fig=plt.figure(figsize=(10, 6))     #(16,8) (14,8)
    
    # Plotting average close prices
    sns.lineplot(data=grouped_data.sort_values(by=['Year', 'Month_no'], ascending=[False, True]), x='Month_Name', y='avg_close', hue='Year', palette="tab10")
    plt.title('Monthly Average Close Price trend for Each Year', fontsize=20)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Close Price(Rs.)', fontsize=16)
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    # Showing Plot
    plt.tight_layout()
    plt.show()
    return st.pyplot(fig)

"""### Maximum Close Price Trend"""
@st.cache_resource
def max_close_trend(grouped_data):
    
    # Setting plot style
    sns.set_theme(style="whitegrid")
    
    # Creating a figure with subplots for both maximum and minimum close prices
    fig=plt.figure(figsize=(10, 6))
    
    # Plotting maximum close prices
    sns.lineplot(data=grouped_data.sort_values(by=['Year', 'Month_no'], ascending=[False, True]), x='Month_Name', y='max_close', hue='Year', palette="tab10", marker="o")
    plt.title('Monthly Highest Close Price trend for Each Year', fontsize=20)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Close Price(Rs.)', fontsize=16)
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    # Showing Plot
    plt.tight_layout()
    plt.show()
    return st.pyplot(fig)

"""### Minimum Close Price Trend"""
@st.cache_resource
def min_close_trend(grouped_data):
    
    fig=plt.figure(figsize=(10, 6))
    
    # Plotting minimum close prices
    sns.lineplot(data=grouped_data.sort_values(by=['Year', 'Month_no'], ascending=[False, True]), x='Month_Name', y='min_close', hue='Year', palette="tab10", marker="o")
    plt.title('Monthly Lowest Close Price trend for Each Year', fontsize=20)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Close Price(Rs.)', fontsize=16)
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    # Showing plot
    plt.tight_layout()
    plt.show()
    return st.pyplot(fig)

"""### Max-Min Close Price"""
@st.cache_resource
def max_min_close(grouped_data):
    
    max_close_data=grouped_data.loc[grouped_data.groupby('Year')['max_close'].idxmax()]
    min_close_data=grouped_data.loc[grouped_data.groupby('Year')['min_close'].idxmin()]

    # Setting the figure size
    fig=plt.figure(figsize=(10, 6))     #(14,8) (12,6) 
    
    # Calculating bar width and positions
    bar_width = 0.35
    index = np.arange(len(grouped_data['Year'].unique()))
    
    # Creating color palettes for max and min months
    unique_cat1 = max_close_data['Month_Name'].unique().tolist()
    unique_cat2 = min_close_data['Month_Name'].unique().tolist()
    unique=unique_cat1+unique_cat2
    unique_categories = list(set(unique))
    
    color_palette = sns.color_palette('bright', len(unique_categories))   #Set1,Set2,Set3
    
    color_map = dict(zip(unique_categories, color_palette))
    
    # Plotting max monthly averages (left bars)
    max_bars = plt.bar(index - bar_width/2, max_close_data['max_close'], 
                       width=bar_width, 
                       color=[color_map[cat] for cat in max_close_data['Month_Name']])
    # Plotting min monthly averages (right bars)
    min_bars = plt.bar(index + bar_width/2, min_close_data['min_close'], 
                       width=bar_width,
                       #color=[min_color_map[cat] for cat in min_monthly_avg['Month_Name']]
                       color=[color_map[cat] for cat in min_close_data['Month_Name']])
    
    # Adding value labels on top of max bars
    for bar in max_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',  
                 ha='center', va='bottom',  
                 fontsize=10,  
                 fontweight='bold',  
                 bbox=dict(facecolor='none', edgecolor='none', alpha=0.7) #facecolor='white' 
        )
    
    # Adding value labels on top of min bars
    for bar in min_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',  
                 ha='center', va='bottom',  
                 fontsize=10,  
                 fontweight='bold',  
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  
        )
    
    # Customizing the plot
    plt.title('Yearwise Maximum and Minimum Close Price', fontsize=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Close Price(Rs.)', fontsize=16)
    
    # Set x-axis ticks and labels
    plt.xticks(index, grouped_data['Year'].unique())
    
    handles = [plt.Rectangle((0,0),1,1, color=color_map[cat]) for cat in unique_categories]
    plt.legend(handles, unique_categories, title='Month',bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    return st.pyplot(fig)

"""### Max-Min High Price"""
@st.cache_resource
def max_min_high(grouped_data):
    
    max_high_data=grouped_data.loc[grouped_data.groupby('Year')['max_high'].idxmax()]
    min_high_data=grouped_data.loc[grouped_data.groupby('Year')['min_high'].idxmin()]

    # Setting the figure size
    fig=plt.figure(figsize=(10, 6))
    
    # Calculating bar width and positions
    bar_width = 0.35
    index = np.arange(len(grouped_data['Year'].unique()))
    
    # Creating color palettes for max and min months
    unique_cat1 = max_high_data['Month_Name'].unique().tolist()
    unique_cat2 = min_high_data['Month_Name'].unique().tolist()
    unique=unique_cat1+unique_cat2
    unique_categories = list(set(unique))
    
    color_palette = sns.color_palette('bright', len(unique_categories))   #Set1,Set2,Set3
    
    color_map = dict(zip(unique_categories, color_palette))
    
    # Plotting max monthly averages (left bars)
    max_bars = plt.bar(index - bar_width/2, max_high_data['max_high'], 
                       width=bar_width, 
                       color=[color_map[cat] for cat in max_high_data['Month_Name']])
    # Plotting min monthly averages (right bars)
    min_bars = plt.bar(index + bar_width/2, min_high_data['min_high'], 
                       width=bar_width,
                       color=[color_map[cat] for cat in min_high_data['Month_Name']])
    
    # Adding value labels on top of max bars
    for bar in max_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',  
                 ha='center', va='bottom',  
                 fontsize=10,  
                 fontweight='bold',  
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  
        )
    
    # Adding value labels on top of min bars
    for bar in min_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',  
                 ha='center', va='bottom',  
                 fontsize=10,  
                 fontweight='bold',  
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  
        )
    
    # Customizing the plot
    plt.title('Yearwise Maximum and Minimum High Price', fontsize=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('High Price(Rs.)', fontsize=16)
    
    # Setting x-axis ticks and labels
    plt.xticks(index, grouped_data['Year'].unique())
    
    handles = [plt.Rectangle((0,0),1,1, color=color_map[cat]) for cat in unique_categories]
    plt.legend(handles, unique_categories, title='Month',bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    plt.tight_layout()         
    plt.show()
    return st.pyplot(fig)

"""### Max-Min Low Price"""
@st.cache_resource
def max_min_low(grouped_data):
    
    max_low_data=grouped_data.loc[grouped_data.groupby('Year')['max_low'].idxmax()]
    min_low_data=grouped_data.loc[grouped_data.groupby('Year')['min_low'].idxmin()]

    # Setting the figure size
    fig=plt.figure(figsize=(10, 6))
    
    # Calculating bar width and positions
    bar_width = 0.35
    index = np.arange(len(grouped_data['Year'].unique()))
    
    # Creating color palettes for max and min months
    unique_cat1 = max_low_data['Month_Name'].unique().tolist()
    unique_cat2 = min_low_data['Month_Name'].unique().tolist()
    unique=unique_cat1+unique_cat2
    unique_categories = list(set(unique))
    
    color_palette = sns.color_palette('bright', len(unique_categories))   #Set1,Set2,Set3
    
    color_map = dict(zip(unique_categories, color_palette))
    
    # Plotting max monthly averages (left bars)
    max_bars = plt.bar(index - bar_width/2, max_low_data['max_low'], 
                       width=bar_width, 
                       color=[color_map[cat] for cat in max_low_data['Month_Name']])
    # Plotting min monthly averages (right bars)
    min_bars = plt.bar(index + bar_width/2, min_low_data['min_low'], 
                       width=bar_width,
                       #color=[min_color_map[cat] for cat in min_monthly_avg['Month_Name']]
                       color=[color_map[cat] for cat in min_low_data['Month_Name']])
    
    # Adding value labels on top of max bars
    for bar in max_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',  
                 ha='center', va='bottom',  
                 fontsize=10,  
                 fontweight='bold',  
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  
        )
    
    # Adding value labels on top of min bars
    for bar in min_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',  
                 ha='center', va='bottom',  
                 fontsize=10,  
                 fontweight='bold',  
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  
        )
    
    # Customizing the plot
    plt.title('Yearwise Maximum and Minimum Low Price', fontsize=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Low Price(Rs.)', fontsize=16)
    
    # Setting x-axis ticks and labels
    plt.xticks(index, grouped_data['Year'].unique())
    
    handles = [plt.Rectangle((0,0),1,1, color=color_map[cat]) for cat in unique_categories]
    plt.legend(handles, unique_categories, title='Month',bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    return st.pyplot(fig)


