# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:09:51 2024

@author: Rahul Nigam
"""
######### Importing necessary libraries and packages #########

import pandas as pd
import numpy as np
import math
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pandas.io.json import json_normalize

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score

# Importing the data.py and eda.py modules for data ingestion and preprocessing 
import data
import eda
 
# Importing for Time Series Analysis
import timeseries
 

def main():
    html_temp=""":rainbow[***Welcome to the web app!***] """
    st.markdown(html_temp)
    st.title("Stock Price Prediction")
    st.divider()

    df = data.fetch_data()
    
    if data.selected_stock and data.selected_year:
        company = data.company_name
        
        # getting cleaned  stock price data
        df001=eda.clean_data(df)
        grouped_data=eda.process_data(df)
        
    #################### Exploratory Data Analysis ################
        
        st.subheader("Stock Data Analysis",divider="red")
        eda.plot_stock_price()
        eda.avg_close_trend(grouped_data)
        eda.max_close_trend(grouped_data)
        eda.min_close_trend(grouped_data)
        eda.max_min_close(grouped_data)
        eda.max_min_high(grouped_data)  
        eda.max_min_low(grouped_data)
    
    ############ Initialising the MongoDb database #################
           
        # getting database connection and storing the raw data 
        data.db_connect()
        if data.connection_successful:
            st.success("Pinged your deployment. You are successfully connected to MongoDB!")
            if data.ingest_data and data.connected:
                st.success(f"**Successfully ingested all the retrieved stock data of {data.selected_stock} to the Database....**")
        else:
            st.error(f"**Sorry! Database connection failed! An error occurred -** {data.error_occurred}")
        
    ################## Implementing ML Models ######################
    
        st.sidebar.subheader("ML Models",divider="red")
        
        ########### Data Splitting ###########
        # getting the close price
        df002 = df001[['close']] #use double square bracket to create df
        
        global future_days
        future_days = st.sidebar.slider("#### **Forcasting days**", 1, 100, step=1, value=25, key="future_days")
        
        # Creating a new cloumn (predicted) shifted to 'x' days up
        df002['Shifted'] = df002[['close']].shift(-future_days)
        
        # Creating a variable to predict 'x' days out into the future
        X = np.array(df002.drop(['Shifted'],axis=1))[:-future_days]
        y = np.array(df002['Shifted'])[:-future_days] 
        
        @st.cache_data(persist=True)
        def split(df002):
            x_train , x_test , y_train , y_test = train_test_split(X, y, test_size = future_days/len(X), shuffle=False, random_state =0)
            return x_train, x_test, y_train, y_test
        
        x_train, x_test, y_train, y_test = split(df002)
        
        models = {"Time Series Analysis":'TS',"Support Vector Machine (SVM)":"SVM", "Linear Regression":"LR", "Decision Tree Regression":"DT", "Random Forest Regression":"RF"}
        regressor = st.sidebar.selectbox("#### Choose Regressor", ("Time Series Analysis","Support Vector Machine (SVM)", "Linear Regression", "Decision Tree Regression","Random Forest Regression") , index=None, placeholder='ML Algorithm')
        if regressor:
            pass
        else:
            st.write("***Plaese select the algorithm from drop-down list(side-bar) then click on Prediction button and see the results!***")       
    ######## ML-Models : Prediction, Plot and Performance ############
                        ######## Also ##########
    ######## Storing actual-predicted values & results to MongoDB ##########
        
        def plot_future_data():
            fig=plt.figure(figsize=(10,6))  # (16,8)
            plt.title(f'Forcasting Plot by {models[regressor]}',fontsize = 20)
            plt.xlabel('Days',fontsize = 16)
            plt.ylabel('Close Price(Rs.)',fontsize=16)
            plt.plot(forecast_df['Forecast'])
            st.pyplot(fig)
            
        def actual_and_predicted():
            global valid
            valid = df002[-future_days:]
            valid['Date'] = df001['date']
            valid['prediction'] = y_pred
            valid = valid.drop(['Shifted'],axis =1)
            global valid_1
            valid_1 = valid[['Date','close','prediction']]
            valid_1['Date'] = df['Date']
            valid_1.index = range(1, len(valid_1) + 1)
    
        def plot_model_data():
            fig=plt.figure(figsize=(10,6))      # (16,8)
            plt.title(f'Original and {models[regressor]} Prediction',fontsize = 20)
            plt.xlabel('Days',fontsize = 16)
            plt.ylabel('Close Price(Rs.)',fontsize=16)
            plt.plot(df001['close'][:X.shape[0]])    # X (close)
            plt.plot(valid[['close','prediction']]) # valid data, y_pred
            plt.legend(['Orig','Valid data','Prediction']) 
            plt.show()
            st.pyplot(fig)
            
            st.write('')
            
            ####### Close view ########
            fig=plt.figure(figsize=(10,6))
            plt.title(f'{models[regressor]} Prediction for last {future_days} days',fontsize = 20)
            plt.xlabel('Days',fontsize = 16)
            plt.ylabel('Close Price(Rs.)',fontsize=16)
            plt.plot(valid['close'],color="pink", label='Valid data')
            plt.plot(valid['prediction'],color="purple", label='Prediction')
            plt.legend(loc='best') # fontsize=16
            st.pyplot(fig)
            
        def eval_metrics():
            st.subheader(f"Evaluation Metrics-{regressor}",divider='red')   
            global eval_metrics_data
            eval_metrics_data = {
            'Root Mean Squared Error[RMSE]':math.sqrt(mean_squared_error(y_test, y_pred)),
            'Mean Absolute Percentage Error[MAPE]':mean_absolute_percentage_error(y_test, y_pred)*100,    
            'Mean Absolute Error[MAE]':mean_absolute_error(y_test, y_pred),
            'R-squared Score':r2_score(y_test, y_pred)
            }
            
            for metric in eval_metrics_data:
                value = eval_metrics_data[metric]
                st.write(f":green-background[**{metric}:** ] {round(value, 4)}")
                                    
        def display_store_result():
            if data.connection_successful:
                st.write("#### :red-background[ML - Models(Evaluation Metrics) Comparison Table]")
                
                global metric_columns
                metric_columns=eval_metrics_data.keys()
                
                all_columns=["Model Name"]
                all_columns.extend(metric_columns)
                
                filterr={'Model Name':regressor}
                collection_lst=data.db.list_collection_names()
                
                if f'{company}_ML_Results' in collection_lst:
                    data.db[f'{company}_ML_Results'].update_one(filterr,{"$set":eval_metrics_data},upsert=True)      
                else:
                    data.db.create_collection(f'{company}_ML_Results')
                    data.db[f'{company}_ML_Results'].insert_one({"$set":eval_metrics_data})
                       
                result_cursor=data.db[f'{company}_ML_Results'].find()
                compare_tbl=list(result_cursor)
                result_cursor.close()
                
                compare_df = json_normalize(compare_tbl)
                
                # Dropping the MongoDB '_id' field
                compare_df.drop('_id', axis=1, inplace=True)
                compare_df.set_index("Model Name",drop=True,inplace=True)
                compare_df = compare_df.iloc[:, [3, 1, 0, 2]]
                st.dataframe(compare_df)
                
                if f'{company}_TSA_Results' in collection_lst:
                    if data.db[f'{company}_TSA_Results'].count_documents({}) != 0:
                        st.write("#### :red-background[Time Series - Models(Evaluation Metrics) Comparison Table]")
                        result_cursor2=data.db[f'{company}_TSA_Results'].find()
                        ts_compare_tbl=list(result_cursor2)
                        result_cursor2.close()
                        
                        ts_compare_df = json_normalize(ts_compare_tbl)
                        
                        # Dropping the MongoDB '_id' field
                        ts_compare_df.drop('_id', axis=1, inplace=True)
                        ts_compare_df.set_index("Model Name",drop=True,inplace=True)
                        ts_compare_df = ts_compare_df.iloc[:, [3, 1, 0, 2]]
                        st.dataframe(ts_compare_df)
                    else:
                        pass
                else:
                    pass
            else:
                st.write("##### :red-background[:red[Please connect to database for storage and result comparison table!]]")
    
        def store_model_predicted():
            # Setting the collection for predicted price to work with
            data.db.drop_collection(f'{company}_ML({models[regressor]})_Prediction')
            data.db.create_collection(f'{company}_ML({models[regressor]})_Prediction') 
            
            # Converting data into the dict format
            valid['Date'] = valid['Date'].astype("string")  
            jsondata_pred = valid.to_dict(orient="records")
            data.db[f'{company}_ML({models[regressor]})_Prediction'].insert_many(jsondata_pred)
            
            st.success("**Results along with original & model-predicted stock price data are successfully ingested into database.**")
            st.markdown("<h2 style='text-align: center; color: green;'>**** Thank you! ****</h2>", unsafe_allow_html=True)
        
        ################## Time Series Analysis ##############
        
        if regressor == "Time Series Analysis":
            timeseries.tsa(df,future_days)
            
        ################## Support Vector Machine ###############
        
        if regressor == "Support Vector Machine (SVM)":
            st.sidebar.write("##### Hyperparameters :")
            C = st.sidebar.number_input("***C (Regularization parameter)***", 1.00, 10.0, step=0.01, key="C")
            kernel = st.sidebar.radio("***Kernel***", ("rbf", "linear"), key="kernel") 
            gamma = st.sidebar.radio("***Gamma (Kernal coefficient)***", ("scale", "auto"), key="gamma")
            
            if st.sidebar.button("**Show Prediction**", key="regressor"):
                st.subheader("Support Vector Machine (SVM) results",divider="red")
                model = SVR(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                x_future = df001[['close']][-future_days:]
                x_future = np.array(x_future)
                model.fit(X, y)
                model_forecast = model.predict(x_future)
                forecast_df = pd.DataFrame(model_forecast,columns=['Forecast'])
                forecast_df.index = range(1, len(forecast_df) + 1)

                st.write(f'**Forecasted stock price for upcoming {future_days} days :** ',forecast_df)
                plot_future_data()
                st.write('')
          
                actual_and_predicted()
                st.write(f'**Original and Predicted values by SVM Regression for last {future_days} days :** \n',valid_1)
                     
                plot_model_data()
                st.write('')
                
                eval_metrics()
                display_store_result()
                
                if data.connection_successful:
                    store_model_predicted()
                else:
                    st.markdown("<h2 style='text-align: center; color: green;'>**** Thank you! ****</h2>", unsafe_allow_html=True)
    
        ################ Linear Regression ###################
        
        if regressor == "Linear Regression":
            
            if st.sidebar.button("**Show Prediction**", key="regressor"):
                st.subheader("Linear Regression Results",divider="red")
                model = LinearRegression()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                x_future = df001[['close']][-future_days:]
                x_future = np.array(x_future)
                model.fit(X, y)
                model_forecast = model.predict(x_future)
                forecast_df = pd.DataFrame(model_forecast,columns=['Forecast'])
                forecast_df.index = range(1, len(forecast_df) + 1)

                st.write(f'**Forecasted stock price for upcoming {future_days} days :** ',forecast_df)
                plot_future_data()
                st.write('')
                
                actual_and_predicted()
                st.write(f'**Original and Predicted values by Linear Regression for last {future_days} days :** \n',valid_1)
                     
                plot_model_data()
                st.write('')
                
                eval_metrics()
                display_store_result()
                
                if data.connection_successful:
                    store_model_predicted()
                else:
                    st.markdown("<h2 style='text-align: center; color: green;'>**** Thank you! ****</h2>", unsafe_allow_html=True)
        
        ##################### Decision Tree #####################
                               
        if regressor == "Decision Tree Regression":
            
            if st.sidebar.button("**Show Prediction**", key="regressor"):
                st.subheader("Decision Tree Regression Results",divider="red")
                model = DecisionTreeRegressor()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                scoring=model.score(x_test, y_test)
                st.write(scoring)
                
                x_future = df001[['close']][-future_days:]
                x_future = np.array(x_future)
                model.fit(X, y)
                model_forecast = model.predict(x_future)
                forecast_df = pd.DataFrame(model_forecast,columns=['Forecast'])
                forecast_df.index = range(1, len(forecast_df) + 1)

                st.write(f'**Forecasted stock price for upcoming {future_days} days :** ',forecast_df)
                plot_future_data()
                st.write('')
                
                actual_and_predicted()
                st.write(f'**Original and Predicted values by Decision Tree Regression for last {future_days} days :** \n',valid_1)
                
                plot_model_data()
                st.write('')
                
                eval_metrics()
                display_store_result()
                
                if data.connection_successful:
                    store_model_predicted()
                else:
                    st.markdown("<h2 style='text-align: center; color: green;'>**** Thank you! ****</h2>", unsafe_allow_html=True)
    
        #################### Random Forest ######################
        
        if regressor == "Random Forest Regression":
            st.sidebar.write("##### Hyperparameters :")
            n_estimators= st.sidebar.number_input("***The number of trees in the forest***", 100, 500, step=10, key="n_estimators")    
            
            if st.sidebar.button("**Show Prediction**", key="regressor"):
                st.subheader("Random Forest Regression Results",divider="red")
                model = RandomForestRegressor(n_estimators = n_estimators)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                x_future = df001[['close']][-future_days:]
                x_future = np.array(x_future)
                model.fit(X, y)
                model_forecast = model.predict(x_future)
                forecast_df = pd.DataFrame(model_forecast,columns=['Forecast'])
                forecast_df.index = range(1, len(forecast_df) + 1)

                st.write(f'**Forecasted stock price for upcoming {future_days} days :** ',forecast_df)
                plot_future_data()
                st.write('')
                
                actual_and_predicted()
                st.write(f'**Original and Predicted values by Random Forest Regression for last {future_days} days :** ',valid_1)
                 
                plot_model_data()
                st.write('')
                
                eval_metrics()
                display_store_result()
                
                if data.connection_successful:
                    store_model_predicted()
                else:
                    st.markdown("<h2 style='text-align: center; color: green;'>**** Thank you! ****</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
            