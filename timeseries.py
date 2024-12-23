# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:03:28 2024

@author: Rahul Nigam
"""


"""### Time Sereis Analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import streamlit as st
from pandas.io.json import json_normalize

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from pmdarima.arima import auto_arima
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing

from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon

import data
import eda

def tsa(df,days):
    global company
    company = data.company_name

    
    st.sidebar.write("#### Timse Series Models :")
    checkbox1 = st.sidebar.checkbox('Smoothing Models', value=False, disabled=False, key='checkbox1')
    
    if checkbox1:
        st.sidebar.write("##### Tuning Parameters(Smoothing) :")
        alpha = st.sidebar.number_input("***alpha***", 0.00, 1.0, step=0.1, value=0.9, key="alpha")
        beta = st.sidebar.number_input("***beta***", 0.00, 1.0, step=0.01, value=0.01, key="beta")
        phi = st.sidebar.number_input("***phi***", 0.00, 1.0, step=0.1, value=0.2, key="phi")
      
    checkbox2 = st.sidebar.checkbox('ARIMA', key='checkbox2')
    checkbox3 = st.sidebar.checkbox('ETS', key='checkbox3')
    
    @st.cache_data
    def data_split(days):
        ####### Data Splitting #######
        split_index=len(y)-days
        y_train=y[:split_index]
        y_test=y[split_index:]
        return y_train,y_test
      
    def plot_future_data():
        fig=plt.figure(figsize=(10,6))  #(10,6)
        plt.title(f'Forcasting Plot by {best_model}',fontsize = 20)
        plt.xlabel('Days',fontsize = 16)
        plt.ylabel('Close Price(Rs.)',fontsize=16)
        plt.plot(forecasting_df['Forecast'])
        st.pyplot(fig)
        
    def actual_and_predicted(days):
        global valid
        valid = df002[y_train.shape[0]:]
        valid['Date'] = df001['date'] 
        valid['prediction'] = best_predictions #np.round(best_predictions,2)  
        global valid_1
        valid_1 = valid[['Date','close','prediction']]
        valid_1['Date'] = df['Date']
        valid_1.index = range(1, len(valid_1) + 1)

    def plot_model_data(days):
        fig=plt.figure(figsize=(10,6))  
        plt.title(f'Original and {best_model} Prediction',fontsize = 20)
        plt.xlabel('Days',fontsize = 16)
        plt.ylabel('Close Price(Rs.)',fontsize=16)
        if best_model != 'ETS':
            plt.plot(df001['close'])
            plt.plot(valid[['close','prediction']])
            plt.legend(['Orig','Test data','Prediction'],loc='best',fontsize=16)
            st.pyplot(fig)
        else:
            plot_series(y_train_idx, y_test_idx, y_pred_idx, labels=["train", "test", "pred"])
            st.pyplot(plt)  
        st.write('')

        #### close view at the end
        fig=plt.figure(figsize=(10,6))
        plt.title(f'Test data and {best_model} Prediction',fontsize = 20)
        plt.xlabel('Days',fontsize = 16)
        plt.ylabel('Close Price(Rs.)',fontsize=16)
        if best_model != 'ETS':
            plt.plot(valid['close'],color="pink", label='Test data')
            plt.plot(valid['prediction'],color="purple", label='Prediction')
            plt.legend(loc='best',fontsize=16)
            st.pyplot(fig)
        else:
            plot_series(y_test_idx, y_pred_idx, labels=[ "test", "pred"], colors=['pink','purple'])
            plt.legend(loc='best',fontsize=16)
            st.pyplot(plt)
     
    def other_metrics(model_name,y_test,y_pred):
        mape=mean_absolute_percentage_error(y_test, y_pred)
        mape_per=mape*100
        ts_models_mape[model_name] = np.round(mape_per,2)
        ts_models_mae[model_name] = np.round(mean_absolute_error(y_test, y_pred),4)
        ts_models_rsq[model_name] = np.round(r2_score(y_test, y_pred),4)
           
    def eval_metrics():
        st.write("#### :red-background[Time Series (Evaluation Metrics) Comparison Table]")
        
        global eval_metrics_data
        eval_metrics_data = {
        'Model Name':list(ts_models_rmse.keys()),
        'Root Mean Squared Error[RMSE]':list(ts_models_rmse.values()),
        'Mean Absolute Percentage Error[MAPE]':list(ts_models_mape.values()),    
        'Mean Absolute Error[MAE]':list(ts_models_mae.values()),
        'R-squared Score':list(ts_models_rsq.values())
      }
        
        eval_metrics_df=pd.DataFrame(eval_metrics_data)
        eval_metrics_df.set_index('Model Name',inplace=True)
        st.write(eval_metrics_df)
        
    def display_store_results():
        if data.connection_successful:
            result_df=pd.DataFrame(eval_metrics_data)
            jsondata_result=result_df.to_dict(orient="records")
            
            collection_lst=data.db.list_collection_names()
            if f'{company}_TSA_Results' in collection_lst:
                for doc in jsondata_result:
                    query={'Model Name':doc['Model Name']}
                    data.db[f'{company}_TSA_Results'].update_one(query,{"$set":doc},upsert=True)      
            else:
                data.db.create_collection(f'{company}_TSA_Results')
                data.db[f'{company}_TSA_Results'].insert_many(jsondata_result)
            
            if f'{company}_ML_Results' in collection_lst:
                if data.db[f'{company}_ML_Results'].find_one() is not None:
    
                    st.write("#### :red-background[ML - Models(Evaluation Metrics) Comparison Table]")
                    result_cursor=data.db[f'{company}_ML_Results'].find()
                    ml_compare_tbl=list(result_cursor)
                    result_cursor.close()
                    
                    ml_compare_df = json_normalize(ml_compare_tbl)
                    
                    # Dropping the MongoDB '_id' field
                    ml_compare_df.drop('_id', axis=1, inplace=True)
                    ml_compare_df.set_index("Model Name",drop=True,inplace=True)
                    ml_compare_df = ml_compare_df.iloc[:, [3, 1, 0, 2]]
                    st.dataframe(ml_compare_df)
                else:
                    pass
            else:
                pass    
        else:
            st.write("##### :red-background[:red[Please connect to database for storing results and data!]]")
    
    def store_model_predicted():
        # Setting the collection for predicted price to work with
        data.db.drop_collection(f'{company}_TSA({best_model})_Prediction')
        data.db.create_collection(f'{company}_TSA({best_model})_Prediction') 
        
        # Converting data into the dict format
        valid['Date'] = valid['Date'].astype("string")
        jsondata_pred = valid.to_dict(orient="records")
        data.db[f'{company}_TSA({best_model})_Prediction'].insert_many(jsondata_pred)
        
        st.success("**Results along with original & model-predicted stock price data are successfully ingested into database.**")
        st.markdown("<h2 style='text-align: center; color: green;'>**** Thank you! ****</h2>", unsafe_allow_html=True)

    ######## Smoothing Methods(statsmodel Library) #################
    """### Smoothing"""
   
    def smoothing(alpha,beta,phi,days):
        y_train,y_test=data_split(days)
        
        ######### Simple Exponential Smoothing ###########
        model='Simple Exponential Smoothing'
        #alpha = 0.9
        fit1 = SimpleExpSmoothing(y_train).fit(smoothing_level=alpha)  
        y_pred = fit1.forecast(len(y_test)) 
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)
       
        ############# Holt's Method #####################
        ### Linear Trend
        model="Holt's Linear method"
        #beta=0.01
        fit1 = Holt(y_train).fit(smoothing_level=alpha,
                                 smoothing_trend=beta)
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)

        ### *Exponential Trend*
        model="Holt's Exponential method"
        fit1 = Holt(y_train,exponential=True).fit(smoothing_level=alpha,
                                 smoothing_trend=beta)
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)

        ### Linear Damped Trend
        model="Holt's Linear(Damped)"
        #phi=0.2
        fit1 = Holt(y_train,damped_trend=True).fit(smoothing_level=alpha,
                                 smoothing_trend=beta,damping_trend=phi)
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)

        ### Exponential Damped Trend 
        model="Holt's Exponential(Damped)"
        fit1 = Holt(y_train,exponential=True,damped_trend=True).fit(smoothing_level=alpha,
                                 smoothing_trend=beta,damping_trend=phi)
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)
        
        """### Holt-Winter's Method"""
        ################# Holt-Winter's  #####################
        ## HW(Additive)
        model="Holt-Winter's Additive"
        fit1 = ExponentialSmoothing(y_train, seasonal_periods=7,
                                    trend='add', seasonal='add').fit()
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)

        ## HW(Multiplicative)
        model="Holt-Winter's Multiplicative"
        fit1 = ExponentialSmoothing(y_train, seasonal_periods=7,
                                    trend='add', seasonal='mul').fit()
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)

        #### Damped HW ####
        # Additive
        model="Holt-Winter's Additive(Damped)"
        fit1 = ExponentialSmoothing(y_train, seasonal_periods=7,
                                    trend='add', seasonal='add',damped_trend=True).fit()
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)

        # Multiplicative
        model="Holt-Winter's Multiplicative(Damped)"
        fit1 = ExponentialSmoothing(y_train, seasonal_periods=7,
                                    trend='add', seasonal='mul',damped_trend=True).fit()
        y_pred = fit1.forecast(len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)

    ################### ARIMA(pmdarima) ####################
   
    def arima(days):
        model="ARIMA"
        y_train,y_test=data_split(days)
     
        arima_model1 = auto_arima(y_train, trace=True,
                           error_action='ignore',
                           suppress_warnings=True)
        
        y_pred = arima_model1.predict(n_periods=len(y_test))
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred
        other_metrics(model, y_test, y_pred)
        
    ################# ETS(sktime Library) ###################
    def ets(days):
        model="ETS"
        
        indexed_data=df001.set_index('date')
        idx = indexed_data.index.to_period("D")
        indexed_data.index = idx
        global y_idx
        y_idx=indexed_data['close']
        
        global y_train_idx, y_test_idx
        y_train_idx, y_test_idx = temporal_train_test_split(y_idx, test_size=days) 
    
        #### Specifying the forcasting horizon
        fh = ForecastingHorizon(y_test_idx.index,is_relative=False)
        ets_forecaster = AutoETS(auto=True, sp=7, allow_multiplicative_trend=False)
        ets_forecaster.fit(y_train_idx)
        
        global y_pred_idx
        y_pred_idx = ets_forecaster.predict(fh)
        rmse = np.sqrt(mean_squared_error(y_test_idx,y_pred_idx))
        
        ts_models_rmse[model]=np.round(rmse,4)
        model_predictions[model]=y_pred_idx.values
        other_metrics(model, y_test_idx, y_pred_idx)
        
    if st.sidebar.button("**Show Prediction**", key="regressor"):
        st.subheader("Time Series Analysis Result",divider="red")

        ######### fetching cleaned data #############
        
        df001=eda.clean_data(df)
        df002=df001[['close']]
        y=np.array(df002['close'])
        y_train,y_test=data_split(days)
        
        ts_models_rmse={}
        ts_models_mape={}
        ts_models_mae={}
        ts_models_rsq={}

        model_predictions={}
        model_forecast={}

        if checkbox1:
            smoothing(alpha, beta, phi, days)
        if checkbox2:
            arima(days)
        if checkbox3:
            ets(days)
                
        ######### Best Selection #############
        
        rmse_lst=list(ts_models_rmse.values()) 
        best_rmse=min(rmse_lst)
        best_model = [model_name for model_name, rmse_val in ts_models_rmse.items() if rmse_val == best_rmse][0]
        best_predictions = model_predictions[best_model] 
        
        if best_model=="Simple Exponential Smoothing":
            fit1  = SimpleExpSmoothing(y).fit(smoothing_level=alpha)  
            fcast1 = fit1.forecast(days)
            model_forecast["Simple Exponential Smoothing"]=fcast1
            
        if best_model=="Holt's Linear method":
            fit1 = Holt(y).fit(smoothing_level=alpha,
                                     smoothing_trend=beta)
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
            
        if best_model=="Holt's Linear(Damped)": 
            fit1 = Holt(y,damped_trend=True).fit(smoothing_level=alpha,
                                     smoothing_trend=beta,damping_trend=phi)
            
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
           
        if best_model=="Holt's Exponential method":
            fit1 = Holt(y,exponential=True).fit(smoothing_level=alpha,
                                 smoothing_trend=beta)
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
            
        if best_model=="Holt's Exponential(Damped)":
            fit1 = Holt(y,exponential=True,damped_trend=True).fit(smoothing_level=alpha,
                                     smoothing_trend=beta,damping_trend=phi)
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
            
        if best_model=="Holt-Winter's Additive":
            fit1 = ExponentialSmoothing(y, seasonal_periods=7,
                                        trend='add', seasonal='add').fit()
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
            
        if best_model=="Holt-Winter's Multiplicative":
            fit1 = ExponentialSmoothing(y, seasonal_periods=7,
                                        trend='add', seasonal='mul').fit()
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
            
        if best_model=="Holt-Winter's Additive(Damped)":
            fit1 = ExponentialSmoothing(y, seasonal_periods=7,
                                        trend='add', seasonal='add',damped_trend=True).fit()
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
            
        if best_model=="Holt-Winter's Multiplicative(Damped)":
            fit1 = ExponentialSmoothing(y, seasonal_periods=7,
                                        trend='add', seasonal='mul',damped_trend=True).fit()
            fcast1 = fit1.forecast(days)
            model_forecast[best_model]=fcast1
            
        if  best_model=="ARIMA":
            arima_model = auto_arima(y, trace=True,
                               error_action='ignore',
                               suppress_warnings=True)
            
            fcast1 = arima_model.predict(n_periods=days)
            model_forecast[best_model]=fcast1
            
        if  best_model=="ETS":
            
            null_data_idx=pd.Series(np.nan*days,index=pd.date_range(df001[-1:]['date'].values[0],periods=days+1,freq='D'))
            null_idx = null_data_idx.index.to_period("D")
            null_data_idx.index = null_idx
            null_data_idx = null_data_idx[1:]
            
            future_fh = ForecastingHorizon(null_data_idx.index,is_relative=False) 
            ets_forecaster = AutoETS(auto=True, sp=7, allow_multiplicative_trend=True)
            ets_forecaster.fit(y_idx)
            fcast1=ets_forecaster.predict(future_fh)
            model_forecast[best_model]=fcast1.values
            
        future_values = model_forecast[best_model]
        forecasting_df = pd.DataFrame(future_values,columns=['Forecast'])

        forecasting_df.index = range(1, len(forecasting_df) + 1) 
        st.write(f'**Time Series-forecasted stock price for upcoming {days} days :** ',forecasting_df)
        plot_future_data()
        st.write('')
        
        actual_and_predicted(days)
        st.write(f'**Original and Predicted values by {best_model} for previous {days} days:** \n',valid_1)
             
        plot_model_data(days)
        st.write('')
        
        st.subheader("Time Series Evaluation Metrics",divider='red')
       
        st.write(f":green-background[**Current Best Time Series Algorithm :**] ***{best_model}***")    
        if best_model=="ETS":
            model_type=ets_forecaster.summary()
            st.write(":green-background[**Inferred ETS Model Type :**] ")
            st.write( model_type)
        st.write(":green-background[**Curent Best RMSE :**] ", ts_models_rmse[best_model])    
        
        eval_metrics()
        display_store_results()
        
        if data.connection_successful:
            store_model_predicted()
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>**** Thank you! ****</h2>", unsafe_allow_html=True)

        

    
    
