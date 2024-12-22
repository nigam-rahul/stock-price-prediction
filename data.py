# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:07:14 2024

@author: Rahul Nigam
"""
import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup as bs
import requests
from pymongo.mongo_client import MongoClient

######### /*Scrapping the Yahoo Finance website*/ ###########

def fetch_data():
    # Selecting the company of choice
    global stocks
    stocks = {'ADANIENT.BO':'Adani Enterprises','ADANIGREEN.NS':'Adani Green Energy','RELIANCE.NS':'Reliance Industries','TCS.NS':'TCS','ONGC.NS':'ONGC(PSU)','COALINDIA.NS':'Coal India(PSU)',
              'INFY.NS':'Infosys','SBIN.NS':'SBI(PSU)','WIPRO.NS':'Wipro','CIPLA.NS':'Cipla','SUNPHARMA.NS':'Sun Pharma','LICI.NS':'LIC(PSU)','DRREDDY.NS':'Dr. Reddy\'s  Lab','STARHEALTH.NS':'Star Heath Insurance',
              'LALPATHLAB.NS':'Dr. Lal Pathlab','HDFCBANK.NS':'HDFC Bank','HDFCLIFE.NS':'HDFC Life Insurance','SBILIFE.NS':'SBI Life','HINDZINC.NS':'Hindustan Zinc(PSU)',
              'WELCORP.NS':'Welspun corp.','TATASTEEL.NS':'Tata Steel','ABCAPITAL.BO':'Aditya Birla Capital','TECHM.BO' :'Tech Mahindra','KELLTONTEC.BO':'Kellton Tech'}
    stock_desc=[stock_name for ticker_key, stock_name in stocks.items()]
    
    # Selecting no. of years
    global years
    years = {'1y':'1 year', '2y':'2 years', '5y':'5 years','10y': '10 years'}
    year_desc=[year_name for key,year_name in years.items()]
  
    global selected_stock
    selected_stock = st.selectbox('##### *Company selected* ', stock_desc, index=None, placeholder='Choose one of the company',key='stock')
    st.write("***or***")
    global any_stock
    any_stock = st.text_input("##### *Any listed company*",value=None,placeholder="Enter Stock Ticker Symbol of the Company (e.g. TCS.NS or INFY.BO etc.)",key='any_stock')
    if any_stock:
        selected_stock=any_stock
        
    if st.button("Change Company"):
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        #st.success("New session!") 
    
    if selected_stock or any_stock:
        if any_stock:
            global stock_ticker
            stock_ticker=any_stock
                
        for stock_key in stocks.keys():
            if stocks[stock_key]==selected_stock:
                stock_ticker=stock_key
                
        st.sidebar.subheader(f"Past year data for {selected_stock}",divider='red')
        price_tbl = st.sidebar.checkbox("*Display Stock Price Data*",key='price_tbl')

        global selected_year
        selected_year = st.sidebar.selectbox('**Choose the number of year**', year_desc, index=None, placeholder='select year',key='year')
        
        if st.sidebar.button("Change year"):
            st.session_state.clear()
            st.cache_data.clear()
            st.cache_resource.clear()
            #st.success("Cache cleared!")
        
        if selected_year: 
### Scrapping the basic information of the selected company

            headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:104.0) Gecko/20100101 Firefox/104.0'}
            @st.cache_resource
            def web_url():  
                profile_url = f'https://finance.yahoo.com/quote/{stock_ticker}/profile/'
                try:
                    profile_response=requests.get(profile_url,headers=headers)
                    profile_response.raise_for_status()
                    soup_obj = bs(profile_response.content)    
                    return soup_obj, profile_response
                except Exception as err:
                    st.error(f"**Sorry! An error occurred - **{err}")

            soup, web_response = web_url()
            if web_response:
                # Company heading with ticker
                heading=soup.body.select('h1')
                main_name=heading[1].get_text()
                st.header(main_name,divider='red')
                
                price_section = soup.find('section', attrs={'data-testid': 'quote-price'})
                # Current Price
                price_info=price_section.section.div.select('fin-streamer')
                latest_price=price_info[0].get_text()
                st.write(f"## {latest_price} :material/currency_rupee:")
                
                # Change and percentage
                change_amt=price_info[1].get_text(strip=True)
                change_perc=price_info[2].get_text(strip=True)
                
                # Current Time
                time=price_section.section.find('div', attrs={'slot':'marketTimeNotice'})
                current_time=time.get_text(strip=True)
                
                left, middle, right = st.columns([2,3,1])
                left.write(f":blue[**{change_amt}** **{change_perc}**]")
                middle.write(f":blue-background[**{current_time}**]")
                right.caption("*Price is in INR*")
                
                ###### Company related some more information ######
                
                # Full Name
                ast_profile = soup.find('section', attrs={'data-testid': 'asset-profile'}) 
                global company_name
                company_name=ast_profile.header.get_text(strip=True)
                st.subheader(company_name)
                
                # Address
                address=ast_profile.div.div.div
                add_content=address.get_text()
                st.write(f"*{add_content}*")
                
                # Website and Contact detail
                link=ast_profile.div.div.find_all('a')
                contact_num=link[0].get_text(strip=True)
                web=link[1]
                website=web['href']
                c1, c2, c3 = st.columns([2,1,1])
                c1.write(f"[***{website}***](%s)" % website)
                c2.write(f"**{contact_num}**")
                
                st.write("#### Company Profile")
                # Description
                desc = soup.find('section', attrs={'data-testid': 'description'})
                description=desc.select('p')
                desc_content=description[0].get_text(strip=True)
                st.write(f"{desc_content}")
            
############## Fetching the Stock Price Data ##############

#### Fetchig the stock price for selected previous years ### 
            for year_key in years.keys():
                if years[year_key]==selected_year:
                    global year_id
                    year_id=year_key
            @st.cache_data
            def get_stock_price():
                ticker=yf.Ticker(stock_ticker)
                hist_data=ticker.history(period=year_id,interval='1d')
                hist_data=hist_data.reset_index()
                hist_data['Date']=hist_data['Date'].dt.date
                return hist_data
                
            global stock_data
            stock_data=get_stock_price()
            if price_tbl:
                st.subheader(f"Stock price of {company_name} for last {years[year_id]}",divider='red')
                st.dataframe(stock_data)
            return stock_data
            
        else:
            st.write(":red[**Year not selected!**]")
    else:
        st.write(":red[**Company not selected!**]")
         

################# Data Ingestion into MongoDB ######################

def db_connect():
    st.sidebar.subheader("Database Connection ",divider='red')
    
    global ingest_data
    clear=st.sidebar.checkbox("*Clear Old Data & Results*",key='clear')
    ingest_data=st.sidebar.checkbox("*Store Current Data*",key="ingest_data")
    # Entering the password for database
    mypassword = st.sidebar.text_input("**Password Required**", type="password", placeholder="Enter password for DB",key="mypassword")
    global connected
    connected=st.sidebar.button("**Connect and store**")
       
    # Sending a ping to confirm a successful connection
    try:
        # uri (uniform resource identifier) defines the connection parameters 
        uri =  f"mongodb+srv://rahul:{mypassword}@cluster0.shcf6.mongodb.net/?retryWrites=true&w=majority"
        # start client to verify the password and connecting to MongoDB server 
        global client
        client = MongoClient( uri )
        client.admin.command('ping')
        # Creating and selecting database    
        global db
        db = client.Stock_App
        global connection_successful
        connection_successful = True
    except Exception as e:
        global error_occurred
        error_occurred = e
        connection_successful = False
        
    # Storing new and deleting old  data as well as results 
    if connected and connection_successful:
        if ingest_data:
            db.drop_collection(f'{company_name}_Data')
            db.create_collection(f'{company_name}_Data')
            stock_data['Date'] = stock_data['Date'].astype("string")  
            jsondata = stock_data.to_dict(orient="records")
            db[f'{company_name}_Data'].insert_many(jsondata)
            db[f'{company_name}_Data'].delete_many({'Open':'null'})
            if clear:
                db.drop_collection(f'{company_name}_ML_Results') 
                db.create_collection(f'{company_name}_ML_Results')
                db.drop_collection(f'{company_name}_TSA_Results') 
                db.create_collection(f'{company_name}_TSA_Results') 


