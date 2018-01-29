#Pricing #5001
import pandas as pd
import numpy as np

import pymssql
import json

#import _tkinter

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import matplotlib.pyplot as plt
#from sklearn.neighbors import LocalOutlierFactor

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

app = Flask(__name__)
cors = CORS(app, resources=r'/zuna/api/*')

@app.route('/zuna/api/')
def index():
    return 'Yo, its working!'
@app.route('/zuna/api/v1.0/tasks', methods=['POST'])

def get_tasks():
    reqe = request.get_json(silent=True)
    host = reqe[0].get("host")
    username = reqe[0].get("username")
    password = reqe[0].get("password")
    database = reqe[0].get("database")
    item_code = reqe[0].get("item_code")
    #print(type(item_code))
    #host = "192.168.131.103"
    #username = "appuser"
    #password = "Zuna@0210"
    #database = "Zuna_1717"
    try:    
        sql = """SELECT * 
        FROM Cypher_SalesData
        WHERE ItemNumber= """ +"""'"""+ item_code + """'"""
        #print(sql)
        conn = pymssql.connect(host, username, password, database, item_code)
        #print(conn)
        df2 = pd.read_sql(sql, conn)
    except:
        return jsonify([{"error" : "SQL Error"}])

    if len(df2) < 20:
        return jsonify([{"error" : "data are too small to process"}])
    mean_price = df2[['Week', 'ListPrice']].groupby('Week').agg([np.mean])
    mean_quantity = df2[['Week', 'QtyShipped']].groupby('Week').agg([np.mean])
    mean_price['QtyShipped'] = mean_quantity['QtyShipped'].values
    mean_df = mean_price
    a = mean_df.ListPrice.values.flatten().tolist()
    a = [ round(elem, 2) for elem in a ]
    b = mean_df.QtyShipped.values.flatten().tolist()
    b = [ round(elem, 2) for elem in b]
    
    a1 = df2.ListPrice.astype(float)
    b1 = df2.QtyShipped.astype(float)

    #hist
    df3 = df2.groupby( [ "ListPrice"] ).count()
    histPrice = df3.index.astype(float).tolist()
    histPrice = [round(x, 2) for x in histPrice]
    #####removing outliers
    # Computing IQR
    #iqr = df2['ListPrice'][df2['ListPrice'].between(df2['ListPrice'].quantile(.25), df2['ListPrice'].quantile(.75), inclusive=True)]
    #np.random.seed(42)
    #X = df2.ListPrice.reshape(-1,1)
    # fit the model
    #clf = LocalOutlierFactor(n_neighbors=20)
    #y_pred = clf.fit_predict(X)
    #y_pred_outliers = y_pred[df2.size:]
    #df_Clean_Price = X[y_pred==1]
    #df_Clean_Price = iqr

    w = df2.ListPrice.sort_values()
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3-q1
    low = q1 - (1.5*iqr)
    up = q3 + (1.5*iqr)
    df_Clean = df2[(df2.ListPrice >= low) | (df2.ListPrice <= up)]
    

    ##### binning
    data = np.array(df_Clean.ListPrice,dtype='float64')
    bins = np.linspace(data.min(), data.max(), 5)
    digitized = np.digitize(data, bins)
    Binned_Price_mean = [data[digitized == i].mean() for i in range(1, len(bins)+1)]
    Binned_Price_mean = [round(x, 2) for x in Binned_Price_mean]
    Binned_Price_mean = [x for x in Binned_Price_mean if str(x) != 'nan'] 

    cleandf = df_Clean
    cleandf['bins']= digitized
    cleandf['QtyShipped'] = cleandf['QtyShipped'].astype('float64') 
    cleandf['ListPrice'] = cleandf['ListPrice'].astype('float64') 
    Binned_Qty_mean = cleandf.groupby('bins')['QtyShipped'].mean()
    Binned_Qty_mean = [round(x, 2) for x in Binned_Qty_mean]

    Binned_days = np.unique(digitized, return_counts=True)[1]
    
    #Regression
    c = cleandf.ListPrice.reshape(-1,1)
    d = cleandf.QtyShipped.reshape(-1,1)
     
    c_train, c_test, d_train, d_test = train_test_split(c, d, test_size=0.25, random_state=1)
 
    reg = linear_model.LinearRegression()
 
    reg.fit(np.log(c_train), np.log(d_train))
    coeff = reg.coef_

    data = {
        "Line_Plot" :[{
            "Price": a,
            "Quantity": b,
            "Week": mean_df.index.tolist()        
        }],
        "Scatter_Plot" :[{
            "Price": a1.tolist(),
            "Quantity": b1.tolist()    
        }],
        "Histogram" :[{
            "Price": histPrice,
            "Days": df3.Week.tolist() 
        
        }],
        "Price_Details": [{
            "Minimum Price": a1.min().tolist(),
            "Maximum Price": a1.max().tolist(),
            "Most Frequently Occuring Price": a1.mode().tolist()
        }],
        "Quantity_Details": [{
            "Minimum Quantity":b1.min().tolist(),
            "Maximum Quantity": b1.max().tolist(),
            "Average Quantity": round(b1.mean().tolist(),2)
    }],
        "Binning_clean_data": [{
            "Binned_Price_mean":Binned_Price_mean,
            "Binned_Qty_mean":Binned_Qty_mean,
            "Binned_days":Binned_days.tolist()            
        }],
	"Regression_Coefficient": [{
            "Regression_Coeff":coeff.tolist()
        }]
        
            
    }
    #print(data)
    return jsonify(data)
    #return jsonify([{"error" : "data are too small to process"}])

	
if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5001, debug=True)

#jsonObject = [{

#        "host": "192.168.131.103",
#        "username": "appuser",
#        "password": "Zuna@0210",
#        "database": "Zuna_1717"
#    }
	
#];
# ps -fA | grep python
#sudo kill -9 process no.



