from asyncio import exceptions
import csv
import pandas as pd
import numpy as np


from flask import render_template, flash, redirect, url_for, session, jsonify, request

from application import app, preprocessing, predict


def index():

    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        
        user_csv = request.form.get('user_csv').split('\n')
        reader = csv.DictReader(user_csv)   

        results = []

        for row in reader:
            results.append(dict(row))

        fieldnames = [key for key in results[0].keys()]

        return render_template('index.html', results=results, fieldnames=fieldnames, len=len, enumerate=enumerate)


def predict_one():
    if request.method == 'POST':
        instance = request.form.to_dict(flat=False)
        # print(instance)
        instance = instance['instance']
        sample = []
        for i in instance:
            try:
                sample.append(int(i))
            except Exception as err:
                # print(err)
                sample.append(i)

        # print(sample)
        df = pd.DataFrame(columns = ['MONTH', 'DAY', 'TIME', 'STREET1', 'STREET2',
       'ROAD_CLASS', 'DISTRICT', 'LATITUDE', 'LONGITUDE', 'LOCCOORD',
       'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVAGE',
       'DRIVACT', 'DRIVCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE',
       'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER',
       'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY'])
        df.loc[0] = sample[1:]
        y_test = pd.read_csv('application/dataset/y_test.csv', index_col=0)
        prediction = predict.predict(df, y_test.loc[sample[0]])
        classification = [sample[0], prediction[0], y_test.loc[sample[0]][0]]
        
    return render_template('prediction.html',classification=classification, len=len)  

def predict_many():
    if request.method == 'POST':
        instances = request.form.to_dict(flat=False)
        # print(f'Muliple Instances: {instances}')

        predictions = []
        for k, v in instances.items():
            sample = []
            for value in v:
                try:
                    sample.append(int(value))
                except Exception as err:
                    # print(err)
                    sample.append(value)
            df = pd.DataFrame(columns = ['MONTH', 'DAY', 'TIME', 'STREET1', 'STREET2',
            'ROAD_CLASS', 'DISTRICT', 'LATITUDE', 'LONGITUDE', 'LOCCOORD',
            'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVAGE',
            'DRIVACT', 'DRIVCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE',
            'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER',
            'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY'])
            df.loc[0] = sample[1:]
            y_test = pd.read_csv('application/dataset/y_test.csv', index_col=0)
            prediction = predict.predict(df, y_test.loc[sample[0]])
            classification = [sample[0], prediction[0], y_test.loc[sample[0]][0]]
            predictions.append(classification)

    return render_template('predictions.html', predictions=predictions, len=len, enumerate=enumerate)  
    
    
    # X_test, y_test = preprocessing.transform('application/dataset/KSI.csv')

    # if request.method == 'GET':
    #     results = []

    #     with open('application/dataset/X_test.csv', 'r') as csvFile:
    #         reader = csv.DictReader(csvFile, skipinitialspace=True)
    #         count = 0
    #         for line in reader:
    #             results.append(line)
    #             count+=1
    #         print(f'Samples: {count}')
    #         for header in results[0].keys():
    #             print(header)
    #     fieldnames = [key for key in results[0].keys()]

    #     return render_template('index.html', results=results, fieldnames=fieldnames, len=len)

    # elif request.method == 'POST':
    #     print(predict.predict(X_test, y_test))
    #     return render_template('results.html')
