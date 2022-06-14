#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:09:32 2022

@author: yasmin
"""
import pandas as pd
import pandas.io.sql as psql
from sqlalchemy import create_engine, exc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

engine = create_engine('postgresql+psycopg2://root:root@localhost:49200/project')
try:
    engine.connect()
    print("success")
except exc.SQLAlchemyError:
    print("failed connection")


data_analisys=pd.read_csv("/home/yasmin/Documents/Programacion_cientifica/DB_project/tweets/data_analysis.csv")
data_science=pd.read_csv("/home/yasmin/Documents/Programacion_cientifica/DB_project/tweets/data_science.csv")
data_visualization=pd.read_csv("/home/yasmin/Documents/Programacion_cientifica/DB_project/tweets/data_visualization.csv")

try:
    dt_analisys=pd.DataFrame(data_analisys)
    dt_analisys.to_sql("analisys", con=engine,if_exists='replace')
except Exception:
    print("Error")
    
try:
    dt_science=pd.DataFrame(data_science)
    dt_science.to_sql("science", con=engine,if_exists='replace')
except Exception:
    print("Error")
    
try:
    dt_visualization=pd.DataFrame(data_visualization)
    dt_visualization.to_sql("visualization", con=engine,if_exists='replace')
except Exception:
    print("Error")

    
# Petición base de datos
query = "SELECT user_id,username,name,tweet FROM visualization WHERE language=\'en\'"
df_p1= psql.read_sql(query, engine)
print(df_p1)
