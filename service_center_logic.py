import pandas as pd
import os

from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Base, metadata, and ServiceCenters table are no longer needed here,
# as the main_api.py file defines the Mechanics and Locations tables.

# create_tables function is no longer needed.

# load_service_centers function is no longer needed,
# as this is handled by the /normalize/ endpoint in main_api.py

def search_service_centers(engine, location_query: str):
    """
    Searches the dynamically populated Mechanics and Locations tables.
    """
    try:
        with engine.connect() as conn:
            # This query joins the tables and searches by location (address)
            query = text("""
                SELECT 
                    m.name, 
                    m.phone_number AS phone, 
                    l.address AS location, 
                    l.operating_hours AS hours
                FROM 
                    Mechanics m
                JOIN 
                    Locations l ON m.location_id = l.id
                WHERE 
                    l.address LIKE :location
            """)
            
            params = {"location": f"%{location_query}%"}
            
            df = pd.read_sql(query, conn, params=params)
            
        return df.to_dict('records')
    except Exception as e:
        print(f"Error searching service centers: {e}")
        return []

