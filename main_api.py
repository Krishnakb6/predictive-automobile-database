import json
import os
import uuid  # Added for generating booking IDs
from datetime import datetime  # Added for booking timestamp
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, inspect, text, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import io
from datetime import date

# Import all ML logic
import ml_model 
# Import service center logic
import service_center_logic

# --- Database Setup ---
DATABASE_URL = "sqlite:///./mechanics_db.sqlite"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# --- Table Definitions ---

# Staging Table (uses metadata)
Ingestion_Stage = Table('Ingestion_Stage', metadata,
    Column('mech_name', String),
    Column('mech_phone', String),
    Column('location_info', String),
    Column('operating_hours', String),
    Column('slot_details', String),
    Column('part_details', String),
    Column('employee_details', String),
    Column('project_info', String),
    Column('client_info', String),
    Column('order_details', String)
)

SUPPORTED_STAGING_COLUMNS = [
    'mech_name', 'mech_phone', 'location_info', 'operating_hours', 
    'slot_details', 'part_details', 'employee_details', 'project_info', 
    'client_info', 'order_details'
]

# BCNF Tables (uses Base)
class Mechanics(Base):
    __tablename__ = 'Mechanics'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    phone_number = Column(String)
    location_id = Column(Integer)

class Locations(Base):
    __tablename__ = 'Locations'
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String)
    operating_hours = Column(String)

class Parts(Base):
    __tablename__ = 'Parts'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)

class Slots(Base):
    __tablename__ = 'Slots'
    id = Column(Integer, primary_key=True, index=True)
    mechanic_id = Column(Integer)
    time_slot = Column(String)

# Predictive Data Table (uses Base)
class Predictive_Data(Base):
    __tablename__ = 'Predictive_Data'
    id = Column(Integer, primary_key=True, index=True)
    Vehicle_Model = Column(String)
    Mileage = Column(Integer)
    Maintenance_History = Column(String)
    Reported_Issues = Column(Integer)
    Vehicle_Age = Column(Integer)
    Fuel_Type = Column(String)
    Transmission_Type = Column(String)
    Engine_Size = Column(Integer)
    Odometer_Reading = Column(Integer)
    Last_Service_Date = Column(String)
    Warranty_Expiry_Date = Column(String)
    Owner_Type = Column(String)
    Insurance_Premium = Column(Integer)
    Service_History = Column(Integer)
    Accident_History = Column(Integer)
    Fuel_Efficiency = Column(Float)
    Tire_Condition = Column(String)
    Brake_Condition = Column(String)
    Battery_Status = Column(String)
    Need_Maintenance = Column(Integer)

# --- NEW: Bookings Table (uses Base) ---
class Bookings(Base):
    __tablename__ = 'Bookings'
    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(String, unique=True, index=True)
    service_center_name = Column(String)
    booking_time = Column(String)

# --- Create tables ---
metadata.create_all(bind=engine) 
# This now creates all class-based tables (Mechanics, Locations, Predictive_Data, AND Bookings)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Global Mechanics API")

# --- Global ML Artifacts ---
model = None
imputation_values = None

# --- Startup Event (Unchanged) ---
@app.on_event("startup")
def load_model_on_startup():
    global model, imputation_values
    try:
        # 1. Load ML Model
        model, imputation_values = ml_model.load_model_artifacts()
        print("ML model and imputation values loaded successfully.")
        
        # 2. Load Predictive Data into DB (if needed)
        data_file = "vehicle_maintenance_data.csv"
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        load_data = False
        if 'Predictive_Data' not in table_names:
            load_data = True
        else:
            with engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM Predictive_Data")).scalar()
                if count == 0:
                    load_data = True

        if load_data:
            if os.path.exists(data_file):
                pd.read_csv(data_file).to_sql('Predictive_Data', engine, if_exists='replace', index=False)
                print("Loaded CSV data into Predictive_Data table for viewing.")
            else:
                print(f"Warning: {data_file} not found. Predictive_Data table will be empty.")
        else:
            print("Predictive_Data table already exists and is populated.")

        # 3. Service Center Logic
        print("Service center logic is now dynamic and uses normalized data.")

    except FileNotFoundError as e:
        print(f"STARTUP FAILED: {e}")
        print("Please ensure 'vehicle_maintenance_data.csv' is present.")
        raise
    except Exception as e:
        print(f"Error during model loading: {e}")
        raise

# --- Pydantic Models ---
class PredictionInput(BaseModel):
    Vehicle_Model: str | None = None
    Mileage: int | None = None
    Maintenance_History: str | None = None
    Reported_Issues: int | None = None
    Vehicle_Age: int | None = None
    Fuel_Type: str | None = None
    Transmission_Type: str | None = None
    Engine_Size: int | None = None
    Odometer_Reading: int | None = None
    Last_Service_Date: str | None = None
    Warranty_Expiry_Date: str | None = None
    Owner_Type: str | None = None
    Insurance_Premium: int | None = None
    Service_History: int | None = None
    Accident_History: int | None = None
    Fuel_Efficiency: float | None = None
    Tire_Condition: str | None = None

class TableQuery(BaseModel):
    table_name: str
    page: int = 1
    page_size: int = 10
    filters: dict[str, str] | None = None

# --- NEW: Pydantic Model for Booking Request ---
class BookingRequest(BaseModel):
    service_center_name: str

# --- API Endpoints ---

@app.get("/status/")
def get_status():
    """
    Checks if the ML model is loaded and ready.
    """
    if model and imputation_values:
        return {"model_status": "ready", "message": "Ready"}
    else:
        return {"model_status": "loading", "message": "Model is loading..."}

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    # (Unchanged)
    try:
        contents = await file.read() 
        file_bytes = io.BytesIO(contents)
        
        df_raw = None
        if file.filename.endswith('.csv'):
            df_raw = pd.read_csv(file_bytes)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(file_bytes)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload a CSV or Excel file (.xls, .xlsx)."
            )
        
        df_standardized = pd.DataFrame(columns=SUPPORTED_STAGING_COLUMNS)
        common_columns = [col for col in df_raw.columns if col in SUPPORTED_STAGING_COLUMNS]

        if not common_columns:
            return JSONResponse(
                status_code=400,
                content={"detail": f"File '{file.filename}' contained no data with supported column names."}
            )

        df_standardized[common_columns] = df_raw[common_columns]
        df_standardized.to_sql('Ingestion_Stage', engine, if_exists='append', index=False)
        
        return {
            "message": f"File '{file.filename}' uploaded. Found and mapped {len(common_columns)} supported columns."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/data/query/")
def get_table_data_with_filters(query: TableQuery):
    # (Unchanged)
    inspector = inspect(engine)
    if query.table_name not in inspector.get_table_names():
        raise HTTPException(status_code=404, detail=f"Table '{query.table_name}' not found.")
    valid_columns = {col['name'] for col in inspector.get_columns(query.table_name)}

    try:
        with engine.connect() as conn:
            sql_query = f"SELECT * FROM {query.table_name}"
            
            params = {}
            where_clauses = []
            if query.filters:
                for i, (col, val) in enumerate(query.filters.items()):
                    if col in valid_columns and val:
                        param_name = f"val_{i}"
                        where_clauses.append(f"{col} LIKE :{param_name}")
                        params[param_name] = f"%{val}%" # LIKE '%%'
            
            if where_clauses:
                sql_query += " WHERE " + " AND ".join(where_clauses)

            sql_query += f" LIMIT {query.page_size} OFFSET {(query.page - 1) * query.page_size}"
            
            df = pd.read_sql(text(sql_query), conn, params=params)
            
        return json.loads(df.to_json(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying table: {str(e)}")


@app.post("/normalize/")
def normalize_data():
    # (Unchanged)
    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("DELETE FROM Mechanics"))
                conn.execute(text("DELETE FROM Locations"))
                conn.execute(text("DELETE FROM Parts"))
                conn.execute(text("DELETE FROM Slots"))
                
                df_stage = pd.read_sql("SELECT * FROM Ingestion_Stage", conn)
                if df_stage.empty:
                    return {"message": "Staging table is empty. Nothing to normalize."}

                df_locations = df_stage[['location_info', 'operating_hours']].dropna(subset=['location_info']).drop_duplicates().reset_index(drop=True)
                df_locations['id'] = df_locations.index + 1
                df_locations.rename(columns={'location_info': 'address'}, inplace=True)
                df_locations.to_sql('Locations', conn, if_exists='append', index=False)
                location_map = {row['address']: row['id'] for index, row in df_locations.iterrows()}

                df_mechanics = df_stage[['mech_name', 'mech_phone', 'location_info']].dropna(subset=['mech_name']).drop_duplicates().reset_index(drop=True)
                df_mechanics['id'] = df_mechanics.index + 1
                df_mechanics['location_id'] = df_mechanics['location_info'].map(location_map)
                df_mechanics.rename(columns={'mech_name': 'name', 'mech_phone': 'phone_number'}, inplace=True)
                df_mechanics[['id', 'name', 'phone_number', 'location_id']].to_sql('Mechanics', conn, if_exists='append', index=False)
                mechanic_map = {row['name']: row['id'] for index, row in df_mechanics.iterrows()}

                all_parts = set()
                df_stage['part_details'].dropna().str.split(',').apply(lambda parts: all_parts.update(p.strip() for p in parts))
                df_parts = pd.DataFrame(list(all_parts), columns=['name'])
                if not df_parts.empty:
                    df_parts['id'] = df_parts.index + 1
                    df_parts.to_sql('Parts', conn, if_exists='append', index=False)

                slot_data = []
                for _, row in df_stage.dropna(subset=['mech_name', 'slot_details']).iterrows():
                    mechanic_id = mechanic_map.get(row['mech_name'])
                    if mechanic_id:
                        slots = row['slot_details'].split(',')
                        for slot in slots:
                            slot_data.append({'mechanic_id': mechanic_id, 'time_slot': slot.strip()})
                
                if slot_data:
                    df_slots = pd.DataFrame(slot_data).drop_duplicates()
                    df_slots['id'] = range(1, len(df_slots) + 1)
                    df_slots.to_sql('Slots', conn, if_exists='append', index=False)

            conn.execute(text("DELETE FROM Ingestion_Stage"))
        return {"message": "Data normalized successfully and staging table cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization failed: {str(e)}")


@app.post("/predict/")
def predict(input_data: PredictionInput):
    # (Unchanged)
    if not model or not imputation_values:
        raise HTTPException(status_code=503, detail="Model is not ready. Please try again later.")
    try:
        prediction_result = ml_model.get_prediction(
            input_data=input_data.dict(), model=model, imputation_values=imputation_values
        )
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_schedule/")
def predict_schedule(input_data: PredictionInput):
    # (Unchanged)
    if not model or not imputation_values:
        raise HTTPException(status_code=503, detail="Model is not ready. Please try again later.")
    try:
        schedule_result = ml_model.get_schedule_prediction(
            input_data=input_data.dict(), model=model, imputation_values=imputation_values
        )
        return schedule_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/tables/")
def get_tables():
    # (Unchanged)
    inspector = inspect(engine)
    tables = {}
    for table_name in inspector.get_table_names():
        tables[table_name] = [col['name'] for col in inspector.get_columns(table_name)]
    return {"tables": tables}


@app.get("/search_centers/")
def search_centers(location: str = Query(..., min_length=1)):
    # (Unchanged)
    if not location:
        raise HTTPException(status_code=400, detail="Location query parameter is required.")
    results = service_center_logic.search_service_centers(engine, location)
    return results

# --- NEW: Endpoint to handle the booking ---
@app.post("/book_service/")
def book_service(booking_request: BookingRequest):
    """
    Creates a new booking record in the database.
    """
    db = SessionLocal()
    try:
        # Generate a unique booking ID
        booking_id = f"BK-{str(uuid.uuid4())[:8].upper()}"
        timestamp = datetime.now().isoformat()
        
        new_booking = Bookings(
            booking_id=booking_id,
            service_center_name=booking_request.service_center_name,
            booking_time=timestamp
        )
        
        db.add(new_booking)
        db.commit()
        db.refresh(new_booking)
        
        return {
            "message": "Booking confirmed!",
            "booking_id": new_booking.booking_id,
            "service_center_name": new_booking.service_center_name,
            "booking_time": new_booking.booking_time
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Booking failed: {str(e)}")
    finally:
        db.close()