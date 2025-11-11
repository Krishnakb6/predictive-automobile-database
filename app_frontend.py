import streamlit as st
import requests
import pandas as pd
from datetime import date


BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🌎 Global Mechanics Data Platform")


@st.cache_data(ttl=10) 
def get_backend_status():
    """Checks the backend /status/ endpoint."""
    try:
        response = requests.get(f"{BACKEND_URL}/status/")
        if response.status_code == 200:
            return response.json()
        else:
            return {"model_status": "offline"}
    except requests.exceptions.ConnectionError:
        return {"model_status": "offline", "message": "Backend is offline."}


st.sidebar.title("Navigation")
status = get_backend_status()
if status and status.get("model_status") == "ready":
    st.sidebar.success("✅ ML Model Ready")
elif status.get("model_status") == "loading":
    st.sidebar.warning("⏳ ML Model is loading...")
else:
    st.sidebar.error("❌ Backend Offline")

tab = st.sidebar.radio("Go to:", ["Ingestion & Normalization", "Database Viewer", "Predictive Maintenance"])



if tab == "Ingestion & Normalization":
    st.header("Data Ingestion & Normalization")
    st.info("Upload your CSV or Excel file. The system will automatically map any matching columns.")
    st.subheader("Supported Column Names")
    supported_cols = [
        'mech_name', 'mech_phone', 'location_info', 'operating_hours', 
        'slot_details', 'part_details', 'employee_details', 'project_info', 
        'client_info', 'order_details'
    ]
    st.code(f"{', '.join(supported_cols)}")
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel Files", 
        accept_multiple_files=True, 
        type=['csv', 'xlsx', 'xls']
    )
    if uploaded_files:
        for file in uploaded_files:
            files = {"file": (file.name, file.getvalue())} 
            try:
                response = requests.post(f"{BACKEND_URL}/upload_csv/", files=files)
                if response.status_code == 200:
                    st.success(f"{response.json().get('message')}")
                else:
                    st.error(f"Error uploading {file.name}: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError as e:
                 st.error(f"Could not connect to backend: {e}")
    st.divider()
    if st.button("Trigger Normalization", type="primary"):
        try:
            response = requests.post(f"{BACKEND_URL}/normalize/")
            if response.status_code == 200:
                st.success(f"Normalization triggered successfully! {response.json().get('message')}")
            else:
                st.error(f"Error triggering normalization: {response.json().get('detail')}")
        except requests.exceptions.ConnectionError as e:
            st.error(f"Could not connect to backend: {e}")


elif tab == "Database Viewer":
    st.header("Global Database Viewer")

    def clear_filters_and_page():
        st.session_state.active_filters = {}

    try:
        response = requests.get(f"{BACKEND_URL}/tables/")
        if response.status_code == 200:
            if "table_info" not in st.session_state:
                st.session_state.table_info = response.json().get("tables", {})
            
            table_info = st.session_state.table_info
            
            
            viewable_tables = {k: v for k, v in table_info.items() if k != 'ServiceCenters'}

            selected_table = st.selectbox(
                "Select a table to view", 
                viewable_tables.keys(), 
                on_change=clear_filters_and_page
            )

            if selected_table:
                columns = table_info.get(selected_table, [])
                
                with st.expander("Search & Filter"):
                    with st.form(key=f"{selected_table}_filter_form"):
                        filter_inputs = {}
                        form_cols = st.columns(3)
                        
                        for i, col_name in enumerate(columns):
                            default_val = st.session_state.get('active_filters', {}).get(col_name, "")
                            filter_inputs[col_name] = form_cols[i % 3].text_input(
                                f"Filter by {col_name}", 
                                value=default_val
                            )
                        
                        submitted = st.form_submit_button("Apply Filters")
                        
                        if submitted:
                            st.session_state.active_filters = {
                                k: v for k, v in filter_inputs.items() if v.strip()
                            }
                            st.session_state[f"{selected_table}_page"] = 1
                            st.rerun()

                page = st.session_state.get(f"{selected_table}_page", 1)
                active_filters = st.session_state.get('active_filters', {})
                
                if active_filters:
                    st.info(f"Active Filters: `{active_filters}`")

                col1, col2, col3 = st.columns([1.5, 1.5, 7]) 
                if col1.button("⬅️ Previous Page", use_container_width=True):
                    if page > 1:
                        st.session_state[f"{selected_table}_page"] = page - 1
                        st.rerun()
                
                if col2.button("Next Page ➡️", use_container_width=True):
                    st.session_state[f"{selected_table}_page"] = page + 1
                    st.rerun()
                
                col3.write(f"Displaying Page: {page}")

                payload = {
                    "table_name": selected_table,
                    "page": page,
                    "page_size": 10,
                    "filters": active_filters
                }
                
                data_response = requests.post(f"{BACKEND_URL}/data/query/", json=payload)
                
                if data_response.status_code == 200:
                    data = data_response.json()
                    if data:
                        st.dataframe(pd.DataFrame(data), use_container_width=True)
                    else:
                        st.warning("No data found for the current page and filters.")
                        if page > 1:
                            st.session_state[f"{selected_table}_page"] = page - 1
                else:
                    st.error(f"Error fetching data: {data_response.text}")
        else:
            st.error(f"Error fetching table list: {response.text}")
    except requests.exceptions.ConnectionError as e:
        st.error(f"Could not connect to the backend: {e}. Please ensure the backend is running.")


# pred main
elif tab == "Predictive Maintenance":
    st.header("⚙️ Predictive Maintenance & Scheduling")
    st.info("Fill in vehicle details. Leave fields blank to use the dataset's average values for prediction.")

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    with st.form("prediction_form"):
        
        #form layout
        st.subheader("Vehicle & Usage")
        col1, col2, col3 = st.columns(3)
        with col1:
            Vehicle_Model = st.selectbox("Vehicle Model", ["", "Truck", "Van", "Bus", "Car", "SUV"])
            Mileage = st.number_input("Mileage", min_value=0, value=0, step=1000)
            Vehicle_Age = st.number_input("Vehicle Age (Years)", min_value=0, value=0, step=1)
        with col2:
            Fuel_Type = st.selectbox("Fuel Type", ["", "Electric", "Petrol", "Diesel"])
            Odometer_Reading = st.number_input("Odometer Reading", min_value=0, value=0, step=1000)
            Engine_Size = st.number_input("Engine Size (CC)", min_value=0, value=0, step=100)
        with col3:
            Transmission_Type = st.selectbox("Transmission Type", ["", "Automatic", "Manual"])
            Fuel_Efficiency = st.number_input("Fuel Efficiency (e.g., 15.5)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
            Insurance_Premium = st.number_input("Insurance Premium", min_value=0, value=0, step=100)
        st.divider()
        st.subheader("Service & History")
        col4, col5, col6 = st.columns(3)
        with col4:
            Maintenance_History = st.selectbox("Maintenance History", ["", "Poor", "Average", "Good"])
            Reported_Issues = st.number_input("Reported Issues (Count)", min_value=0, value=0, step=1)
            Owner_Type = st.selectbox("Owner Type", ["", "First", "Second", "Third"])
        with col5:
            Service_History = st.number_input("Service History (Count)", min_value=0, value=0, step=1)
            Accident_History = st.number_input("Accident History (Count)", min_value=0, value=0, step=1)
            Tire_Condition = st.selectbox("Tire Condition", ["", "Worn Out", "Good", "New"])
        with col6:
            today = date.today()
            Last_Service_Date = st.date_input("Last Service Date", value=None, max_value=today)
            Warranty_Expiry_Date = st.date_input("Warranty Expiry Date", value=None)
        st.divider()

        submit_col1, submit_col2 = st.columns(2)
        predict_button = submit_col1.form_submit_button("1. Predict Maintenance Need", use_container_width=True)
        schedule_button = submit_col2.form_submit_button("2. Predict & Generate Schedule", type="primary", use_container_width=True)

        if predict_button or schedule_button:
            st.session_state.prediction_result = None 
            st.session_state.search_results = [] 
            
            payload = {
                "Vehicle_Model": Vehicle_Model if Vehicle_Model else None,
                "Mileage": Mileage if Mileage > 0 else None,
                "Maintenance_History": Maintenance_History if Maintenance_History else None,
                "Reported_Issues": Reported_Issues if Reported_Issues > 0 else None,
                "Vehicle_Age": Vehicle_Age if Vehicle_Age > 0 else None,
                "Fuel_Type": Fuel_Type if Fuel_Type else None,
                "Transmission_Type": Transmission_Type if Transmission_Type else None,
                "Engine_Size": Engine_Size if Engine_Size > 0 else None,
                "Odometer_Reading": Odometer_Reading if Odometer_Reading > 0 else None,
                "Last_Service_Date": Last_Service_Date.isoformat() if Last_Service_Date else None,
                "Warranty_Expiry_Date": Warranty_Expiry_Date.isoformat() if Warranty_Expiry_Date else None,
                "Owner_Type": Owner_Type if Owner_Type else None,
                "Insurance_Premium": Insurance_Premium if Insurance_Premium > 0 else None,
                "Service_History": Service_History if Service_History > 0 else None,
                "Accident_History": Accident_History if Accident_History > 0 else None,
                "Fuel_Efficiency": Fuel_Efficiency if Fuel_Efficiency > 0 else None,
                "Tire_Condition": Tire_Condition if Tire_Condition else None
            }

            endpoint_url = ""
            if predict_button:
                endpoint_url = f"{BACKEND_URL}/predict/"
            elif schedule_button:
                endpoint_url = f"{BACKEND_URL}/predict_schedule/"

            try:
                response = requests.post(endpoint_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.prediction_result = result 
                else:
                    st.session_state.prediction_result = None
                    try:
                        detail = response.json().get('detail', 'Unknown API error')
                        st.error(f"Error from API: {detail}")
                    except requests.exceptions.JSONDecodeError:
                        st.error(f"Server returned a non-JSON error (Status {response.status_code}):")
                        st.code(response.text)
            except requests.exceptions.ConnectionError as e:
                st.error(f"Could not connect to backend: {e}")

    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        prediction = result.get("prediction")
        probability = result.get("probability", 0)
        
        st.divider()
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"**Maintenance Required** (Confidence: {probability:.2%})")
            
            # show suggested date 
            if result.get("schedule_message"):
                st.subheader("Service Schedule Suggestion")
                message = result.get("schedule_message")
                suggested_date = result.get("suggested_service_date")
                
                if suggested_date:
                    st.info(f"{message} \n\nSuggested Service Date: **{suggested_date}**")
                else:
                    st.info(message)
            
            # service booking
            st.subheader("Book Your Service")
            st.write("Find a service center in your area.")
            
            location_query = st.text_input("Enter your location (e.g., city or state)")
            
            if st.button("Search Centers", type="primary"):
                st.session_state.search_results = []
                if location_query:
                    try:
                        params = {"location": location_query}
                        search_response = requests.get(f"{BACKEND_URL}/search_centers/", params=params)
                        if search_response.status_code == 200:
                            centers = search_response.json()
                            if centers:
                                st.success(f"Found {len(centers)} service centers matching '{location_query}':")
                                st.session_state.search_results = centers
                            else:
                                st.warning(f"No service centers found matching '{location_query}'.")
                        else:
                            st.error(f"Error searching centers: {search_response.text}")
                    except requests.exceptions.ConnectionError as e:
                        st.error(f"Could not connect to backend: {e}")
                else:
                    st.warning("Please enter a location to search.")
            
            if st.session_state.search_results:
                centers = st.session_state.search_results
                
                for center in centers:
                    col1, col2 = st.columns([3, 1]) 
                    
                    with col1:
                        st.markdown(f"""
                        **{center.get('name', 'N/A')}**
                        - **Phone:** {center.get('phone', 'N/A')}
                        - **Location:** {center.get('location', 'N/A')}
                        - **Hours:** {center.get('hours', 'N/A')}
                        """)
                    
                    with col2:
                        book_key = f"book_{center.get('name', 'unknown')}_{center.get('phone', 'unknown')}"
                        
                       
                        if st.button("Book Now", key=book_key, use_container_width=True):
                            try:
                                #call
                                booking_payload = {"service_center_name": center.get('name')}
                                book_response = requests.post(f"{BACKEND_URL}/book_service/", json=booking_payload)
                                
                                if book_response.status_code == 200:
                                    # Get booking id
                                    booking_id = book_response.json().get('booking_id')
                                    st.toast(f"Booking Confirmed! Your ID is {booking_id} 🚗💨", icon="✅")
                                else:
                                    st.error(f"Booking failed: {book_response.json().get('detail')}")
                            
                            except requests.exceptions.ConnectionError as e:
                                st.error(f"Could not connect to backend: {e}")
                    
                    st.divider() 

        else:
            st.success(f"**Maintenance Not Required** (Confidence: {probability:.2%})")