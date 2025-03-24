import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="Balloon Flight & Weather Dashboard", layout="wide")

import requests
import pandas as pd
import datetime
import time
import math
import plotly.express as px
from openai import OpenAI
from io import StringIO

st.title("Dynamic Balloon Flight & Weather Insight Dashboard")
st.markdown("""
This dashboard displays the live flight history of our global sounding balloons over the past 24 hours. 
It robustly extracts flight data from our live constellation API and visualizes balloon trajectories on an interactive map.
Weather data integration and LLM-driven Q&A feature offer additional operational insights.
""")

# Function to robustly fetch JSON data from a URL, handling corrupted data
def fetch_json(url):
    try:
        response = requests.get(url, timeout=15)
        
        # First attempt: try normal JSON parsing
        try:
            return response.json()
        except Exception as json_error:
            # The API returns corrupted JSON, so let's try to repair it
            content = response.text
            
            # Case 1: Try to repair common JSON errors
            try:
                # Replace trailing commas
                content_fixed = content.replace(",]", "]").replace(",}", "}")
                import json
                return json.loads(content_fixed)
            except:
                pass
                
            # Case 2: Try to extract any valid JSON objects
            try:
                # Look for objects enclosed in curly braces
                import re
                import json
                
                # Find all JSON-like objects in the string
                pattern = r'{[^{}]*}'
                matches = re.findall(pattern, content)
                
                valid_objects = []
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and "latitude" in obj and "longitude" in obj:
                            valid_objects.append(obj)
                    except:
                        continue
                
                if valid_objects:
                    st.info(f"Partially recovered {len(valid_objects)} objects from corrupted data in {url}")
                    return valid_objects
                
                return None
            except:
                # If all recovery attempts fail, return None
                st.warning(f"Could not recover any data from {url}: {json_error}")
                return None
    except Exception as e:
        st.warning(f"Error fetching data from {url}: {e}")
        return None

# Session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.datetime.now()

# Auto-refresh options
refresh_container = st.container()
with refresh_container:
    cols = st.columns([2, 1])
    with cols[0]:
        st.write(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    with cols[1]:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        manual_refresh = st.button("Refresh Now")
    
    if manual_refresh or (auto_refresh and (datetime.datetime.now() - st.session_state.last_refresh).seconds > 30):
        st.session_state.last_refresh = datetime.datetime.now()
        st.rerun()

# Data loading and processing
base_url = "https://a.windbornesystems.com/treasure/"

# Create a progress indicator
with st.spinner("Fetching balloon constellation data..."):
    # Loop through the past 24 hours (files 00.json to 23.json)
    all_data = []
    for i in range(24):
        filename = f"{i:02d}.json"
        url = base_url + filename
        data = fetch_json(url)
        if data is not None:
            # Handle array of arrays format [lat, lon, altitude]
            if isinstance(data, list):
                for entry in data:
                    # Check if entry is an array of 3 values [lat, lon, altitude]
                    if isinstance(entry, list) and len(entry) == 3:
                        # Skip entries with NaN values
                        if any(isinstance(val, float) and math.isnan(val) for val in entry):
                            continue
                        
                        try:
                            lat, lon, altitude = entry
                            
                            # Create timestamp from the file number (hour_ago)
                            timestamp = datetime.datetime.now() - datetime.timedelta(hours=i)
                            
                            all_data.append({
                                "lat": lat,
                                "lon": lon,
                                "altitude": altitude,
                                "timestamp": timestamp.isoformat(),
                                "source": filename,
                                "hour_ago": i
                            })
                        except Exception as e:
                            st.warning(f"Error processing entry in {filename}: {e}")
                    
                    # Also handle object format if present
                    elif isinstance(entry, dict) and all(k in entry for k in ("latitude", "longitude")):
                        try:
                            timestamp = entry.get("timestamp", 
                                               (datetime.datetime.now() - datetime.timedelta(hours=i)).isoformat())
                            
                            entry_data = {
                                "lat": entry["latitude"],
                                "lon": entry["longitude"],
                                "timestamp": timestamp,
                                "source": filename,
                                "hour_ago": i
                            }
                            
                            # Add any other fields that might be in the data
                            for key, value in entry.items():
                                if key not in ["latitude", "longitude", "timestamp"]:
                                    entry_data[key] = value
                                    
                            all_data.append(entry_data)
                        except Exception as e:
                            st.warning(f"Error processing entry in {filename}: {e}")
            
            # Also handle single object format if present
            elif isinstance(data, dict) and all(k in data for k in ("latitude", "longitude")):
                try:
                    timestamp = data.get("timestamp", 
                                     (datetime.datetime.now() - datetime.timedelta(hours=i)).isoformat())
                    
                    entry_data = {
                        "lat": data["latitude"],
                        "lon": data["longitude"],
                        "timestamp": timestamp,
                        "source": filename,
                        "hour_ago": i
                    }
                    
                    # Add any other fields that might be in the data
                    for key, value in data.items():
                        if key not in ["latitude", "longitude", "timestamp"]:
                            entry_data[key] = value
                            
                    all_data.append(entry_data)
                except Exception as e:
                    st.warning(f"Error processing data in {filename}: {e}")

# Data processing and visualization
if all_data:
    df = pd.DataFrame(all_data)
    
    # Try to convert timestamp to datetime format for better visualization
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        st.warning(f"Could not parse timestamp format: {e}")
    
    # Ensure altitude is numeric
    if 'altitude' in df.columns:
        df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
        
    # Display summary statistics
    st.subheader("Flight Data Summary")
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Total Data Points", len(df))
    with metrics_cols[1]:
        st.metric("Data Files", f"{len(df['source'].unique())} files")
    with metrics_cols[2]:
        if 'altitude' in df.columns:
            st.metric("Altitude Range", f"{df['altitude'].min():.2f}m - {df['altitude'].max():.2f}m")
        else:
            st.metric("Time Range", f"{df['hour_ago'].min()} - {df['hour_ago'].max()} hours")
    with metrics_cols[3]:
        if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            st.metric("Time Period", f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
        else:
            st.metric("Sources", f"{len(df['source'].unique())} files")
    
    # Tab-based interface for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Interactive Map", "Raw Data", "Weather Integration", "Analysis (LLM Chat)"])
    
    with tab1:
        st.subheader("Balloon Flight Trajectories")
        
        # Use Plotly for a more interactive map
        fig = px.scatter_mapbox(df, 
                           lat="lat", 
                           lon="lon", 
                           hover_name="source",
                           hover_data=["timestamp", "hour_ago", "altitude"] if "altitude" in df.columns else ["timestamp", "hour_ago"],
                           color="altitude" if "altitude" in df.columns else "hour_ago",
                           color_continuous_scale=px.colors.sequential.Viridis if "altitude" in df.columns else px.colors.cyclical.IceFire,
                           labels={"altitude": "Altitude (m)"} if "altitude" in df.columns else {},
                           zoom=1, 
                           height=600)
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        # Add trajectory lines if timestamps are available
        if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            st.subheader("Balloon Trajectories Over Time")
            
            # Group by balloon ID if available
            if 'balloon_id' in df.columns:
                fig = px.line_mapbox(df, 
                               lat="lat", 
                               lon="lon",
                               color="balloon_id",
                               hover_name="timestamp",
                               zoom=1, 
                               height=600)
                
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Raw Flight Data")
        st.dataframe(df)
        
        # Export options
        st.download_button(
            label="Download as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name=f'balloon_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
        )
    
    with tab3:
        st.subheader("Weather Data Integration")
        st.write("Integrate weather data with balloon coordinates.")
        
        weather_api_options = st.selectbox(
            "Select Weather API Provider",
            ["OpenWeatherMap", "Tomorrow.io", "Visual Crossing", "Other"]
        )
        
        api_key = st.text_input("API Key", type="password", help="Enter your API key for the selected weather provider")
        
        if api_key:
            # Sample a subset of points to avoid too many API calls
            sample_size = min(len(df), 5)
            sampled_points = df.sample(sample_size)
            
            with st.spinner(f"Fetching weather data for {sample_size} sample points..."):
                weather_data = []
                
                for idx, row in sampled_points.iterrows():
                    if weather_api_options == "OpenWeatherMap":
                        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={row['lat']}&lon={row['lon']}&appid={api_key}&units=metric"
                        weather_response = fetch_json(weather_url)
                        
                        if weather_response:
                            weather_data.append({
                                "lat": row['lat'],
                                "lon": row['lon'],
                                "timestamp": row['timestamp'] if 'timestamp' in row else "N/A",
                                "temperature": weather_response.get('main', {}).get('temp', "N/A"),
                                "humidity": weather_response.get('main', {}).get('humidity', "N/A"),
                                "wind_speed": weather_response.get('wind', {}).get('speed', "N/A"),
                                "weather": weather_response.get('weather', [{}])[0].get('description', "N/A"),
                            })
                            
                # Display the integrated data
                if weather_data:
                    st.subheader("Balloon Position + Weather Data")
                    weather_df = pd.DataFrame(weather_data)
                    st.dataframe(weather_df)
                    
                    # Weather Map visualization
                    fig = px.scatter_mapbox(
                        weather_df, 
                        lat="lat", 
                        lon="lon",
                        color="temperature",
                        size=[10] * len(weather_df),
                        hover_data=["temperature", "humidity", "wind_speed", "weather"],
                        color_continuous_scale=px.colors.sequential.Plasma,
                        zoom=1,
                        height=500
                    )
                    
                    fig.update_layout(mapbox_style="open-street-map")
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    st.subheader("Weather-Balloon Insights")
                    st.write("""
                    This integration allows you to understand the relationship between balloon behavior and weather conditions.
                    You can analyze how temperature, wind speed, and other factors affect balloon trajectory and altitude.
                    """)
                    
                    # Add a correlation chart if altitude data is available
                    if 'altitude' in df.columns and len(weather_data) > 2:
                        merged_data = pd.merge(
                            df[['lat', 'lon', 'altitude']],
                            weather_df[['lat', 'lon', 'temperature', 'wind_speed']],
                            on=['lat', 'lon'],
                            how='inner'
                        )
                        
                        if len(merged_data) > 0:
                            st.subheader("Altitude vs Weather Parameters")
                            alt_temp_fig = px.scatter(
                                merged_data, 
                                x="temperature", 
                                y="altitude",
                                trendline="ols",
                                title="Altitude vs Temperature"
                            )
                            st.plotly_chart(alt_temp_fig, use_container_width=True)
                else:
                    st.error("Failed to fetch weather data. Please check your API key and try again.")
        else:
            st.info("Enter a valid API key to fetch and integrate weather data with balloon coordinates.")
            
            # Show example of what will be displayed
            st.subheader("Example Weather Integration (Sample Data)")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Weather_map.png/800px-Weather_map.png", caption="Example Weather Overlay", use_column_width=True)
    
    with tab4:
        st.subheader("Flight Data Analysis")
        
        # Altitude distribution if available
        if 'altitude' in df.columns:
            st.subheader("Altitude Distribution")
            
            # Create altitude histogram
            altitude_fig = px.histogram(
                df, 
                x='altitude',
                nbins=30,
                title="Distribution of Balloon Altitudes",
                labels={"altitude": "Altitude (m)"}
            )
            st.plotly_chart(altitude_fig, use_container_width=True)
            
            # Create altitude vs latitude visualization
            altitude_lat_fig = px.scatter(
                df, 
                x='lat', 
                y='altitude',
                color='hour_ago',
                title="Altitude vs Latitude",
                labels={"altitude": "Altitude (m)", "lat": "Latitude"}
            )
            st.plotly_chart(altitude_lat_fig, use_container_width=True)
            
            # Create 3D visualization
            st.subheader("3D Balloon Positions")
            fig_3d = px.scatter_3d(
                df, 
                x='lat', 
                y='lon', 
                z='altitude',
                color='hour_ago',
                title="3D Visualization of Balloon Positions",
                labels={"altitude": "Altitude (m)"}
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Time series analysis if timestamp is available
        if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            st.subheader("Temporal Analysis")
            
            # Resample data by hour
            try:
                df_resampled = df.set_index('timestamp').resample('1H').count()['lat'].reset_index()
                df_resampled.columns = ['timestamp', 'count']
                
                # Plot time series
                fig = px.line(
                    df_resampled, 
                    x='timestamp', 
                    y='count',
                    title="Balloon Data Points by Hour"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # If altitude is available, plot average altitude over time
                if 'altitude' in df.columns:
                    try:
                        # Ensure altitude is numeric
                        altitude_data = df.copy()
                        altitude_data['altitude'] = pd.to_numeric(altitude_data['altitude'], errors='coerce')
                        
                        # Drop NaN values
                        altitude_data = altitude_data.dropna(subset=['altitude'])
                        
                        # Compute mean by hour
                        alt_by_time = altitude_data.set_index('timestamp').resample('1H').mean(numeric_only=True).reset_index()
                        
                        # Check if we got valid data
                        if 'altitude' in alt_by_time.columns and not alt_by_time['altitude'].empty:
                            alt_time_fig = px.line(
                                alt_by_time,
                                x='timestamp',
                                y='altitude',
                                title="Average Balloon Altitude Over Time",
                                labels={"altitude": "Average Altitude (m)"}
                            )
                            st.plotly_chart(alt_time_fig, use_container_width=True)
                        else:
                            st.info("Could not generate altitude time series due to insufficient data.")
                    except Exception as e:
                        st.warning(f"Could not analyze altitude time series: {e}")
                
            except Exception as e:
                st.warning(f"Could not perform time series analysis: {e}")
        
        # LLM-driven insights
        st.subheader("LLM-Driven Insights")
        st.markdown("If you'd like to use OpenAI to analyze flight patterns and get operational insights, please enter your API key below.")

        openai_api_key = st.text_input("OpenAI API Key", type="password")

        if openai_api_key:
            st.info("OpenAI integration enabled. Ask a question about the flight data:")
            query = st.text_area("Your question:", placeholder="E.g., What are the patterns in balloon altitude over time? or What operational insights can you derive from this data?")
            
            model_option = st.selectbox(
                "Select OpenAI Model",
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=1
            )
            
            if st.button("Submit Query"):
                with st.spinner("Analyzing data and generating insights..."):
                    try:
                        # Initialize OpenAI client with the provided API key
                        client = OpenAI(api_key=openai_api_key)
                        
                        # Prepare a summary of the data for the LLM
                        buffer = StringIO()
                        
                        # Basic dataset summary
                        buffer.write(f"Dataset Summary:\n")
                        buffer.write(f"- Total data points: {len(df)}\n")
                        buffer.write(f"- Time period: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}\n")
                        
                        if 'altitude' in df.columns:
                            buffer.write(f"- Altitude range: {df['altitude'].min():.2f}m to {df['altitude'].max():.2f}m\n")
                            buffer.write(f"- Average altitude: {df['altitude'].mean():.2f}m\n")
                        
                        # Geographic distribution
                        buffer.write(f"- Latitude range: {df['lat'].min():.4f} to {df['lat'].max():.4f}\n")
                        buffer.write(f"- Longitude range: {df['lon'].min():.4f} to {df['lon'].max():.4f}\n")
                        
                        # Sample of the data (10 random points)
                        buffer.write("\nSample Data Points (10 random selections):\n")
                        sample_df = df.sample(min(10, len(df)))
                        for idx, row in sample_df.iterrows():
                            buffer.write(f"- Location: ({row['lat']:.4f}, {row['lon']:.4f})")
                            if 'altitude' in row:
                                buffer.write(f", Altitude: {row['altitude']:.2f}m")
                            buffer.write(f", Time: {row['hour_ago']} hours ago\n")
                        
                        # Statistical summary
                        buffer.write("\nStatistical Summary:\n")
                        stat_summary = df.describe().to_string()
                        buffer.write(stat_summary)
                        
                        # Add altitude distribution information if available
                        if 'altitude' in df.columns:
                            buffer.write("\nAltitude Distribution (counts by range):\n")
                            altitude_bins = pd.cut(df['altitude'], bins=5)
                            altitude_counts = df.groupby(altitude_bins).size()
                            buffer.write(altitude_counts.to_string())
                        
                        data_summary = buffer.getvalue()
                        
                        # Make the OpenAI API call
                        response = client.chat.completions.create(
                            model=model_option,
                            messages=[
                                {"role": "system", "content": "You are an expert in analyzing balloon flight data and providing operational insights. The user will provide you with data about atmospheric balloon trajectories and ask questions about it."},
                                {"role": "user", "content": f"Here is the summary of balloon flight data:\n\n{data_summary}\n\nBased on this data, please answer the following question: {query}"}
                            ],
                            max_tokens=1500,
                            temperature=0.2,
                        )
                        
                        # Extract and display the response
                        llm_response = response.choices[0].message.content
                        
                        st.subheader("OpenAI Analysis")
                        st.markdown(llm_response)
                        
                    except Exception as e:
                        st.error(f"Error calling OpenAI API: {str(e)}")
                        st.info("Please check your API key and try again. Make sure you have access to the specified model.")
        else:
            st.info("Enter your OpenAI API key to enable AI-powered analysis of balloon flight patterns and operational insights.")

else:
    st.error("No valid flight data could be extracted from the API.")

st.markdown("---")
st.markdown("""
**Notes:** I chose this project because it offers an exciting opportunity to merge real-time flight data with environmental insights, 
empowering operational decisions through a richer understanding of balloon dynamics. The integration of weather data provides crucial 
context for understanding balloon behavior at different altitudes and geographical locations, while the 3D visualization capabilities
allow for better spatial understanding of the balloon constellation. The LLM-powered analysis offers deeper insights into operational
optimization and pattern recognition across the fleet.
""")
