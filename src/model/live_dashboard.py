"""
Live Dashboard Class for EV Demand Monitoring
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class LiveDashboard:
    def __init__(self):
        """Initialize connection to database"""
        self.conn = None
        self.db_type = None
        
        # Try PostgreSQL connection first
        try:
            self.conn = st.connection("postgresql", type="sql")
            self.db_type = "postgresql"
            st.success("✅ Connected to PostgreSQL database")
        except Exception as e:
            st. warning(f"⚠️ PostgreSQL connection failed: {e}")
            
            # Fallback to SQLite
            try:
                import sqlite3
                db_path = "src/model/live_predictions.db"
                if os.path.exists(db_path):
                    self.conn = sqlite3.connect(db_path)
                    self.db_type = "sqlite"
                    st.success(f"✅ Connected to SQLite database at {db_path}")
                else:
                    st.error(f"❌ SQLite database not found at {db_path}")
            except Exception as e2:
                st.error(f"❌ SQLite connection failed: {e2}")
    
    def get_live_data(self):
        """Fetch live prediction data from database"""
        if self.conn is None:
            st.warning("⚠️ No database connection available")
            return pd.DataFrame()
        
        try:
            if self.db_type == "postgresql":
                # Read from PostgreSQL
                query = "SELECT * FROM live_predictions ORDER BY timestamp DESC LIMIT 500"
                df = self.conn.query(query, ttl=2)
                st.info(f"📊 Loaded {len(df)} records from PostgreSQL")
                return df
                
            elif self.db_type == "sqlite":
                # Read from SQLite
                query = "SELECT * FROM live_predictions ORDER BY timestamp DESC LIMIT 500"
                df = pd.read_sql_query(query, self.conn)
                st.info(f"📊 Loaded {len(df)} records from SQLite")
                return df
                
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            
            # Try to show table structure for debugging
            try:
                if self.db_type == "sqlite":
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor. fetchall()
                    st.info(f"Available tables: {tables}")
            except:
                pass
                
            return pd.DataFrame()
    
    def create_metrics_cards(self, predictions_df):
        """Display metric cards"""
        if predictions_df.empty:
            st.warning("⚠️ No data available for metrics")
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(predictions_df))
        
        with col2:
            if 'error' in predictions_df.columns:
                mae = predictions_df['error'].abs().mean()
                st.metric("Overall MAE", f"{mae:.2f}")
            else:
                st.metric("Overall MAE", "N/A")
        
        with col3:
            if 'processing_time_ms' in predictions_df.columns:
                avg_latency = predictions_df['processing_time_ms'].mean()
                st.metric("Avg Latency", f"{avg_latency:.2f}ms")
            else:
                st.metric("Avg Latency", "N/A")
        
        with col4:
            st.metric("Confidence", "0.95")

    def create_charts(self, predictions_df):
        """Create visualization charts"""
        if predictions_df.empty:
            st.warning("⚠️ No data available for charts")
            return
        
        # Check required columns
        required_cols = ['timestamp', 'actual_sales', 'predicted_sales']
        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(predictions_df.columns)}")
            return
        
        # Time series chart
        recent = predictions_df.sort_values('timestamp'). tail(100)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=recent['timestamp'], 
            y=recent['actual_sales'], 
            name='Actual',
            mode='lines+markers'
        ))
        fig_ts.add_trace(go. Scatter(
            x=recent['timestamp'], 
            y=recent['predicted_sales'], 
            name='Predicted', 
            line=dict(dash='dash'),
            mode='lines+markers'
        ))
        fig_ts.update_layout(
            title="Real-time Sales: Actual vs Predicted", 
            height=350,
            xaxis_title="Timestamp",
            yaxis_title="Sales"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Error distribution and category performance
        col1, col2 = st. columns(2)
        
        with col1:
            if 'error' in predictions_df.columns:
                fig_err = px.histogram(
                    predictions_df, 
                    x='error', 
                    title="Error Distribution",
                    nbins=30
                )
                st.plotly_chart(fig_err, use_container_width=True)
            else:
                st. warning("No error column available")
        
        with col2:
            if 'vehicle_category' in predictions_df. columns and 'error' in predictions_df.columns:
                cat_perf = predictions_df.groupby('vehicle_category')['error']. mean(). abs(). reset_index()
                fig_cat = px.bar(
                    cat_perf, 
                    x='vehicle_category', 
                    y='error', 
                    title="MAE by Category"
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st. warning("No vehicle category data available")