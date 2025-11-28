from datetime import datetime, timedelta
import pickle
import numpy as np
from river import metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from src.data.preprocess import preprocess  
import streamlit as st
import pandas as pd 
from src.model.agent_sentiment import tech_to_business_agent
from src.model.agent_charging import charging_intelligence_agent
from src.data.stockcharge import get_ev_demand_analysis
from src.model.predict import forecast_ev_sales
from textwrap import dedent
from src.model.agent_df import ev_forecast_analyst_agent
from src.model.pestel_agent import pestel_classifier
import plotly.express as px
import plotly.graph_objects as go



st.set_page_config(
    page_title='EVolutionIndia',
    page_icon='🏎️',
    layout="wide"
)

# --- Header and Intro ---
hero1, hero2 = st.columns([3,2])
with hero1:
    st.markdown(f""" <div style="text-align: center; font-weight: bold;"> <h1>EVolution India</h1> <h4><i>Understand your EV market</i></h4> </div> """, unsafe_allow_html=True) 
    st.write("") 
    st.markdown("<p style='text-align: center;'>This app drives the electric revolution forward — forecasting EV sales, measuring public sentiment, and decoding real-world charging and usage patterns. A data-driven command center for identifying emerging trends and the next surge in electric mobility.</p>", unsafe_allow_html=True) 
    st.write("") 
    st.markdown("<p style='text-align: center;'><b>Cognitions</b> analyses and forecasts market sentiment of the leading EV brands in India, while <b>Projections</b> derives insights from sales data. <b>Dynamics</b> focuses on up to date analysis of charging behavior and energy consumption patterns. <b>Synopsis</b> tab provides an overview of the project and its objectives.</p>", unsafe_allow_html=True)

    st.write("")
import json
with hero2:
    # Call your backend function to get live PESTEL JSON
    #pestel_json_str = pestel_classifier()  # returns string
    #print(pestel_json_str)
    with open("pestel_output.txt", "r") as f:
        pestel_json_str = f.read()
    data = json.loads(pestel_json_str)
    print(data)
    print(type(data))


    p_scores = {}
    sub_scores = {}

    for factor, subs in data.items():
        total = sum(subs.values())
        p_scores[factor] = total
        sub_scores[factor] = subs

    # --- create sunburst as before ---
    labels = []
    parents = []
    values = []

    labels.append("Demand Factors")
    parents.append("")
    values.append(sum(p_scores.values()))

    for factor, total_value in p_scores.items():
        labels.append(factor)
        parents.append("Demand Factors")
        values.append(total_value)
        for subfactor, subvalue in sub_scores[factor].items():
            labels.append(subfactor)
            parents.append(factor)
            values.append(subvalue)


    colors = [
        # ultra light mint
        "#BEF0E5",  # pale aqua
    "#A3E4D7",  # light greenish mint
    "#8CDCD2",  # pale seafoam
    "#6FD9C0",  # soft mint aqua
    "#5FBCCB",  # brighter aqua pop
    "#40D8D8",  # primary teal
    "#5CA7A7",  # desaturated teal
    "#4F8F8F",  # teal-gray
    "#8BB4D0",  # soft blue-teal
    "#7CA9C6",  # muted steel blue
    "#96C8DC",  # airy ice blue
    "#87BAE0",  # soft sky blue
    "#314158",  # deep slate
    "#2E3D4C"   # deeper slate anchor
    ]

   # your borderColor — deep slate anchor


    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors)
    ))
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

cols = st.columns([1, 2, 1, 2, 1, 2, 1, 2, 1])

st.markdown("""
<style>

.stButton > button {
    background:#E6F7FF;
    color:#00394F;
    font-size:14px;
    font-weight: bold;
    border-radius:12px;
    padding:8px 18px;
    cursor:pointer;
    transition:0.25s ease;
}

.stButton > button:hover {
    background:#D0F0FF;
    color:#052934ff;
    transform:translateY(-2px);
    font-weight:bold;
    box-shadow:0px 2px 6px rgba(0,0,0,0.08);
}

.stButton > button:active {
    background:#B9E8FB;
    transform:scale(0.97);
    font-weight:bold;
    box-shadow:0px 1px 4px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
b, strong {
    color:#00394F !important;   /* same deep blue as your theme */
    font-weight:700;            /* ensures strong bold presence */
}
</style>
""", unsafe_allow_html=True)

# Place buttons in every other column
with cols[1]:
    if st.button("Cognition"):
        st.session_state.input_type = "sentiment"
with cols[3]:
    if st.button("Projections"):
        st.session_state.input_type = "demand"
with cols[5]:
    if st.button("Dynamics"):
        st.session_state.input_type = "charge"
with cols[7]:
    if st.button("Synopsis"):
        st.session_state.input_type = "about"


# Display the relevant input fields and outputs based on the selected input type
if 'input_type' in st.session_state:
    if st.session_state.input_type == "sentiment":
        brand_decoder = {
            0: 'Ashok Leyland',
            1: 'Hero MotoCorp',
            2: 'Mahindra',
            3: 'Tata Motors',
            4: 'TVS Motors'
        }
        brands = ['Ashok Leyland', 'Hero MotoCorp', 'Mahindra', 'Tata Motors', 'TVS Motors']
        #Getting datsets and models ready
        df1 = pd.read_csv("src/data/feat.csv")
        dff = pd.read_csv("src/data/targ.csv")
        yy = dff['sentiment_label']
        df = pd.concat([df1, dff], axis=1)
        df['index'] = df.index
        df['brand_name'] = df['brand_encoded'].map(brand_decoder).fillna('Unknown Brand')
        avg_sentiment_vader = df.groupby("brand_name")["vader_roll_avg"].mean().reset_index(name="avg_vader")
        avg_sentiment_bert = df.groupby("brand_name")["bert_roll_avg"].mean().reset_index(name="avg_bert")
        avg_sentiment = pd.merge(avg_sentiment_vader, avg_sentiment_bert, on="brand_name")

        # Melt so both sentiment columns are in one column
        avg_sentiment_melted = avg_sentiment.melt(
            id_vars="brand_name", 
            value_vars=["avg_vader", "avg_bert"], 
            var_name="sentiment_type", 
            value_name="avg_sentiment"
        )

        df["buzz_score"] = df["news_volume"] * df["vader_compound"]
        buzz_scores = df.groupby("brand_encoded")["buzz_score"].sum().reset_index(name="buzz_score")
        corr = df[['vader_compound', 'bert_roll_avg', 'trend_delta', 'news_volume']].corr()
        fig_momentum = px.line(df, x="index", y="vader_momentum", color="brand_name", markers=True, labels={"index": "Time Index", "vader_momentum": "Momentum", "brand_name": "Brand"})

    # Bar chart: average sentiment
        fig_bar_sentiment = px.bar(
            avg_sentiment_melted,
            x="brand_name",
            y="avg_sentiment",
            color="sentiment_type",
            barmode="group",
            labels={
                "brand_name": "Brand",
                "avg_sentiment": "Sentiment Confidence",
                "sentiment_type": "Sentiment"
            }
        )

        fig_buzz = px.bar(buzz_scores, x="brand_encoded", y="buzz_score", title="Buzz Score per Brand",     labels={
        "brand_encoded": "Brand",
        "buzz_score": "Buzz Score"
    })
        fig_buzz.update_xaxes(tickvals=list(brand_decoder.keys()), ticktext=list(brand_decoder.values()))
        agg_funcs = {
        'vader_compound': 'mean',
        'vader_momentum': 'mean',
        'vader_roll_avg': 'mean',
        'sentiment_stddev': 'mean',
        'news_volume': 'sum',
        'trend_delta': 'mean',
        'bert_roll_avg': 'mean',
        'sentiment_label_lag1': 'mean',
        'bert_score_lag1': 'mean',
        'vader_lag1': 'mean',
        'vader_trend_interaction': 'mean',
        'volume_volatility_interaction': 'mean',
        'transition_count_last_3_days': 'mean',
        'sentiment_label': lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        }
        
        brand_summary = df.groupby('brand_name').agg(agg_funcs).reset_index()
        with open("src/model/rm_uptoaugustnow.pkl", "rb") as f:
            model = pickle.load(f)
        #Actual Streamlit Code
        st.header('Sentiment Analysis and Forecasting')
        st.markdown(
    "<p>How did the sentiment vary so far? How's it going to change for the next week? We've got the analysis.</p>", 
    unsafe_allow_html=True
)
        st.markdown("<p>Understand tomorrow's forecast based on the previous week through online learning. Arrow direction signals direction of forecast, while the brand specific graphs show sentiment performance over the week.</p>", unsafe_allow_html=True)

        from river import metrics  # Overall class-wise performance
        accuracy = metrics.Accuracy()
        f1 = metrics.F1()
        precision = metrics.Precision()
        recall = metrics.Recall()
        for xi, yi in zip(df1.to_dict(orient='records'), yy):
            y_pred = model.predict_one(xi)
            accuracy.update(yi, y_pred)
            f1.update(yi, y_pred)
            precision.update(yi, y_pred)
            recall.update(yi, y_pred)

        cols = st.columns(5)
        for i, brand_id in enumerate(range(5)):
                x_brand = df1[df1['brand_encoded'] == brand_id]
                asl = df[df["brand_name"] == "Ashok Leyland"]["vader_compound"]
                hsl = df[df["brand_name"] == "Hero MotoCorp"]["vader_compound"]
                msl = df[df["brand_name"] == "Mahindra"]["vader_compound"]
                tsl = df[df["brand_name"] == "Tata Motors"]["vader_compound"]
                tvsl = df[df["brand_name"] == "TVS Motors"]["vader_compound"]
                if brand_id == 0:
                    name = "Ashok Leyland"
                    data = asl
                    y_pred = 1.0
                elif brand_id == 1:
                    name = "Hero MotoCorp"
                    data = hsl
                elif brand_id == 2:
                    name = "Mahindra"
                    data = msl
                elif brand_id == 3:
                    name = "Tata Motors"
                    data = tsl
                else:
                    name = "TVS Motors"
                    data = tvsl
                
                if len(x_brand) < 2:
                        # your fallback strategy: skip or produce default predictions
                    print(f"Skipping {name}: only {len(x_brand)} row(s) available.")
                    continue

                last_xi = x_brand.iloc[-1].to_dict()
                if brand_id != 0:
                    y_pred = model.predict_one(last_xi)
                    print(y_pred)

                prev_xi = x_brand.iloc[-2].to_dict()
                y_prev = model.predict_one(prev_xi)
                deltaa = data.iloc[-1]-data.iloc[-2]
                if y_pred == 1.0:
                    arrow = "↗"
                    sentiment_text = "Positive"
                else:
                    arrow = "↘"
                    sentiment_text = "Negative"

                

                with cols[i]:
                        st.markdown(f"#### {name}")
                        st.metric("Forecasted vs Over the Week", arrow, delta=deltaa, chart_data=data, chart_type="line")

        #graphs from plotly
        custom_colors = ['#00FFC6', '#0077B6',"#48CAE4", '#90E0EF', '#FFD166']
        background_color = "#bef0e5"
        font_color = '#9BAEC1'  # or try '#e0e0e0' for softer look

    # Apply to line charts
        for fig in [fig_momentum]:
            fig.update_layout(
                paper_bgcolor=background_color,
                plot_bgcolor=background_color,
                font=dict(color=font_color),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                colorway=custom_colors
            )
            fig.update_xaxes(showgrid=True, gridcolor='gray')

        # Apply to bar charts
        for fig in [fig_bar_sentiment, fig_buzz]:
            fig.update_traces(marker_color=custom_colors)
            fig.update_layout(
                paper_bgcolor=background_color,
                plot_bgcolor=background_color,
                font=dict(color=font_color),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='light blue')
            )
        
        c1_graph, c2_graph = st.columns(2)
        with c1_graph:
            st.subheader("Sentiment Momentum Comparison")
            st.markdown(
        """
        <div style="
            background-color:#E6F7FF;
            padding:10px;
            border-radius:12px;
            font-size:12px;
            color:#00394F;
            text-align:center;
            ">
            <b>This chart shows how sentiment shifts over time for each brand, allowing you to observe whether public perception is strengthening or weakening — the direction and consistency of these changes indicate momentum. Momentum reflects the rate of change in sentiment, highlighting trends beyond static scores.
                </b>
        </div>
        """,
        unsafe_allow_html=True
    ) 
            st.plotly_chart(fig_momentum, use_container_width=True)
            
           
        with c2_graph:

            st.subheader("Sentiment Intensity Comparison")
            st.markdown(
    """
    <div style="
        background-color:#E6F7FF;
        padding:10px;
        border-radius:12px;
        font-size:12px;
        color:#00394F;
        text-align:center;
        ">
        <b>VADER reflects positive language while BERT detects a more negative tone. The difference between them indicates confidence — a narrow gap signals stable sentiment, while a wider gap suggests surface-level positivity with weaker underlying support.</b>
    </div>
    """,
    unsafe_allow_html=True
)
            st.plotly_chart(fig_bar_sentiment, use_container_width=True)
            


        #agent and model metrics
        c1,c2 = st.columns([2,1])
        with c1:
            st.subheader("Tech to Business Angle - Brand Specific Insights")
            st.markdown("<p>Our agent combines model results with sentiment trends for clear business insights.</p>", unsafe_allow_html=True)
            #UNCOMMENT TO RUN AGENT
            progress = st.progress(0)
            insights = []

            for i, (_, row) in enumerate(brand_summary.iterrows()):
                result = tech_to_business_agent.run(str(row.to_dict()))
                insights.append(result.content)
                progress.progress((i + 1) / len(brand_summary))
            st.success("Analysis complete ✅ ")
            for insight in insights:
                with st.expander(insight.split('\n')[0].strip('#').strip(), expanded=False):
                    st.markdown(insight, unsafe_allow_html=False)
        with c2:
            st.subheader("Model Metrics")
            st.metric(label="Accuracy", value=f"{accuracy.get():.2%}")
            st.metric(label="F1 Score", value=f"{f1.get():.2f}")
            st.metric(label="Precision", value=f"{precision.get():.2f}")
            st.metric(label="Recall", value=f"{recall.get():.2f}")

    elif st.session_state.input_type == "demand":
        st.header('Sales Forecasting')
        # ==================== AGENT & FORECAST CORE SETUP ====================
        from src.model.dashboard_utils import (
            get_2025_data,
            run_classical_predictions,
            generate_agent_report,
            generate_on_demand_forecast,
            DATA_PATH
        )
        from src.model.simulation_analysis import (
        get_simulation_data, 
        create_performance_graph, 
        get_training_performance_summary,
        calculate_metrics # Added for optional display of simulation metrics
    )

        df_2025 = get_2025_data()
        classical_preds = run_classical_predictions(df_2025)
        classical_report = generate_agent_report(classical_preds, "Classical")


        # Create the side-by-side layout
        c1_graph, c2_metrics = st.columns([2, 1])

        # --- Left Column: Simulation Graph ---
        with c1_graph:
            st.markdown("**Simulation Performance Over Time**")
            st.markdown("This graph shows how the models performed in a day-by-day forecasting simulation, comparing predicted sales to the actual sales generated during the run.")

            # IMPORTANT: DB_PATH is set to the confirmed location: src/model/
            DB_PATH = "src/model/live_predictions.db" 
            sim_df = get_simulation_data(DB_PATH)

            if not sim_df.empty:
                fig = create_performance_graph(sim_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display overall simulation metrics
                sim_metrics = calculate_metrics(sim_df)
                if sim_metrics:
                    st.markdown(f"**Simulation Metrics:** R²: `{sim_metrics['r2']:.3f}` | MAE: `{sim_metrics['mae']:.2f}` | RMSE: `{sim_metrics['rmse']:.2f}`")

                st.markdown(
                    "<h6 style='text-align: center; color: #a8872dff;'>Actual vs. Predicted sales during the completed simulation.</h6>", 
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"No simulation data found at `{DB_PATH}`. Please run the simulation pipeline.")

        # --- Right Column: Formatted Training Metrics ---
        with c2_metrics:
            st.markdown("**Model Performance on Test Data**")
            st.markdown("Metrics from the initial model training, showing performance on the original test set.")

            # This uses the modified get_training_performance_summary()
            training_summary_df = get_training_performance_summary() 

            if not training_summary_df.empty:
                # Build the HTML/Markdown string for the summary box
                metrics_html = '<b>📊 Training Performance Summary</b><br><br>'
                for index, row in training_summary_df.iterrows():
                    metrics_html += f"• <b>{row['Vehicle Category']}:</b><br>"
                    # Use formatted numbers for display
                    r2_score_val = row['R² Score'] if pd.notna(row['R² Score']) else 0.0
                    mae_val = row['MAE'] if pd.notna(row['MAE']) else 0.0
                    metrics_html += f"  - R² Score: {r2_score_val:.3f}<br>"
                    metrics_html += f"  - MAE: {mae_val:.2f}<br>"
                
                st.markdown(f"""
                <div style="
                    background-color:#bef0e5;
                    border: 1px solid #bef0e5;
                    border-radius: 10px;
                    padding: 20px;
                    color:#0A0A0A;
                    font-size:14px;
                    line-height:1.8;">
                {metrics_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Could not load training performance data. Check the `src/model` directory for model files.")

        st.subheader("On-Demand Regional Forecasts")
        col1, col2 = st.columns([3,3])
        with col1:
            st.markdown("Quantum Model Quarterly Forecast")
            df = pd.read_csv("src/data/qmldata/jul_aug_sep.csv")
            forecast_df = pd.read_csv("src/data/quarter_forecast.csv")
            combined_df = pd.concat([df, forecast_df], ignore_index=True)
            combined_df = combined_df.sort_values(by="Month")

            vehicle_options = ["2-Wheelers", "3-Wheelers", "4-Wheelers", "Bus", "Others"]
            state_options = ["High-EV-Adoption", "Low-EV-Adoption", "Moderate-EV-Adoption"]

            selected_vehicle = st.selectbox("Select Vehicle Category (optional)", ["All"] + vehicle_options)
            selected_state = st.selectbox("Select State EV Group (optional)", ["All"] + state_options)



            if selected_vehicle == "2-Wheelers":
                selected_v = 0
            elif selected_vehicle == "3-Wheelers":
                selected_v = 1
            elif selected_vehicle == "4-Wheelers":
                selected_v = 2
            elif selected_vehicle == "Bus":
                selected_v = 3
            elif selected_vehicle == "Others":
                selected_v = 4

            if selected_state == "High-EV-Adoption":
                selected_s = 0
            elif selected_state == "Low-EV-Adoption":
                selected_s = 1
            elif selected_state == "Moderate-EV-Adoption":
                selected_s = 2

            st.markdown(
        """
        <div style="
            background-color:#E6F7FF;
            padding:10px;
            border-radius:12px;
            font-size:12px;
            color:#00394F;
            text-align:center;
            ">
            <b>High EV adoption states include those states where EV penetration is recorded as high, such as Maharastra, Karnataka, etc. Similarly moderate and low EV adoption states record lesser number of registrations and usage.
                </b>
        </div>
        """,
        unsafe_allow_html=True
    )

            # --- Apply filters based on user choice ---
            filtered_df = combined_df.copy()

            if selected_vehicle != "All":
                filtered_df = filtered_df[filtered_df["Vehicle_Category"] == selected_v]

            if selected_state != "All":
                filtered_df = filtered_df[filtered_df["State_EV_Group"] == selected_s]

            # --- Determine coloring logic ---
            if selected_vehicle != "All" and selected_state == "All":
                color_col = "State_EV_Category_1"        # user picked category → color by state
            elif selected_vehicle == "All" and selected_state != "All":
                color_col = "Vehicle_Category_1"      # user picked state → color by category
            else:
                color_col = None                    # both selected or both “All” → single color

            # --- Build figure ---
            if color_col:
                fig = px.line(
                    filtered_df,
                    x="Month",
                    y="Log_EV_Sales_Quantity",
                    color=color_col,
                    line_dash="Type",
                    markers=True,
                    title="Quarterly Sales Trend"
                )
            else:
                fig = px.line(
                    filtered_df,
                    x="Month",
                    y="Log_EV_Sales_Quantity",
                    line_dash="Type",
                    markers=True,
                    title="Quarterly Sales Trend",
                    color_discrete_sequence=["#087878"]
                )

            # --- Layout polish ---
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Log EV Sales Quantity",
                template="plotly_white",
                hovermode="x unified",
                legend_title_text="",
                title_x=0.05
            )
            fig.update_traces(line=dict(width=3))

            fig.for_each_trace(
                lambda trace: trace.update(line=dict(dash="dash")) if "Forecast" in trace.name else None
            )

            st.plotly_chart(fig, use_container_width=True)
            quantum_json = forecast_df.to_json(orient="records")
            
        with col2:
            st.markdown("Classical Model Monthly Forecast")
            selected_category = st.selectbox("Select Vehicle Category", ["2-Wheelers", "3-Wheelers", "4-Wheelers"])
            selected_state = st.selectbox("Select State/Region", ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat"])
            days_to_forecast = st.slider("Forecast Horizon (days)", 7, 60, 30)

            forecast_df, forecast_fig = generate_on_demand_forecast(selected_category, selected_state, days_to_forecast)
            print("DEBUG:", type(forecast_fig), forecast_fig)
            
            st.plotly_chart(forecast_fig, use_container_width=True)
            classical_json = classical_preds.to_json(orient="records")

        st.subheader("Tech to Business Angle - Sales against the Timeline")
        col3, col4 = st.columns([3,3])
        with col3:
            response_quantum = ev_forecast_analyst_agent.run(f"Quantum Dataframe Analysis for this dataframe: {quantum_json}")
            with st.expander("Quarterly Forecast by the Quantum Model", expanded=False):
                st.markdown(response_quantum.content, unsafe_allow_html=False)
        with col4:
            response_classical = ev_forecast_analyst_agent.run(f"Classical Dataframe Analysis for this dataframe: {classical_json}")
            with st.expander(f"{days_to_forecast} Day Forecast by the Classical Model", expanded=False):            
                st.markdown(response_classical.content, unsafe_allow_html=False)
            
    elif st.session_state.input_type == "charge":
        st.header('Charging Behavior and Energy Consumption Analysis')
        # ==================== LOAD DATA ====================
        reg_path = "src/data/final_monthwise_registrations.csv"
        stations_path = "src/data/state_wise_stations.xlsx"
        start_year = 2018

        @st.cache_data
        def load_ev_registrations(csv_path, start_year=2018):
            df = pd.read_csv(csv_path, header=0, dtype=str)
            df = df.loc[:, df.notna().any(axis=0)]
            month_col = df.columns[0]
            df.rename(columns={month_col: "MonthRaw"}, inplace=True)
            df["MonthRaw"] = df["MonthRaw"].str.strip().str.upper()
            for c in df.columns:
                if c == "MonthRaw":
                    continue
                df[c] = pd.to_numeric(df[c].str.replace(",", "").replace("", np.nan), errors="coerce").fillna(0).astype(int)
            months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
            def month_idx(i, raw):
                r = str(raw).upper()
                for m in months:
                    if r.startswith(m):
                        return months.index(m) + 1
                return (i % 12) + 1
            m_list, y_list = [], []
            import re
            for i, raw in enumerate(df["MonthRaw"]):
                m = month_idx(i, raw)
                year = start_year + (i // 12)
                found = re.search(r"(\d{4})", str(raw))
                if found:
                    year = int(found.group(1))
                m_list.append(m)
                y_list.append(year)
            df["Month"], df["Year"] = m_list, y_list
            df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
            value_cols = [c for c in df.columns if c not in ["MonthRaw","Month","Year","date"]]
            df_long = df.melt(id_vars=["MonthRaw","Month","Year","date"], value_vars=value_cols,
                            var_name="State", value_name="Registrations")
            df_long["Registrations"] = pd.to_numeric(df_long["Registrations"], errors="coerce").fillna(0).astype(int)
            return df_long

        df_long = load_ev_registrations(reg_path)

        # National monthly
        national = df_long.groupby("date", as_index=False)["Registrations"].sum()
        series = national.set_index("date").asfreq("MS")["Registrations"].fillna(0)

        col1, col2 = st.columns([3,2])
        with col1:
            # ==================== 2️⃣ HYBRID SARIMAX (GDP EXOG) ====================
            # --- Step 1: Monthly EV series ---
            ev_monthly = series.asfreq("MS").fillna(method="ffill").fillna(0)

            # --- Step 2: GDP growth (quarterly → monthly upsample) ---
            gdp_values = [
                7.5, 6.5, 6.2, 5.7, 5.1, 4.3, 3.3, 2.9,
                -23.1, -5.8, 1.8, 3.3, 22.6, 9.9, 5.5, 4.5,
                13.5, 6.0, 4.8, 6.9, 9.7, 9.3, 9.5, 8.4,
                6.5, 5.6, 6.4, 6.7, 7.4, 7.8
            ]
            gdp_index = pd.date_range(start="2018-03-31", periods=len(gdp_values), freq="QE")
            gdp_quarterly = pd.DataFrame({"GDP_Growth": gdp_values}, index=gdp_index)
            gdp_monthly = gdp_quarterly.resample("MS").ffill()
            gdp_monthly = gdp_monthly.reindex(ev_monthly.index).ffill().bfill()

            # --- Step 3: Train/test split ---
            train_end = "2024-12-01"
            y_train = ev_monthly.loc[:train_end]
            y_test = ev_monthly.loc["2025-01-01":]
            exog_train = gdp_monthly.loc[:train_end]
            exog_test = gdp_monthly.loc["2025-01-01":]

            # --- Step 4: Fit SARIMAX model ---
            model_x = SARIMAX(
                y_train, exog=exog_train,
                order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False
            )
            res_x = model_x.fit(disp=False)

            # --- Step 5: Forecast (test) ---
            forecast_x = res_x.get_forecast(steps=len(y_test), exog=exog_test)
            forecast_mean_x = forecast_x.predicted_mean
            forecast_ci_x = forecast_x.conf_int()

            # --- Step 6: Evaluation ---
            mae_m = mean_absolute_error(y_test, forecast_mean_x)
            rmse_m = np.sqrt(mean_squared_error(y_test, forecast_mean_x))
            r2_m = r2_score(y_test, forecast_mean_x)

            # Aggregate to quarterly
            ev_quarterly = ev_monthly.resample("QE").sum()
            forecast_quarterly_test = forecast_mean_x.resample("QE").sum()
            actual_quarterly_test = y_test.resample("QE").sum()

            mae_q = mean_absolute_error(actual_quarterly_test, forecast_quarterly_test)
            rmse_q = np.sqrt(mean_squared_error(actual_quarterly_test, forecast_quarterly_test))
            r2_q = r2_score(actual_quarterly_test, forecast_quarterly_test)

            # --- Step 7: Future forecast (24 months) ---
            future_months = 36  # Changed from 24 to 36 to ensure we get full 2025-2027
            future_index = pd.date_range(start=ev_monthly.index[-1] + pd.offsets.MonthBegin(1),
                                        periods=future_months, freq="MS")

        
            future_exog = pd.DataFrame({"GDP_Growth": gdp_monthly["GDP_Growth"].iloc[-1]}, index=future_index)
            future_pred = res_x.get_forecast(steps=future_months, exog=future_exog)
            future_mean = future_pred.predicted_mean
            future_ci = future_pred.conf_int()

            # Aggregate to quarterly for downstream energy section
            ev_forecast = future_mean.resample("QE").sum()

        
            # --- Step 8: Plot ---
            fig_sarimax = go.Figure()
            fig_sarimax.add_trace(go.Scatter(x=ev_monthly.index, y=ev_monthly, mode="lines", name="Observed"))
            fig_sarimax.add_trace(go.Scatter(x=forecast_mean_x.index, y=forecast_mean_x, mode="lines", name="Forecast (Test)", line=dict(color="orange")))
            fig_sarimax.add_trace(go.Scatter(x=future_mean.index, y=future_mean, mode="lines", name="Future Forecast", line=dict(color="green")))
            fig_sarimax.add_trace(go.Scatter(x=future_ci.index, y=future_ci.iloc[:, 0], fill=None, mode="lines", line_color="green", showlegend=False))
            fig_sarimax.add_trace(go.Scatter(x=future_ci.index, y=future_ci.iloc[:, 1], fill='tonexty', mode="lines", line_color="green", opacity=0.2, name="Confidence Interval"))
            fig_sarimax.update_layout(
                title="Hybrid SARIMAX Quarterly Forecast — GDP as Exogenous Variable",
                xaxis_title="Year", yaxis_title="EV Registrations",
                paper_bgcolor="#bef0e5", plot_bgcolor="#bef0e5", font=dict(color="#000000")
            )
            st.plotly_chart(fig_sarimax, use_container_width=True)

            category_share = {'TWO WHEELER': 0.57, 'THREE WHEELER': 0.33, 'FOUR WHEELER': 0.10}
            ev_data = {
                'TWO WHEELER': {'Daily_km': 25, 'Range_km': 141, 'Battery_kWh': 2.98},
                'THREE WHEELER': {'Daily_km': 60, 'Range_km': 129, 'Battery_kWh': 3.7},
                'FOUR WHEELER': {'Daily_km': 40, 'Range_km': 312, 'Battery_kWh': 30.2}
            }
            vehicle_lifetime_years = 10
            charging_efficiency = 0.9

            # Compute energy forecast
            all_quarters = pd.concat([ev_quarterly, ev_forecast])
            fleet_df = pd.DataFrame(index=all_quarters.index)
            for cat, share in category_share.items():
                fleet_df[f'{cat}_New'] = all_quarters * share

            for cat in category_share.keys():
                new_v = fleet_df[f'{cat}_New'].values
                active = np.zeros(len(all_quarters))
                for i in range(len(all_quarters)):
                    for j in range(i + 1):
                        age_yrs = (i - j) / 4
                        if age_yrs <= vehicle_lifetime_years:
                            survival = 1 - (age_yrs / vehicle_lifetime_years)
                            active[i] += new_v[j] * survival
                fleet_df[f'{cat}_Active'] = active

            days_in_quarter = pd.Series(all_quarters.index).diff().dt.days.fillna(91).values
            for cat in category_share.keys():
                dE = ev_data[cat]['Battery_kWh'] * ev_data[cat]['Daily_km'] / ev_data[cat]['Range_km']
                fleet_df[f'{cat}_Energy_MWh'] = (fleet_df[f'{cat}_Active'] * dE * days_in_quarter / 1000 / charging_efficiency)

            fleet_df["TotalEnergy_MWh"] = fleet_df[[f'{cat}_Energy_MWh' for cat in category_share]].sum(axis=1)
            future_energy = fleet_df.loc[fleet_df.index > ev_quarterly.index[-1]]
            future_energy.to_csv("ev_energy_forecast_2025_2027.csv")

            fig_energy = px.line(
                future_energy[[f'{cat}_Energy_MWh' for cat in category_share]].reset_index().melt(id_vars='index', var_name='Category', value_name='Energy_MWh'),
                x='index', y='Energy_MWh', color='Category',
                title="Energy Consumption Prediction (2025–2027)", markers=True,
                labels={"index": "Year", "Energy_MWh": "Energy Demand (MWh)", "Category": "Vehicle Category"}
            )
            fig_energy.update_layout(paper_bgcolor="#bef0e5", plot_bgcolor="#bef0e5", font=dict(color="#000000"))


            st.plotly_chart(fig_energy, use_container_width=True)
            st.markdown(
                """
                <div style="
                    background-color:#E6F7FF;
                    padding:10px;
                    border-radius:12px;
                    font-size:12px;
                    color:#00394F;
                    text-align:center;
                    ">
                    <b>Energy demand forecast broken down by vehicle category</b>
                </div>
                """,
                unsafe_allow_html=True
            ) 
        with col2:
            st.subheader("Model Evaluation Metrics")
            st.write("**Monthly Performance (Test period)**")
            st.write(f"• MAE = {mae_m:,.2f} • RMSE = {rmse_m:,.2f} • R² = {r2_m:.3f}")
            st.write("**Quarterly Performance (Aggregated)**")
            st.write(f"• MAE = {mae_q:,.2f} • RMSE = {rmse_q:,.2f} • R² = {r2_q:.3f}")
            st.write("")
            df = pd.read_csv("ev_energy_forecast_2025_2027.csv")
            df.to_json("output.json", orient="records", indent=4)
            
            with open("output.json") as f:
                dataset = json.load(f)
            st.subheader("Tech to Business Angle - Charging Behavior Analysis")
            with st.spinner("Analyzing..."):
                result = charging_intelligence_agent.run(f"Analyze the EV energy consumption dataset for 2025-2027: {json.dumps(dataset)} Provide insights on charging behavior, energy demand trends, and infrastructure implications.")
            st.markdown(result.content, unsafe_allow_html=False)
            st.write("")
            st.subheader("⚡ EV Energy Demand Calculation Formula")
            st.latex(r"""
            E_{\text{quarter}} = 
            \frac{N_{\text{active}} \times D_{\text{daily}} \times B_{\text{kWh}} \times \text{Days}_{\text{quarter}}}
            {R_{\text{km}} \times 1000 \times \eta}
            """)
            
            st.markdown("""
            <div style="
                background-color:#E8F0EB;
                border: 1px solid #E8F0EB;
                border-radius: 10px;
                padding: 15px;
                color:#0A0A0A;
                font-size:14px;
                line-height:1.8;">
            <b>Where:</b><br>
            • <b>E<sub>quarter</sub></b> — Energy demand in <b>MWh</b><br>
            • <b>N<sub>active</sub></b> — Active EV fleet<br>
            • <b>D<sub>daily</sub></b> — Avg daily distance (km)<br>
            • <b>B<sub>kWh</sub></b> — Battery capacity (kWh)<br>
            • <b>R<sub>km</sub></b> — Vehicle range (km)<br>
            • <b>η</b> — Charging efficiency (<b>0.9</b>)
            </div>
            """, unsafe_allow_html=True)

        # ==================== 4️⃣ INFRASTRUCTURE WITH PREDICTIONS ====================
        stations_df = pd.read_excel(stations_path)
        stations_df.columns = stations_df.columns.astype(str).str.strip()
        top5 = stations_df.sort_values(by="2024", ascending=False).head(5)

        # Linear regression predictions for each state
        prediction_years = [2025, 2026, 2027]
        top5_predictions = []

        for _, row in top5.iterrows():
            state = row['State']
            X = np.array([2022, 2023, 2024]).reshape(-1, 1)
            y = np.array([row['2022'], row['2023'], row['2024']])
            
            lr = LinearRegression()
            lr.fit(X, y)
            
            predictions = lr.predict(np.array(prediction_years).reshape(-1, 1))
            
            for year, pred in zip(prediction_years, predictions):
                top5_predictions.append({
                    'State': state,
                    'Year': str(year),
                    'Stations': max(0, int(pred))
                })

        # Combine historical and predictions
        top5_long = top5.melt(id_vars="State", value_vars=["2022","2023","2024"], 
                            var_name="Year", value_name="Stations")
        predictions_df = pd.DataFrame(top5_predictions)
        predictions_df['Type'] = 'Predicted'
        top5_long['Type'] = 'Actual'

        combined_df = pd.concat([top5_long, predictions_df], ignore_index=True)

        fig_st = px.bar(
            combined_df,
            x="State", y="Stations", color="Year", 
            pattern_shape="Type",
            barmode="group",
            title="Top 5 States — Charging Stations (Actual & Predicted)"
        )
        fig_st.update_layout(paper_bgcolor="#bef0e5", plot_bgcolor="#bef0e5", font=dict(color="#000000"))

        # ==================== 5️⃣ NATIONAL INFRASTRUCTURE PREDICTIONS ====================
        # Calculate national totals
        national_stations = stations_df[['2022', '2023', '2024']].sum()
        X_national = np.array([2022, 2023, 2024]).reshape(-1, 1)
        y_national = national_stations.values

        lr_national = LinearRegression()
        lr_national.fit(X_national, y_national)

        # Predict for future years
        future_years = np.array([2025, 2026, 2027]).reshape(-1, 1)
        national_predictions = lr_national.predict(future_years)

        # Create visualization
        years_all = [2022, 2023, 2024, 2025, 2026, 2027]
        stations_all = list(y_national) + list(national_predictions)
        types = ['Actual', 'Actual', 'Actual', 'Predicted', 'Predicted', 'Predicted']

        national_df = pd.DataFrame({
            'Year': years_all,
            'Stations': [max(0, int(s)) for s in stations_all],
            'Type': types
        })

        fig_national = px.bar(
            national_df,
            x='Year', y='Stations', color='Type',
            title="National Charging Stations (Actual & Predicted)",
            text='Stations'
        )
        fig_national.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_national.update_layout(paper_bgcolor="#bef0e5", plot_bgcolor="#bef0e5", font=dict(color="#000000"))
        
        # Side-by-side layout for infrastructure charts
        c1_infra, c2_infra = st.columns(2)
        with c1_infra:
            st.plotly_chart(fig_st, use_container_width=True)
            st.markdown(
        """
        <div style="
            background-color:#E6F7FF;
            padding:10px;
            border-radius:12px;
            font-size:12px;
            color:#00394F;
            text-align:center;
            ">
            <b>State-wise infrastructure growth and predictions</b>
        </div>
        """,
        unsafe_allow_html=True
    ) 
        
        with c2_infra:
            st.plotly_chart(fig_national, use_container_width=True)
            st.markdown(
        """
        <div style="
            background-color:#E6F7FF;
            padding:10px;
            border-radius:12px;
            font-size:12px;
            color:#00394F;
            text-align:center;
            ">
            <b>National-level charging station trends and forecasts</b>
        </div>
        """,
        unsafe_allow_html=True
    ) 


        # ==================== 6️⃣ INFRASTRUCTURE vs ENERGY DEMAND ANALYSIS ====================
        # Assumptions for analysis
        avg_charger_power_kw = 50
        chargers_per_station = 3
        utilization_rate = 0.10
        hours_per_quarter = 24 * 91

        # Calculate quarterly charging capacity from predicted stations
        quarterly_capacity = []
        for year in [2026, 2027]:
            year_idx = year - 2025
            stations = national_predictions[year_idx]
            
            capacity_mwh = (stations * chargers_per_station * avg_charger_power_kw * 
                        hours_per_quarter * utilization_rate / 1000)
            
            for q in range(3):
                quarterly_capacity.append({
                    'Quarter': f"Q{q+1} {year}",
                    'Capacity_MWh': capacity_mwh,
                    'Year': year,
                    'Quarter_Num': q+1
                })

        capacity_df = pd.DataFrame(quarterly_capacity)

        # Get predicted energy demand
        future_energy_df = fleet_df.loc[fleet_df.index > ev_quarterly.index[-1]].reset_index()
        future_energy_df = future_energy_df.head(12)
        
        future_energy_df['Year'] = future_energy_df['index'].dt.year
        future_energy_df['Quarter_Num'] = future_energy_df['index'].dt.quarter
        future_energy_df['Quarter_Period'] = future_energy_df['Year'].astype(str) + 'Q' + future_energy_df['Quarter_Num'].astype(str)
        future_energy_df['Quarter'] = future_energy_df['Quarter_Period'] 
        
        future_energy_df = future_energy_df[
            (future_energy_df['Year'] >= 2025) & (future_energy_df['Year'] <= 2027)
        ].head(12)
        
        capacity_df['Quarter_Period'] = capacity_df.apply(
            lambda x: f"{x['Year']}Q{x['Quarter_Num']}", axis=1
        )
        
        comparison_df = capacity_df.merge(
            future_energy_df[['Quarter_Period', 'TotalEnergy_MWh']], 
            on='Quarter_Period', 
            how='left'
        )
        
        comparison_df['Surplus_Deficit_MWh'] = comparison_df['Capacity_MWh'] - comparison_df['TotalEnergy_MWh']
        comparison_df['Adequacy_Ratio'] = comparison_df['Capacity_MWh'] / comparison_df['TotalEnergy_MWh']
        
        # Visualization
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=comparison_df['Quarter'], 
            y=comparison_df['Capacity_MWh'],
            name='Infrastructure Capacity',
            marker_color='lightblue'
        ))
        fig_compare.add_trace(go.Bar(
            x=comparison_df['Quarter'], 
            y=comparison_df['TotalEnergy_MWh'],
            name='Energy Demand',
            marker_color='coral'
        ))
        fig_compare.update_layout(
            title="Charging Infrastructure Capacity vs Energy Demand (2026-2027)",
            xaxis_title="Quarter",
            yaxis_title="Energy (MWh)",
            barmode='group',
            paper_bgcolor="#bef0e5", plot_bgcolor="#bef0e5", font=dict(color="#000000")
        )
        
        # Side-by-side layout for comparison chart and summary
        c1_adequacy, c2_adequacy = st.columns([2, 1])
        with c1_adequacy:
            st.plotly_chart(fig_compare, use_container_width=True)
            st.markdown(
        """
        <div style="
            background-color:#E6F7FF;
            padding:10px;
            border-radius:12px;
            font-size:12px;
            color:#00394F;
            text-align:center;
            ">
            <b>Comparison of infrastructure capacity vs predicted energy demand</b>
        </div>
        """,
        unsafe_allow_html=True
    )
        
        with c2_adequacy:
            # Summary metrics
            avg_adequacy = comparison_df['Adequacy_Ratio'].mean()
            min_adequacy = comparison_df['Adequacy_Ratio'].min()
            deficit_quarters = (comparison_df['Surplus_Deficit_MWh'] < 0).sum()
            st.write("")
            st.write("")
            st.markdown(f"""
            <div style="
                background-color:#E8F0EB;
                border: 1px solid #E8F0EB;
                border-radius: 10px;
                padding: 20px;
                color:#0A0A0A;
                font-size:15px;
                line-height:1.8;">
            <b>Adequacy Summary</b><br><br>
            • <b>Avg Ratio:</b> {avg_adequacy:.2f}x<br>
            • <b>Min Ratio:</b> {min_adequacy:.2f}x<br>
            • <b>Deficit Quarters:</b> {deficit_quarters}/{len(comparison_df)}<br><br>
            <b>Interpretation:</b><br>
            {'✅ Infrastructure is <b>adequate</b>.' if avg_adequacy >= 1.0 else '⚠️ Infrastructure may be <b>insufficient</b>.'}<br><br>
            Ratio >1.0 = sufficient<br>
            Ratio <1.0 = potential shortfall
            </div>
            """, unsafe_allow_html=True)
        
    elif st.session_state.input_type == "about":
        news_container = st.container(border=True)
         # Fetch and display news summary
        with news_container:
            col_left, col_right = st.columns(2)
            from src.model.news import fetch_news_data
            #UNCOMMENT TO RUN AGENT 
            news_summary = fetch_news_data()
            half = len(news_summary) // 2
            with col_left:
                st.header('On the Headlines')
                st.markdown(news_summary[:half])
            with col_right:
                st.markdown("-" + news_summary[half:])

        col_why, col_about = st.columns([2,1], gap="small") 
        with col_why:
            st.header('Why electric vehicles?')
            st.markdown("In 2025, the world stands at the crossroads of innovation and sustainability, and electric vehicles are at the heart of this transformation. Our project explores the evolving landscape of EV technology—where clean energy, intelligent systems, and advanced engineering converge to redefine mobility. By focusing on electric vehicles, we aim to contribute to a future that is not only more efficient and connected but also environmentally responsible, aligning technological progress with global sustainability goals.")
            st.write("")

            st.markdown("""
                <style>
                :root{
                --bg:#E6F7FF;
                --text:#00394F;
                --muted:#00394Faa;
                --radius:12px;
                --gap:12px;
                }

                .metrics-grid{
                display:grid;
                grid-template-columns: repeat(3, 1fr);
                gap:var(--gap);
                width:100%;
                }

                .metric{
                background-color: var(--bg);
                padding:10px;
                border-radius:var(--radius);
                font-size:12px;
                color:var(--text);
                text-align:center;
                width:100%;
                box-shadow:none !important;
                }

                .metric-value{
                font-size:28px;
                font-weight:700;
                color:var(--text);
                }

                .metric-label{
                font-size:13px;
                color:var(--muted);
                margin-top:3px;
                }

                .metric-foot{
                font-size:12px;
                margin-top:6px;
                color:var(--text);
                }

                .sparkline{width:100px;height:30px}
                </style>

                <div class="metrics-grid">

                <div class="metric">
                <div class="metric-value">7.02 Mn</div>
                <div class="metric-label">EVs in India</div>
                </div>

                <div class="metric">
                <div class="metric-value">29,277</div>
                <div class="metric-label">Number of Charging Stations</div>
                </div>

                <div class="metric">
                <div class="metric-value">10.0 Mn Tons</div>
                <div class="metric-label">Carbon Emissions Reduced</div>
                </div>

                </div>
                """, unsafe_allow_html=True)


            st.markdown("")
            st.markdown("Leveraging advanced online models, hybrid quantum-classical forecasts and agentic AI transforming technical output to business insights, this dashboard provides a comprehensive overview into the current EV demand through factors like sales, sentiment and charging behavior. It further approaches the business domain with a well rounded dynamic PESTEL analysis, identifying the various demand drivers of the EV domain, along with the extent of contribution of each factor.", unsafe_allow_html=True)
            st.markdown("")
        with col_about:
            st.header('About the App')
            st.markdown("This application is developed as part of our Capstone project, and has the following functionalities: ")
            st.markdown("""
            - **Sentiment Analysis and Forecasting**: Analyze public sentiment towards various EV brands and forecast future sentiment trends to help brands understand market perception.
            - **Sales Forecasting**: Predict future sales of electric vehicles based on historical sales data, market trends, and external factors.
            - **Charging Behavior and Energy Consumption Analysis**: Examine how EV users charge their vehicles and their energy consumption patterns to optimize charging infrastructure and energy management.
            - **PESTEL Analysis for EV Demand Drivers**: Evaluate the Political, Economic, Social, Technological, Environmental, and Legal factors influencing the demand for electric vehicles in India.
            - **Agentic AI**: As an intermediary for every functionality, our agents translate numbers to insights and decisions.
            """)    
            st.caption("Atharva Sreekar, Shubham Mahanti, Stuthi Shrisha | Capstone 2025")



#disclaimer and footer
st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown(
        """
        <div style="
            background-color:#E6F7FF;
            padding:10px;
            border-radius:12px;
            font-size:12px;
            color:#00394F;
            text-align:center;
            ">
            <h6><b>Disclaimer</b></h6>
            <b>This application is designed to analyse and forecast trends in the electric vehicle (EV) market using historical data and machine learning models.
            The predictions and insights provided are based on the data available up to the current date and may not account for unforeseen market changes, technological advancements, or regulatory shifts.
                </b>
        </div>
        """,
        unsafe_allow_html=True
    ) 
st.markdown(" ")
st.markdown("<p style='text-align: center; color: #000000;'>© 2025 EVolutionIndia. All rights reserved.</p>", unsafe_allow_html=True)
