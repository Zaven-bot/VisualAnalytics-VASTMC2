# --- streamlit_app.py (with Expanders) ---

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# --- Load your saved outputs ---

# These would already be produced by your earlier work
popular_locations = pd.read_csv('Problem-1/plots/popular_locations.csv')   # Total visit counts
pivot_table = pd.read_csv('Problem-1/plots/pivot_table.csv', index_col=0)   # Pivoted heatmap (location x hour)
anomaly_counts = pd.read_csv('Problem-1/plots/anomaly_counts.csv')          # Number of anomalies per location

# --- Streamlit Dashboard ---

st.title('GASTech Behavioral Analysis Dashboard')

st.markdown("""
Welcome to the GASTech Behavioral Analysis dashboard.

This dashboard summarizes patterns of employee behaviors and business visits in Abila, based on transactional and geospatial temporal data collected between January 6th and January 19th.

Expand each section below to explore different aspects of the analysis.
""")

tab1, tab3 = st.tabs(["📍 Problem 1: Business Behavior", "🚗 Problem 3: Vehicle & Card Analysis"])

with tab1:
    # --- Section 1: Most Popular Locations ---
    with st.expander("📊 Most Popular Locations (Problem 1)"):
        st.subheader('Most Popular Locations (Total Visits)')
        fig1, ax1 = plt.subplots(figsize=(14,6))
        sns.barplot(x='visit_count', y='location', data=popular_locations, palette='mako', ax=ax1)
        ax1.set_title('Most Popular Locations')
        ax1.set_xlabel('Number of Visits')
        ax1.set_ylabel('Location')
        st.pyplot(fig1)

    # --- Section 2: Visit Patterns Heatmap ---
    with st.expander("🕒 Visit Patterns by Hour (Problem 1)"):
        st.subheader('Visit Patterns by Hour (Heatmap)')
        fig2, ax2 = plt.subplots(figsize=(16,12))
        sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.0f', ax=ax2)
        ax2.set_title('Total Visit Counts by Business and Hour')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Location')
        st.pyplot(fig2)

    # --- Section 3: Anomalies Detected ---
    with st.expander("🚨 Anomalies Detected (Problem 1)"):
        st.subheader('Anomalies Detected Per Business')
        fig3, ax3 = plt.subplots(figsize=(12,6))
        ax3.barh(anomaly_counts['location'], anomaly_counts['num_anomalies'])
        ax3.set_xlabel('Number of Anomalies')
        ax3.set_ylabel('Business Location')
        ax3.set_title('Anomalies Detected Per Business')
        ax3.invert_yaxis()
        ax3.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig3)
        
        st.markdown("""---
        - Anomalies were detected using a Median + IQR filtering strategy.
        - Location-hours with fewer than 5 samples were excluded to ensure robustness.
        """)

    # --- Conclusion (Problem 1) ---
    with st.expander("📋 Summary / Conclusion (Problem 1)"):
        st.subheader('Conclusion')
        conclusion_text = """
    In this analysis, we aimed to identify patterns of typical customer traffic across various business locations and flag anomalous behavior based on deviations from those patterns.

    To robustly handle the sparsity and variability inherent in the dataset, we filtered out location-hour groups with fewer than 5 active days and used a median and IQR-based anomaly detection method rather than simple averages.

    This approach resulted in the detection of **29 anomalies** across the 14-day period.  
    The anomalies were concentrated primarily among **cafés, restaurants, and food service businesses** such as **Katerina’s Café**, **Guy’s Gyros**, and **Jack’s Magical Beans**.  
    Most irregularities occurred during **peak operating hours** (lunchtime and evening), aligning with real-world expectations about when customer traffic is most sensitive to disruptions.

    Overall, the anomaly detection results were credible and pointed toward specific businesses and times that warrant further investigation.  
    The improved method of using medians and interquartile ranges ensured that the anomalies identified reflect meaningful deviations rather than random fluctuations, especially in a sparsely populated dataset.

    Due to the lack of precise GPS coordinates for individual businesses in the provided datasets (Abila shapefiles, Kronos Island shapefiles, and MC2 Tourist Map), and in order to avoid introducing artificial or fabricated geospatial data, we chose not to construct a dynamic location-based heatmap.  
    Instead, we focused our analysis on **temporal and behavioral patterns** derived directly from the transaction records, ensuring that our results are grounded solely in the provided, verifiable data.
        """
        
        st.markdown(conclusion_text)

        # --- Add Download Button ---
        st.download_button(
            label="📥 Download Problem 1 Summary",
            data=conclusion_text,
            file_name='problem1_summary.txt',
            mime='text/plain'
        )

with tab3:
    # --- Section 3.5: Credit Card to Vehicle Match Distribution ---
    with st.expander("💳 Credit Card to Vehicle Match Distribution (Problem 3)"):
        st.subheader("Distribution of Match Ratios Across Credit Cards")

        import matplotlib.pyplot as plt
        import numpy as np

        # Load credit card match summary
        primary_vehicle_summary_df = pd.read_csv("Problem-3/plots/primary_vehicle_summary.csv")

        # Setup the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define bins (0–0.1, 0.1–0.2, ..., 0.9–1.0)
        bins = np.linspace(0, 1, 11)

        # Create histogram
        counts, bin_edges, patches = ax.hist(
            primary_vehicle_summary_df['match_ratio_to_primary'],
            bins=bins,
            edgecolor='black',
            color='skyblue',
            rwidth=0.9
        )

        # Title and Labels
        ax.set_title('Distribution of Match Ratios Across Credit Cards', fontsize=16)
        ax.set_xlabel('Match Ratio (Primary Vehicle Matches / Total Purchases)', fontsize=14)
        ax.set_ylabel('Number of Credit Cards', fontsize=14)

        # Bin center ticks
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([f'{int(b*100)}%' for b in bin_centers], fontsize=12)

        # Annotate bars
        for count, x in zip(counts, bin_centers):
            ax.text(x, count + 0.5, int(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

    # --- Section 4: Daily Car Behavior Analysis (Problem 3) ---
    with st.expander("🚗 Daily Car Behavior Analysis (Problem 3)"):
        st.subheader("Idle Status Heatmap and Car Activity Summary")

        # Load cleaned data
        idle_matrix = pd.read_csv("Problem-3/plots/cleaned_idle_matrix.csv")  # Ensure this path and file exist
        idle_matrix["date"] = pd.to_datetime(idle_matrix["date"])

        # Select date
        available_dates = sorted(idle_matrix["date"].dt.date.unique())
        # Build display labels with weekday
        date_options = [
            (date, f"{date} ({pd.to_datetime(date).strftime('%A')})") for date in available_dates
        ]
        selected_label = st.selectbox("Select Date", [label for _, label in date_options])
        selected_date = next(date for date, label in date_options if label == selected_label)

        def plot_idle_heatmap(data, selected_date):
            day_df = data[data["date"] == pd.to_datetime(selected_date)]
            heatmap_data = day_df.pivot(index="id", columns="hour", values="status").fillna(-1).sort_index()

            colorscale = [
                [0.0, "blue"],
                [0.5, "orange"],
                [1.0, "red"]
            ]

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index.astype(str),
                colorscale=colorscale,
                colorbar=dict(title="Status", tickvals=[-1, 0, 1], ticktext=["No Data", "Active", "Idle"]),
                zmin=-1,
                zmax=1
            ))

            day_name = pd.to_datetime(selected_date).strftime("%A")
            fig.update_layout(
                title=f"Idle Status Heatmap for <span style='color:orange'>{selected_date} ({day_name})</span>",
                xaxis_title="Hour of Day",
                yaxis_title="Car ID",
                height=600
            )

            st.plotly_chart(fig)

        plot_idle_heatmap(idle_matrix, selected_date)


        # --- Status Totals by Car for Selected Date ---
        day_name = pd.to_datetime(selected_date).strftime("%A")

        st.markdown(
            f"""### 📊 Total Status Hours per Car on <span style='color:orange; font-weight:bold'>{selected_date} ({day_name})</span>""",
            unsafe_allow_html=True
        )

        # Sort control
        daily_sort_option = st.selectbox(
            "Sort cars by status hours (daily):",
            options=["Unsorted", "Least to Greatest", "Greatest to Least"],
            key="daily_sort"
        )

        # Compute total hours per car for selected date
        daily_df = idle_matrix[idle_matrix["date"] == pd.to_datetime(selected_date)]
        status_summary = daily_df.groupby(["id", "status"]).size().unstack(fill_value=0).rename(columns={
            -1: "No-Data Hours",
            0: "Active Hours",
            1: "Idle Hours"
        }).reset_index()

        for col in ["No-Data Hours", "Active Hours", "Idle Hours"]:
            plot_df = status_summary.copy()

            if daily_sort_option == "Least to Greatest":
                plot_df = plot_df.sort_values(by=col, ascending=True)
            elif daily_sort_option == "Greatest to Least":
                plot_df = plot_df.sort_values(by=col, ascending=False)

            fig = px.bar(
                plot_df,
                x="id",
                y=col,
                title=f"{col} per Car ({selected_date}) — {daily_sort_option}",
                labels={"id": "Car ID", col: "Hours"},
            )
            fig.update_layout(xaxis_type="category", xaxis_title="Car ID", yaxis_title="Hours")
            st.plotly_chart(fig)


        # --- Average Status Hours Summary ---

    # --- Section 5: Average Car Behavior Analysis (Problem 3) ---
    with st.expander("🚗 Average Car Behavior Analysis (Problem 3)"):
        st.subheader("📈 Average Status Hours per Car (Over 14 Days)")

        # User selects sort order
        sort_option = st.selectbox(
            "Sort cars by average hours:",
            options=["Unsorted", "Least to Greatest", "Greatest to Least"]
        )

        # Prepare summary
        avg_summary = idle_matrix.groupby(["id", "status"]).size().unstack(fill_value=0) / idle_matrix["date"].nunique()
        avg_summary = avg_summary.rename(columns={
            -1: "Avg No-Data Hours",
            0: "Avg Active Hours",
            1: "Avg Idle Hours"
        }).reset_index()

        # Plot all three categories
        for col in ["Avg No-Data Hours", "Avg Active Hours", "Avg Idle Hours"]:
            plot_df = avg_summary.copy()

            if sort_option == "Least to Greatest":
                plot_df = plot_df.sort_values(by=col, ascending=True)
            elif sort_option == "Greatest to Least":
                plot_df = plot_df.sort_values(by=col, ascending=False)

            fig = px.bar(
                plot_df,
                x="id",
                y=col,
                title=f"{col} per Car ({sort_option})",
                labels={"id": "Car ID", col: "Average Hours"},
            )
            fig.update_layout(xaxis_type="category", xaxis_title="Car ID", yaxis_title="Average Hours")
            st.plotly_chart(fig)

    # --- Conclusion (Problem 3) ---
    with st.expander("📋 Summary / Conclusion (Problem 3)"):
        st.subheader('Conclusion')

        conclusion_text_p2 = ("""In this phase of the analysis, we focused on uncovering behavioral patterns associated with company vehicles, based on cleaned GPS records and transaction timestamps.

    By generating a heatmap of vehicle statuses (idle, active, or missing data) on an hourly basis, we were able to visualize car behavior across each day. These patterns revealed that while many vehicles were inactive or missing data for large portions of the day, certain cars showed consistent activity at regular working hours.

    To explore behavior in greater depth, we analyzed daily status totals (idle, active, no-data) per car and allowed for user-controlled sorting. This helped expose which vehicles were heavily used versus underutilized on any given date. Additionally, we created summary plots of average daily hours across all 14 days. These average patterns highlighted outliers, such as cars that were disproportionately idle or consistently had gaps in GPS coverage.

    Finally, we attempted to infer relationships between credit card usage and car presence through co-location analysis. While most cards had weak spatial-temporal overlap with specific cars (under 10% match), a few exhibited stronger ties, potentially indicating ownership. However, substantial gaps in GPS coverage and the lack of direct identifiers limited the strength of these inferences.

    Overall, the results point toward a small number of vehicles being heavily used, while many others remained largely idle. Data sparsity, particularly in GPS tracking, introduces uncertainty that must be considered when interpreting both vehicle activity and card ownership hypotheses.""")

        st.markdown(conclusion_text_p2)

        st.download_button(
            label="📥 Download Problem 3 Summary",
            data=conclusion_text_p2,
            file_name='problem2_summary.txt',
            mime='text/plain'
        )

# --- Footer Notes ---
st.markdown("""
---
© 2025 GASTech Behavioral Analysis | Project by Ian Unebasami, Stephanie Hung, Sean Cheng
""")

