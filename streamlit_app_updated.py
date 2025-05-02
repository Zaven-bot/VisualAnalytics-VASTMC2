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

    # --- Section 6: Animated Vehicle Movement Map (Problem 3, Plotly Edition) ---
    with st.expander("🗺️ Animated Vehicle Movement Map (Problem 3, Plotly Edition)"):
        import plotly.express as px
        import plotly.graph_objects as go
        import base64
        import random

        st.subheader("Vehicle Movement Over Time with Map Overlay")

        @st.cache_data
        def load_gps_data():
            df = pd.read_csv("Problem-3/plots/gps_mapped_filled.csv", parse_dates=["timestamp"])
            df["date"] = df["timestamp"].dt.date
            df["time"] = df["timestamp"].dt.strftime("%H:%M")
            return df

        df = load_gps_data()

        # --- Format date options with weekday ---
        unique_dates = sorted(df["date"].unique())
        date_display = [f"{d} ({pd.to_datetime(d).strftime('%A')})" for d in unique_dates]
        selected_label = st.selectbox("Select Date", date_display, key="plotly_date_selector")
        selected_date = str(next(d for d in unique_dates if str(d) in selected_label))

        filtered_day = df[df["date"] == pd.to_datetime(selected_date).date()]
        all_labels = sorted(filtered_day["label"].dropna().unique())
        selected_labels = st.multiselect("Select People to Display", all_labels, default=all_labels)

        plot_df = filtered_day[filtered_day["label"].isin(selected_labels)]

        # Assign consistent colors
        if "label_colors" not in st.session_state:
            st.session_state.label_colors = {}
        for label in all_labels:
            if label not in st.session_state.label_colors:
                st.session_state.label_colors[label] = f"#{random.randint(0, 0xFFFFFF):06x}"
        color_map = st.session_state.label_colors

        # Forward-fill for animation continuity
        all_times = sorted(plot_df["time"].unique())
        expanded = []
        for label in selected_labels:
            person_df = plot_df[plot_df["label"] == label].copy()
            person_df = person_df.set_index("time").reindex(all_times, method="ffill").reset_index()
            person_df["label"] = label
            expanded.append(person_df)
        full_df = pd.concat(expanded, ignore_index=True)

        # Background image
        def get_encoded_image(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        encoded_img = get_encoded_image("raw-data/MC2-tourist.jpg")
        lat_min, lat_max = df["lat"].min(), df["lat"].max()
        lon_min, lon_max = df["lon"].min(), df["lon"].max()
        bounds = [[lat_min - 0.003, lon_min + 0.0005], [lat_max + 0.004, lon_max + 0.0005]]

        # --- Speed control ---
        speed = st.selectbox("Animation Speed", ["Normal", "Fast", "Very Fast"])
        speed_ms = {"Normal": 300, "Fast": 150, "Very Fast": 50}[speed]

        fig = px.scatter(
            full_df,
            x="lon", y="lat",
            animation_frame="time",
            animation_group="label",
            color="label",
            color_discrete_map=color_map,
            hover_name="label",
            height=700,
        )

        fig.update_traces(marker=dict(size=6), showlegend=True)

        fig.update_layout(
            title=f"Vehicle Movement on {selected_date}",
            xaxis=dict(showgrid=False, range=[bounds[0][1], bounds[1][1]]),
            yaxis=dict(showgrid=False, range=[bounds[0][0], bounds[1][0]], scaleanchor="x"),
            images=[dict(
                source=f"data:image/jpg;base64,{encoded_img}",
                xref="x", yref="y",
                x=bounds[0][1], y=bounds[1][0],
                sizex=bounds[1][1] - bounds[0][1],
                sizey=bounds[1][0] - bounds[0][0],
                sizing="stretch",
                opacity=0.75,
                layer="below"
            )],
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0.1, "y": -0.1,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": speed_ms, "redraw": True},
                            "fromcurrent": True
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    }
                ]
            }],
            margin=dict(r=0, l=0, t=30, b=0),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)


    # # --- Section 6: Animated Vehicle Movement Map (Problem 3) ---
    # with st.expander("🗺️ Animated Vehicle Movement Map (Problem 3)"):
    #     import plotly.express as px
    #     import plotly.graph_objects as go
    #     import random

    #     st.subheader("Vehicle Movement Over Time")

    #     @st.cache_data
    #     def load_filled_gps():
    #         df = pd.read_csv("Problem-3/plots/gps_mapped_filled.csv", parse_dates=["timestamp"])
    #         df["time_str"] = df["timestamp"].dt.strftime("%H:%M")
    #         df["date"] = df["timestamp"].dt.date.astype(str)
    #         return df

    #     gps_filled = load_filled_gps()

    #     # Select the date
    #     selected_date = st.selectbox("Select a Date", sorted(gps_filled["date"].unique()))
    #     day_df = gps_filled[gps_filled["date"] == selected_date]

    #     # Select specific people
    #     unique_people = sorted(day_df["label"].unique())
    #     selected_people = st.multiselect("Select People to Display", unique_people, default=unique_people)

    #     # Filter data
    #     filtered_df = day_df[day_df["label"].isin(selected_people)]

    #     # Assign consistent colors per person
    #     if "color_map" not in st.session_state:
    #         st.session_state.color_map = {}
    #     for person in selected_people:
    #         if person not in st.session_state.color_map:
    #             st.session_state.color_map[person] = f"#{random.randint(0, 0xFFFFFF):06x}"
    #     color_map = st.session_state.color_map

    #     # Create animated plot
    #     fig = px.scatter_mapbox(
    #         filtered_df,
    #         lat="lat",
    #         lon="lon",
    #         color="label",
    #         animation_frame="time_str",
    #         hover_name="label",
    #         zoom=13,
    #         height=700,
    #         mapbox_style="carto-positron",
    #         color_discrete_map=color_map,
    #     )

    #     fig.update_layout(
    #         margin={"r":0,"t":0,"l":0,"b":0},
    #         legend_title="Person",
    #         mapbox=dict(center={"lat": filtered_df["lat"].mean(), "lon": filtered_df["lon"].mean()})
    #     )

    #     st.plotly_chart(fig, use_container_width=True)


    # --- Conclusion (Problem 3) ---
    with st.expander("📋 Summary / Conclusion (Problem 3)"):
        st.subheader('Conclusion')

        conclusion_text_p2 = ("""
            We conducted an analysis to determine whether the owners of credit cards and loyalty cards could be inferred from the available data. Based on spatial-temporal correlation between credit card transaction timestamps and GPS data from company vehicles, we found that ownership inference is only plausible for a small subset of credit cards, and not feasible for loyalty cards. Specifically, only 2 out of 55 credit cards demonstrated a strong co-location pattern—where a high percentage of transactions consistently aligned with the presence of a particular vehicle. These few instances suggest possible ownership. However, the remaining cards showed weak or inconsistent overlap, making confident identification unlikely.

            Our evidence is drawn from the calculation of a “match ratio” for each card: the proportion of purchases where a vehicle was detected nearby (within a 1-minute window). For each card, we identified its most frequently matched vehicle and computed this ratio. A histogram of these match ratios reveals a heavily right-skewed distribution, confirming that most cards lack consistent spatial-temporal overlap with any single vehicle.

            There are multiple uncertainties in our method. The 1-minute matching window may be too narrow or too broad depending on the logging intervals. Furthermore, proximity alone does not imply ownership—a nearby vehicle does not guarantee the occupant made the transaction. Shared vehicles or cards (e.g., carpooling or team purchases) further complicate interpretation. The assumption that each card belongs to a unique person may not hold true in all cases.

            There are also significant uncertainties in the data itself. Most vehicles had 17–19 hours per day with no GPS signal, creating large blind spots in possible matches. The absence of precise business coordinates made spatial comparisons approximate. Additionally, loyalty card swipes were relatively sparse and irregular, making them even more difficult to connect meaningfully with individuals. As a result, while there are limited cases where credit card ownership may be reasonably inferred, the majority of records are too ambiguous to draw definitive conclusions.
            """
        )

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

