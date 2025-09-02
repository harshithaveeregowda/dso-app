import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import base64

logo_path = "images/cbs_logo_with_background.png"

# --- Page Configuration ---
st.set_page_config(
    page_title="DSO Prediction and Simulation",
    page_icon=logo_path,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display a logo in the sidebar
st.logo(logo_path, icon_image=logo_path)

# Path to logo
base_path = Path(__file__).parent
logo_path = base_path / "images" / "cbs_logo_with_background.png"

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if logo_path.exists():
    # Convert logo to base64
    img_base64 = image_to_base64(logo_path)

else:
    st.write("Logo not found!")

# Define an orange shades palette
orange_shades = [
    "#F07C00",  # A - strong orange
    "#F4B619",  # B - golden yellow
    "#1C8C7C",  # C - teal green
    "#06485D",  # D - deep blue
    "#FF9E1C",  # lighter orange
    "#FFD447",  # lighter yellow
    "#3EB89F",  # lighter teal
    "#0A6A83",  # lighter blue
    "#CC6200",  # darker orange
    "#BFA30F",  # darker yellow
]

# ----------------------------------------------------------------------------------------------------
data_file = base_path / "data" / "dso_showcase_dataset.csv"

# Load Excel file
if data_file.exists():
    df = pd.read_csv(data_file)
    # Clean column names
    df.columns = [col.replace('.', '').replace('_', ' ').strip() for col in df.columns]
    st.session_state.data = df

    # Apply styling via CSS
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            color: #F07C00; /* Orange */
        }

        /* Multiselect dropdown background in grey gradient */
        .stMultiSelect [data-baseweb="select"] > div {
            background: linear-gradient(135deg, #E0E0E0, #A9A9A9);
            border-radius: 5px;
            padding: 5px;
        }

        /* Selected items highlighted in orange */
        .stMultiSelect [data-baseweb="select"] span {
            background-color: #FF8C42 !important; /* orange */
            color: #000000 !important;
            border-radius: 3px;
        }

        /* Sidebar label font color */
        .css-1d391kg p {
            color: #333333;
            font-weight: bold;
        }
        
        /* Change the slider track (background) */
        .css-14xtw13 .stSlider > div > div > div > div {
            background: #F07C00 !important;
        }

        /* Change the slider handle */
        .css-14xtw13 .stSlider > div > div > div > div > div {
            background: #F07C00 !important;
            border: 2px solid #F07C00 !important;
        }

    
        /* slider active (filled) part */
        div[data-baseweb="slider"] div[data-testid="stThumbValue"] ~ div > div {
            background: orange !important;  /* active section becomes orange */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # Clear the session state if no file is uploaded
    st.session_state.data = None
    st.session_state.model_trained = False
    # ----------------------------------------------------------------------------------------------------

########################################################################################################

# --- Helper Functions ---
def get_performance_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of performance metrics."""
    metrics = {
        "R-squared": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }
    return metrics

def train_xgb_model(X_train, y_train, params):
    """Trains an XGBoost model and returns it."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
    
def get_clustered_ses(df_train, features, target, cluster_col):
    """Fits an OLS model and returns results with clustered standard errors."""
    try:
        # Add a constant for the intercept
        X = sm.add_constant(df_train[features])
        y = df_train[target]
        
        # Fit the model
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_train[cluster_col]})
        return model.summary()
    except Exception as e:
        return f"Could not compute clustered standard errors: {e}"


# --- App State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'lin_model' not in st.session_state:
    st.session_state.lin_model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = []


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")




    # Only show the rest of the sidebar if data is loaded
    if st.session_state.data is not None:
        df = st.session_state.data
        all_cols = df.columns.tolist()

        st.subheader("Variable Selection")
        target_variable = st.selectbox("Select Target Variable (DSO)", all_cols, index=len(all_cols)-1 if all_cols else 0)
        
        default_features = [
            'Payment Terms Days', 'Invoice Error Rate', 'Forecast Accuracy',
            'Contract Extension Days', 'Avg Days Late Last3 Days'
        ]
        
        available_features = [col for col in all_cols if col != target_variable and col != 'Customer ID']
        
        valid_default_features = [f for f in default_features if f in available_features]

        features = st.multiselect("Select Feature Variables", available_features, default=valid_default_features)
        st.session_state.features = features


        st.subheader("Model Parameters")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        
        with st.expander("XGBoost Hyperparameters"):
            n_estimators = st.slider("Number of Estimators (n_estimators)", 50, 500, 100, 10)
            max_depth = st.slider("Max Depth", 3, 15, 5, 1)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        
        xgb_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}

        if st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Training models... This may take a moment."):
                if len(features) > 0:
                    df_clean = df.dropna(subset=features + [target_variable])
                    X = df_clean[features]
                    y = df_clean[target_variable]
                    
                    df_train_ols = df_clean.loc[X.index]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    st.session_state.X_train_df = X_train
                    st.session_state.df_train_ols = df_train_ols.loc[X_train.index]

                    st.session_state.xgb_model = train_xgb_model(X_train, y_train, xgb_params)

                    lin_model = LinearRegression()
                    lin_model.fit(X_train, y_train)
                    st.session_state.lin_model = lin_model

                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.model_trained = True
                else:
                    st.warning("Please select at least one feature variable.")
            st.success("Models trained successfully!")

# --- Main Page ---
st.title("💼 Days Sales Outstanding (DSO) Analysis")
st.markdown("An interactive tool to predict DSO, understand its key drivers, and simulate the impact of business decisions.")

if st.session_state.data is None:
    st.info("👋 Welcome! To begin, please upload your CSV data using the sidebar.")
    st.markdown("""
    **Getting Started:**
    1.  Click on the `>` arrow in the top-left corner to open the sidebar.
    2.  Click 'Browse files' to upload your DSO data in CSV format.
    
    Your data should ideally contain columns for:
    - A unique identifier for clustering (like 'Customer ID')
    - The target variable to predict (like 'DSO actual Days')
    - Feature variables that might influence the target (like 'Payment Terms Days', 'Invoice Error Rate', etc.)
    """)
else:
    df = st.session_state.data
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Model Performance", "🧠 Prediction Explanations", "🔮 DSO Simulation"])

    with tab1:
        st.header("Exp Data Analysis")
        st.markdown("A first look at your data.")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())

        #st.subheader("Summary Statistics")
        #st.dataframe(df.describe())

        st.subheader("Summary Statistics")

        # Pick only numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # User selects which columns/features to include
        selected_cols = st.multiselect(
            "Select numeric columns to plot:",
            options=numeric_cols,
            default=numeric_cols[:5]  # first 5 as default
        )

        if selected_cols:
            # Extract descriptive stats for selected columns
            desc_df = df[selected_cols].describe().loc[["mean", "std", "25%", "75%"]].T

            # Reset index for Plotly
            desc_df.reset_index(inplace=True)
            desc_df = desc_df.rename(columns={"index": "Feature"})

            # Melt for long format
            desc_melted = desc_df.melt(
                id_vars="Feature",
                value_vars=["mean", "std", "25%", "75%"],
                var_name="Statistic",
                value_name="Value"
            )

            # Auto-scale figure height: 60px per feature + padding
            fig_height = 60 * len(selected_cols) + 300

            # Plot grouped bar chart
            fig = px.bar(
                desc_melted,
                x="Feature",
                y="Value",
                color="Statistic",
                barmode="group",
                text="Value",
                #title="Selected Summary Statistics",
                height=fig_height,
                width=1000,
                color_discrete_sequence=orange_shades
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor="white",
                margin=dict(l=40, r=40, t=60, b=150)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👆 Please select at least one column to display statistics.")

        if len(st.session_state.features) > 0:
            st.subheader("Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Feature Distributions")
                feature_to_plot = st.selectbox("Select a feature to see its distribution", st.session_state.features)
                fig_hist = px.histogram(df, x=feature_to_plot, marginal="box", title=f"Distribution of {feature_to_plot}", color_discrete_sequence=orange_shades)
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                st.markdown(f"#### Relationship with DSO")
                fig_scatter = px.scatter(df, x=feature_to_plot, y=target_variable, trendline="ols", title=f"{feature_to_plot} vs. {target_variable}", color_discrete_sequence=orange_shades)
                st.plotly_chart(fig_scatter, use_container_width=True)


            st.markdown("#### Correlation Heatmap")
            corr_df = df[st.session_state.features + [target_variable]]
            corr_matrix = corr_df.corr()
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=orange_shades,
                colorbar=dict(title='Correlation')
            ))
            fig_heatmap.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)


    with tab2:
        st.header("Model Performance")
        if not st.session_state.model_trained:
            st.info("Train the models in the sidebar to see performance metrics.")
        else:
            y_pred_xgb = st.session_state.xgb_model.predict(st.session_state.X_test)
            y_pred_lin = st.session_state.lin_model.predict(st.session_state.X_test)
            metrics_xgb = get_performance_metrics(st.session_state.y_test, y_pred_xgb)
            metrics_lin = get_performance_metrics(st.session_state.y_test, y_pred_lin)
            
            st.subheader("Performance on Test Set")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🌳 XGBoost Model")
                for name, val in metrics_xgb.items():
                    st.metric(label=name, value=f"{val:.3f}")
            with col2:
                st.markdown("#### 📈 Linear Regression")
                for name, val in metrics_lin.items():
                    st.metric(label=name, value=f"{val:.3f}")
            
            st.subheader("Feature Importance (XGBoost)")
            importance = pd.DataFrame({
                'feature': st.session_state.features,
                'importance': st.session_state.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            fig_importance = px.bar(importance, x='importance', y='feature', orientation='h', title="Feature Importance", color_discrete_sequence=orange_shades)
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.subheader("Linear Regression Coefficients")
            st.markdown("Coefficients from a standard OLS model. For robust inference, especially with grouped data (like customers), using clustered standard errors is recommended.")
            cluster_col = st.selectbox("Select a column for clustering standard errors (e.g., Customer ID)", df.columns)
            if cluster_col:
                with st.spinner("Calculating Clustered Standard Errors..."):
                    summary = get_clustered_ses(st.session_state.df_train_ols, st.session_state.features, target_variable, cluster_col)
                    st.text(summary)


    with tab3:
        st.header("Prediction Explanations (SHAP)")
        if not st.session_state.model_trained:
            st.info("Train the models in the sidebar to generate explanations.")
        else:
            explainer = shap.Explainer(st.session_state.xgb_model)
            shap_values = explainer(st.session_state.X_test)
            
            st.subheader("Global Feature Impact")
            st.markdown("The SHAP summary plot shows the impact of each feature on the model's output. Each point is a single observation. Red means a high feature value, blue means low.")
            # Create figure
            fig_summary = plt.figure()

            # Custom orange colormap (light → dark orange)
            orange_cmap = plt.cm.Oranges

            # Pass colormap to SHAP summary_plot
            shap.summary_plot(
                shap_values,
                st.session_state.X_test,
                show=False,
                cmap=orange_cmap  # <-- use orange colormap
            )

            # Display in Streamlit
            st.pyplot(fig_summary)
            plt.close(fig_summary)
            
            st.subheader("Individual Prediction Breakdown")
            st.markdown("Select a single observation from the test set to see how the model arrived at its prediction.")
            observation_index = st.slider("Select an observation index", 0, len(st.session_state.X_test)-1, 0, 1)
            
            st.markdown(f"**Explaining Observation {observation_index}**")

            force_plot_fig = shap.force_plot(
                explainer.expected_value,
                shap_values.values[observation_index,:],
                st.session_state.X_test.iloc[observation_index,:],
                show=False,
                matplotlib=True
            )
            st.pyplot(force_plot_fig, bbox_inches='tight')
            plt.close(force_plot_fig)

            actual_val = st.session_state.y_test.iloc[observation_index]
            predicted_val = st.session_state.xgb_model.predict(st.session_state.X_test.iloc[[observation_index]])[0]
            col1, col2 = st.columns(2)
            col1.metric("Actual DSO", f"{actual_val:.2f} days")
            col2.metric("Predicted DSO", f"{predicted_val:.2f} days")
            with st.expander("View feature values for this observation"):
                st.dataframe(st.session_state.X_test.iloc[[observation_index]])


    with tab4:
        st.header("DSO Simulation Tool")
        if not st.session_state.model_trained:
            st.info("Train the models in the sidebar to run simulations.")
        else:
            st.markdown("Use the sliders to simulate changes to business drivers and see the potential impact on the average predicted DSO for the test set.")
            # Original test data
            X_original = st.session_state.X_test.copy()

            # Simulator-1 works on its own copy
            X_sim1 = X_original.copy()

            # Simulator-2 works on its own copy
            X_sim2 = X_original.copy()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Simulator-1")

                simulation_cols_1 = st.columns(3)
                col_idx = 0
                for feature in st.session_state.features:
                    with simulation_cols_1[col_idx % 3]:
                        min_val_1 = X_sim1[feature].min()
                        max_val_1 = X_sim1[feature].max()
                        mean_val_1 = X_sim1[feature].mean()

                        if min_val_1 < 1 and max_val_1 <= 1:
                            change_1 = st.slider(f"Change {feature} (pp)", -0.25, 0.25, 0.0, 0.01, key=f"sim_{feature}")
                            X_sim1[feature] += change_1
                            X_sim1[feature] = np.clip(X_sim1[feature], 0, 1)
                        else:
                            change_1 = st.slider(f"Change {feature} (days)", -int(mean_val_1), int(mean_val_1), 0, 1, key=f"sim_{feature}")
                            X_sim1[feature] += change_1
                            X_sim1[feature] = np.clip(X_sim1[feature], 0, None)
                    col_idx += 1

                st.subheader("Simulator-1 Results")

                pred_before_1 = st.session_state.xgb_model.predict(st.session_state.X_test)
                pred_after_1 = st.session_state.xgb_model.predict(X_sim1)
                avg_before_1 = np.mean(pred_before_1)
                avg_after_1 = np.mean(pred_after_1)
                avg_impact_1 = avg_after_1 - avg_before_1

                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Original Avg. Predicted DSO", f"{avg_before_1:.2f} days")
                res_col2.metric("Simulated Avg. Predicted DSO", f"{avg_after_1:.2f} days", delta=f"{avg_impact_1:.2f} days")

                if abs(avg_impact_1) > 0.01:
                    if avg_impact_1 < 0:
                        st.success(f"**This combination of changes could reduce the average DSO by {abs(avg_impact_1):.2f} days.**")
                    else:
                        st.warning(f"**This combination of changes could increase the average DSO by {avg_impact_1:.2f} days.**")
                else:
                    st.info("No significant change in average DSO with current simulation settings.")

            with col2:

                st.subheader("Simulator-2")

                simulation_cols_2 = st.columns(3)
                col_idx = 0
                for feature in st.session_state.features:
                    with simulation_cols_2[col_idx % 3]:
                        min_val = X_sim2[feature].min()
                        max_val = X_sim2[feature].max()
                        mean_val = X_sim2[feature].mean()

                        if min_val < 1 and max_val <= 1:
                            change = st.slider(f"Change {feature} (pp)", -0.25, 0.25, 0.0, 0.01, key=f"sim_{feature}_2")
                            X_sim2[feature] += change
                            X_sim2[feature] = np.clip(X_sim2[feature], 0, 1)
                        else:
                            change = st.slider(f"Change {feature} (days)", -int(mean_val), int(mean_val), 0, 1, key=f"sim_{feature}_2")
                            X_sim2[feature] += change
                            X_sim2[feature] = np.clip(X_sim2[feature], 0, None)
                    col_idx += 1

                st.subheader("Simulator-2 Results")

                pred_before = st.session_state.xgb_model.predict(st.session_state.X_test)
                pred_after = st.session_state.xgb_model.predict(X_sim2)
                avg_before = np.mean(pred_before)
                avg_after = np.mean(pred_after)
                avg_impact = avg_after - avg_before

                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Original Avg. Predicted DSO", f"{avg_before:.2f} days")
                res_col2.metric("Simulated Avg. Predicted DSO", f"{avg_after:.2f} days", delta=f"{avg_impact:.2f} days")

                if abs(avg_impact) > 0.01:
                    if avg_impact < 0:
                        st.success(f"**This combination of changes could reduce the average DSO by {abs(avg_impact):.2f} days.**")
                    else:
                        st.warning(f"**This combination of changes could increase the average DSO by {avg_impact:.2f} days.**")
                else:
                    st.info("No significant change in average DSO with current simulation settings.")

            # Original X_test
            X_orig = st.session_state.X_test.copy()
            features = st.session_state.features
            model = st.session_state.xgb_model

            # Slider bounds
            bounds_dict = {
                "Avg Days Late Last3 Days": (-5, 5),
                "Contract Extension Days": (-3, 3),
                "Forecast Accuracy": (-0.25, 0.25),
                "Invoice Error Rate": (-0.25, 0.25),
                "Payment Terms Days": (-53, 53)
            }

            # Feature type / precision mapping
            feature_precision = {
                "Avg Days Late Last3 Days": 0,  # int
                "Contract Extension Days": 0,  # int
                "Forecast Accuracy": 2,  # 2 decimals
                "Invoice Error Rate": 2,  # 2 decimals
                "Payment Terms Days": 0  # int
            }

            # Target DSO
            target_dso = st.number_input("Enter target DSO", value=104.1, step=0.1)

            # Monte Carlo simulation
            results = []
            for _ in range(5000):
                X_sim = X_orig.copy()
                for f in features:
                    low, high = bounds_dict[f]

                    if feature_precision[f] == 0:
                        # strictly integer increment
                        delta = np.random.randint(np.floor(low), np.ceil(high) + 1)
                    else:
                        delta = np.random.uniform(low, high)

                    new_val = X_sim[f] + delta

                    # Clip values for small-range floats
                    if low < 1 and high <= 1:
                        new_val = np.clip(new_val, 0, 1)
                    else:
                        new_val = np.clip(new_val, 0, None)

                    # Round to correct precision
                    X_sim[f] = np.round(new_val, feature_precision[f])

                # Predict DSO
                pred = model.predict(X_sim)
                avg_pred = np.mean(pred)

                if np.isclose(avg_pred, target_dso, atol=0.5):
                    row = {}
                    for f in features:
                        mean_val = X_sim[f].mean()
                        # round mean to the right precision
                        row[f] = int(mean_val) if feature_precision[f] == 0 else round(mean_val, 2)
                    row["Predicted DSO"] = round(avg_pred, 2)
                    results.append(row)

            # Display results
            results_df = pd.DataFrame(results)
            st.write("Possible slider combinations for target DSO:")
            #st.dataframe(results_df)

            df_filtered = results_df[
                (results_df["Payment Terms Days"] >= -53) & (results_df["Payment Terms Days"] <= 53) &
                (results_df["Invoice Error Rate"] >= -0.25) & (results_df["Invoice Error Rate"] <= 0.25)
                ]
            st.dataframe(df_filtered)

            # Original X_test
            X_orig = st.session_state.X_test.copy()
            features = st.session_state.features
            model = st.session_state.xgb_model

            # Slider bounds (same as Simulator-1)
            bounds_dict = {
                "Avg Days Late Last3 Days": (-5, 5),
                "Contract Extension Days": (-3, 3),
                "Forecast Accuracy": (-0.25, 0.25),
                "Invoice Error Rate": (-0.25, 0.25),
                "Payment Terms Days": (-53, 53)
            }

            # Target DSO
            target_dso = st.number_input("Enter DSO", value=104.1, step=0.1)

            # Monte Carlo simulation
            results = []
            for _ in range(5000):  # generate 5000 random combinations
                X_sim = X_orig.copy()
                for f in features:
                    low, high = bounds_dict[f]
                    X_sim[f] += np.random.uniform(low, high)
                    # Apply clipping same as Simulator-1
                    if low < 1 and high <= 1:
                        X_sim[f] = np.clip(X_sim[f], 0, 1)
                    else:
                        X_sim[f] = np.clip(X_sim[f], 0, None)

                pred = model.predict(X_sim)
                avg_pred = np.mean(pred)

                if np.isclose(avg_pred, target_dso, atol=0.5):  # keep combinations within ±0.5 day
                    row = {f: X_sim[f].mean() for f in features}
                    row["Predicted DSO"] = avg_pred
                    results.append(row)

            # Display results
            results_df = pd.DataFrame(results)
            st.write("Possible slider combinations for target DSO:")
            st.dataframe(results_df)

#---------------------------------------------Begin: cbs Footer---------------------------------------------------------
# Footer injection
footer_logo_path = "images/cbs_footer_logo.png"

# Encode the image to base64
with open(footer_logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

# Footer HTML
footer_html = f"""
<div style='position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #180501;
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 999;'>
    <img src="data:image/png;base64,{logo_base64}" style="height:40px; margin-right:15px;">
    <span style="color:white; font-size:14px;">DSO Analysis App</span>
</div>
"""

# Inject footer
st.markdown(footer_html, unsafe_allow_html=True)
