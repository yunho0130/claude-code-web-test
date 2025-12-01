"""
Boston House Price Prediction Dashboard with PandasAI

This dashboard provides:
- Data exploration with PandasAI natural language queries
- Interactive visualizations
- Machine learning-based price predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_boston_data():
    """Load Boston Housing dataset."""
    # Boston Housing dataset features and target
    # Original dataset from UCI ML Repository
    data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

    try:
        df = pd.read_csv(data_url)
        # Rename 'medv' to 'PRICE' for clarity
        if 'medv' in df.columns:
            df = df.rename(columns={'medv': 'PRICE'})
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create synthetic Boston-like data as fallback
        return create_synthetic_boston_data()


def create_synthetic_boston_data():
    """Create synthetic Boston Housing-like data as fallback."""
    np.random.seed(42)
    n_samples = 506

    df = pd.DataFrame({
        'crim': np.random.exponential(3.5, n_samples),
        'zn': np.random.uniform(0, 100, n_samples),
        'indus': np.random.uniform(0, 28, n_samples),
        'chas': np.random.binomial(1, 0.07, n_samples),
        'nox': np.random.uniform(0.3, 0.9, n_samples),
        'rm': np.random.normal(6.3, 0.7, n_samples),
        'age': np.random.uniform(2, 100, n_samples),
        'dis': np.random.uniform(1, 12, n_samples),
        'rad': np.random.randint(1, 25, n_samples),
        'tax': np.random.uniform(180, 720, n_samples),
        'ptratio': np.random.uniform(12, 22, n_samples),
        'b': np.random.uniform(0, 400, n_samples),
        'lstat': np.random.uniform(1, 38, n_samples),
    })

    # Generate realistic prices based on features
    df['PRICE'] = (
        22 +
        df['rm'] * 3.5 -
        df['lstat'] * 0.5 -
        df['crim'] * 0.2 +
        df['chas'] * 2 +
        np.random.normal(0, 2, n_samples)
    ).clip(5, 50)

    return df


# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'crim': 'Per capita crime rate by town',
    'zn': 'Proportion of residential land zoned for lots over 25,000 sq.ft.',
    'indus': 'Proportion of non-retail business acres per town',
    'chas': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
    'nox': 'Nitric oxides concentration (parts per 10 million)',
    'rm': 'Average number of rooms per dwelling',
    'age': 'Proportion of owner-occupied units built prior to 1940',
    'dis': 'Weighted distances to five Boston employment centres',
    'rad': 'Index of accessibility to radial highways',
    'tax': 'Full-value property-tax rate per $10,000',
    'ptratio': 'Pupil-teacher ratio by town',
    'b': '1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town',
    'lstat': '% lower status of the population',
    'PRICE': 'Median value of owner-occupied homes in $1000s'
}


def init_pandasai():
    """Initialize PandasAI if API key is provided."""
    try:
        from pandasai import SmartDataframe
        from pandasai.llm import OpenAI

        api_key = st.session_state.get('openai_api_key', '')
        if api_key:
            llm = OpenAI(api_token=api_key)
            return llm
        return None
    except ImportError:
        return None


def train_model(X_train, X_test, y_train, y_test, model_type):
    """Train and evaluate a model."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    model = models[model_type]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'R² Score': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }

    return model, y_pred, metrics


def main():
    # Header
    st.markdown('<p class="main-header">🏠 Boston House Price Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Dashboard with PandasAI Integration</p>', unsafe_allow_html=True)

    # Load data
    df = load_boston_data()

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # PandasAI API Key input
    st.sidebar.subheader("🤖 PandasAI Settings")
    api_key = st.sidebar.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help="Enter your OpenAI API key to enable natural language queries with PandasAI"
    )
    if api_key:
        st.session_state['openai_api_key'] = api_key

    # Navigation
    st.sidebar.subheader("📍 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Data Overview", "📈 Visualizations", "🤖 PandasAI Query", "🔮 Price Prediction"]
    )

    # Main content based on selected page
    if page == "📊 Data Overview":
        show_data_overview(df)
    elif page == "📈 Visualizations":
        show_visualizations(df)
    elif page == "🤖 PandasAI Query":
        show_pandasai_query(df)
    elif page == "🔮 Price Prediction":
        show_prediction(df)


def show_data_overview(df):
    """Display data overview page."""
    st.header("📊 Data Overview")

    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Avg Price", f"${df['PRICE'].mean()*1000:,.0f}")
    with col4:
        st.metric("Price Range", f"${df['PRICE'].min()*1000:,.0f} - ${df['PRICE'].max()*1000:,.0f}")

    st.divider()

    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Feature descriptions
    st.subheader("📝 Feature Descriptions")
    desc_df = pd.DataFrame([
        {"Feature": k, "Description": v}
        for k, v in FEATURE_DESCRIPTIONS.items()
    ])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)


def show_visualizations(df):
    """Display visualizations page."""
    st.header("📈 Data Visualizations")

    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Price Distribution", "Correlation Heatmap", "Feature vs Price", "Scatter Matrix"]
    )

    if viz_type == "Price Distribution":
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x='PRICE', nbins=30,
                title='Distribution of House Prices',
                labels={'PRICE': 'Price ($1000s)'},
                color_discrete_sequence=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df, y='PRICE',
                title='Box Plot of House Prices',
                labels={'PRICE': 'Price ($1000s)'},
                color_discrete_sequence=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation = df.corr()
        sns.heatmap(
            correlation, annot=True, cmap='RdBu_r', center=0,
            fmt='.2f', square=True, ax=ax
        )
        plt.title('Feature Correlation Heatmap', fontsize=14)
        st.pyplot(fig)

        # Top correlations with price
        st.subheader("🎯 Top Correlations with Price")
        price_corr = correlation['PRICE'].drop('PRICE').sort_values(key=abs, ascending=False)

        fig = px.bar(
            x=price_corr.values,
            y=price_corr.index,
            orientation='h',
            title='Feature Correlations with Price',
            labels={'x': 'Correlation', 'y': 'Feature'},
            color=price_corr.values,
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Feature vs Price":
        feature = st.selectbox(
            "Select Feature",
            [col for col in df.columns if col != 'PRICE']
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df, x=feature, y='PRICE',
                title=f'{feature} vs Price',
                labels={'PRICE': 'Price ($1000s)'},
                trendline='ols',
                color_discrete_sequence=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df, x=feature, nbins=30,
                title=f'Distribution of {feature}',
                color_discrete_sequence=['#43A047']
            )
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Scatter Matrix":
        selected_features = st.multiselect(
            "Select Features (max 5)",
            [col for col in df.columns if col != 'PRICE'],
            default=['rm', 'lstat', 'ptratio']
        )

        if selected_features:
            if len(selected_features) > 5:
                st.warning("Please select at most 5 features for better visualization.")
            else:
                features_with_price = selected_features + ['PRICE']
                fig = px.scatter_matrix(
                    df[features_with_price],
                    dimensions=features_with_price,
                    color='PRICE',
                    title='Scatter Matrix',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)


def show_pandasai_query(df):
    """Display PandasAI query page."""
    st.header("🤖 PandasAI Natural Language Query")

    st.info("""
    **How to use PandasAI:**
    1. Enter your OpenAI API key in the sidebar
    2. Type your question about the data in natural language
    3. PandasAI will analyze the data and provide insights

    **Example questions:**
    - What is the average house price?
    - Which features have the strongest correlation with price?
    - Show houses with more than 7 rooms
    - What is the price distribution for houses near the Charles River?
    """)

    api_key = st.session_state.get('openai_api_key', '')

    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use PandasAI.")

        # Show sample queries without API
        st.subheader("📝 Sample Data Queries (No API Required)")

        query_type = st.selectbox(
            "Select a pre-built query",
            [
                "Average price by room count",
                "High-value properties (Price > $30k)",
                "Properties near Charles River",
                "Low crime rate areas",
                "Statistical summary by price range"
            ]
        )

        if st.button("Run Query"):
            if query_type == "Average price by room count":
                df['rm_category'] = pd.cut(df['rm'], bins=[0, 5, 6, 7, 10], labels=['<5', '5-6', '6-7', '>7'])
                result = df.groupby('rm_category')['PRICE'].agg(['mean', 'count', 'std']).round(2)
                st.dataframe(result, use_container_width=True)

            elif query_type == "High-value properties (Price > $30k)":
                result = df[df['PRICE'] > 30][['rm', 'lstat', 'ptratio', 'crim', 'PRICE']].sort_values('PRICE', ascending=False)
                st.dataframe(result, use_container_width=True)
                st.write(f"Found {len(result)} high-value properties")

            elif query_type == "Properties near Charles River":
                river_props = df[df['chas'] == 1]
                non_river = df[df['chas'] == 0]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("River Properties", len(river_props))
                    st.metric("Avg Price (River)", f"${river_props['PRICE'].mean()*1000:,.0f}")
                with col2:
                    st.metric("Non-River Properties", len(non_river))
                    st.metric("Avg Price (Non-River)", f"${non_river['PRICE'].mean()*1000:,.0f}")

            elif query_type == "Low crime rate areas":
                low_crime = df[df['crim'] < df['crim'].quantile(0.25)]
                result = low_crime[['crim', 'rm', 'PRICE']].sort_values('crim')
                st.dataframe(result.head(20), use_container_width=True)
                st.write(f"Found {len(low_crime)} properties in low crime areas")

            elif query_type == "Statistical summary by price range":
                df['price_range'] = pd.cut(df['PRICE'], bins=[0, 15, 25, 35, 50], labels=['Low', 'Medium', 'High', 'Premium'])
                result = df.groupby('price_range').agg({
                    'rm': 'mean',
                    'lstat': 'mean',
                    'crim': 'mean',
                    'PRICE': ['count', 'mean']
                }).round(2)
                st.dataframe(result, use_container_width=True)
    else:
        # PandasAI with API key
        try:
            from pandasai import SmartDataframe
            from pandasai.llm import OpenAI

            query = st.text_area(
                "Enter your question about the data:",
                placeholder="e.g., What is the average house price for properties with more than 6 rooms?"
            )

            if st.button("Ask PandasAI"):
                if query:
                    with st.spinner("Analyzing data..."):
                        try:
                            llm = OpenAI(api_token=api_key)
                            sdf = SmartDataframe(df, config={"llm": llm})
                            response = sdf.chat(query)

                            st.subheader("📊 Result")
                            if isinstance(response, pd.DataFrame):
                                st.dataframe(response, use_container_width=True)
                            elif isinstance(response, (plt.Figure, plt.Axes)):
                                st.pyplot(response)
                            else:
                                st.write(response)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a question.")

        except ImportError:
            st.error("PandasAI is not installed. Please run: pip install pandasai")


def show_prediction(df):
    """Display prediction page."""
    st.header("🔮 House Price Prediction")

    # Model selection and training
    st.subheader("🎛️ Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Gradient Boosting", "Linear Regression"]
        )

        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100

    with col2:
        selected_features = st.multiselect(
            "Select Features for Training",
            [col for col in df.columns if col != 'PRICE'],
            default=['rm', 'lstat', 'ptratio', 'crim', 'nox', 'dis']
        )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for training.")
        return

    # Prepare data
    X = df[selected_features]
    y = df['PRICE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            model, y_pred, metrics = train_model(X_train_scaled, X_test_scaled, y_train, y_test, model_type)

            # Store model in session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['features'] = selected_features

            # Display metrics
            st.subheader("📊 Model Performance")

            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{metrics['R² Score']:.4f}")
            col2.metric("RMSE", f"${metrics['RMSE']*1000:,.0f}")
            col3.metric("MAE", f"${metrics['MAE']*1000:,.0f}")

            # Prediction vs Actual plot
            st.subheader("📈 Prediction vs Actual")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test.values, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='#1E88E5', opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title='Predicted vs Actual Prices',
                xaxis_title='Actual Price ($1000s)',
                yaxis_title='Predicted Price ($1000s)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance (for tree-based models)
            if model_type in ['Random Forest', 'Gradient Boosting']:
                st.subheader("🎯 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)

                fig = px.bar(
                    importance_df, x='Importance', y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)

    # Custom prediction
    st.divider()
    st.subheader("🏠 Predict Custom House Price")

    if 'model' not in st.session_state:
        st.info("Please train a model first to make predictions.")
    else:
        st.write("Enter values for each feature:")

        feature_values = {}
        cols = st.columns(3)

        for i, feature in enumerate(st.session_state['features']):
            with cols[i % 3]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())

                feature_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=FEATURE_DESCRIPTIONS.get(feature, '')
                )

        if st.button("💰 Predict Price"):
            input_data = pd.DataFrame([feature_values])
            input_scaled = st.session_state['scaler'].transform(input_data)
            prediction = st.session_state['model'].predict(input_scaled)[0]

            st.success(f"### Predicted House Price: **${prediction*1000:,.0f}**")

            # Show where this prediction falls in the distribution
            percentile = (df['PRICE'] < prediction).mean() * 100
            st.info(f"This price is higher than {percentile:.1f}% of houses in the dataset.")


if __name__ == "__main__":
    main()
