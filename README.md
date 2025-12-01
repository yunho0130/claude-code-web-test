# Boston House Price Prediction Dashboard

A Streamlit-based interactive dashboard for exploring and predicting Boston house prices using PandasAI integration.

## Features

- **Data Overview**: Explore the Boston Housing dataset with statistical summaries
- **Visualizations**: Interactive charts including correlation heatmaps, scatter plots, and distribution plots
- **PandasAI Integration**: Natural language queries for data exploration (requires OpenAI API key)
- **ML Prediction**: Train and compare multiple models (Linear Regression, Random Forest, Gradient Boosting)
- **Custom Predictions**: Predict house prices with custom feature inputs

## Installation

```bash
# Clone the repository
git clone https://github.com/yunho0130/claude-code-web-test.git
cd claude-code-web-test

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the dashboard
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

## PandasAI Setup (Optional)

To use natural language queries:
1. Get an OpenAI API key from https://platform.openai.com
2. Enter the API key in the sidebar of the dashboard

## Dataset

The dashboard uses the Boston Housing dataset which includes:
- **CRIM**: Per capita crime rate
- **ZN**: Proportion of residential land zoned for large lots
- **INDUS**: Proportion of non-retail business acres
- **CHAS**: Charles River dummy variable
- **NOX**: Nitric oxides concentration
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Property-tax rate
- **PTRATIO**: Pupil-teacher ratio
- **B**: Proportion of Black residents
- **LSTAT**: % lower status of the population
- **PRICE**: Median value of owner-occupied homes (target variable)