import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Supermarket Sales Prediction App',
                   layout='wide')


# ---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    cbr = CatBoostRegressor(iterations=parameter_iteration,
                            random_state=parameter_random_state,
                            learning_rate=parameter_learning_rate,
                            depth=parameter_depth)

    # rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
    #                            random_state=parameter_random_state,
    #                            max_features=parameter_max_features,
    #                            criterion=parameter_criterion,
    #                            min_samples_split=parameter_min_samples_split,
    #                            min_samples_leaf=parameter_min_samples_leaf,
    #                            bootstrap=parameter_bootstrap,
    #                            oob_score=parameter_oob_score,
    #                            n_jobs=parameter_n_jobs)
    cbr.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = cbr.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = cbr.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(cbr.get_params())


# ---------------------------------#
st.write("""
# The Machine Learning App
In this implementation, the *CatBoostRegressor()* function is used in this app for build a regression model using the **CatBoost** algorithm.
Try adjusting the hyperparameters!
""")

# ---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](F://Samsung Innovation Campus (SIC)//Tasks//Final Project//Stores.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_iteration = st.sidebar.slider('Number of iteration', 0, 20, 2, 1)
    parameter_learning_rate = st.sidebar.slider('Number of learning rate', 0, 10, 1, 1)
    parameter_depth = st.sidebar.slider('Number of depth', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 116, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])



# ---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        # diabetes = load_diabetes()
        # X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        # Y = pd.Series(diabetes.target, name='response')
        # df = pd.concat( [X,Y], axis=1 )

        # st.markdown('The Diabetes dataset is used as the example.')
        # st.write(df.head(5))

        # Boston housing dataset
        stores = pd.read_csv('F:\Samsung Innovation Campus (SIC)\Tasks\Final Project\Stores.csv')
        X = pd.DataFrame(stores.data, columns=stores.feature_names)
        Y = pd.Series(stores.target, name='Store_Sales')
        df = pd.concat([X, Y], axis=1)

        st.markdown('The Supermarket store branches sales dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)