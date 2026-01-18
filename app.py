import joblib
import pandas as pd
import streamlit as st
from train import HousePricePredictor
import warnings
warnings.filterwarnings("ignore")

@st.cache_resource
def load_pipeline(file_path):
  return joblib.load(file_path)

pipeline = load_pipeline("pipeline.pkl")

st.title("House Price Predictor")
st.write("Enter the features of the house to predict its sale price.")

st.header("Basic House Features")
column1, column2 = st.columns(2)

with column1:
  overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
  total_sf = st.number_input("Total Square Footage", min_value=500, max_value=6000, value=2000)
  lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=50000, value=10000)

with column2:
  year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1990)
  neighborhood = st.selectbox("Neighborhood",
                              options=[
                                  'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
                                  'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
                                  'Blueste'
                              ])
  gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", min_value=300, max_value=5000, value=1500)

ful_bath = st.selectbox("Full Baths", options=[0, 1, 2, 3, 4])
garage_area = st.number_input("Garage Area (sq ft)", 0, 1000, 500)
garage_cars = st.number_input("Garage Cars", 0, 4, 2)

# Advanced Options
st.header("Advanced Options")
show_advanced = st.checkbox("Show Advanced Features")

if show_advanced:
  col1, col2, col3 = st.columns(3)
  with col1:
    bedroom_abv_gr = st.selectbox("Bedrooms Above Grade", options=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    kitchen_qual = st.selectbox("Kitchen Quality", options=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
    exter_qual = st.selectbox("Exterior Quality", options=['Po', 'Fa', 'TA', 'Gd', 'Ex'])
  with col2:
    bsmt_qual = st.selectbox("Basement Quality", options=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])
    garage_finish = st.selectbox("Garage Finish", options=['None', 'Unf', 'RFn', 'Fin'])
    paved_drive = st.selectbox("Paved Driveway", options=['N', 'P', 'Y'])
  with col3:
    central_air = st.selectbox("Central Air", options=['N', 'Y'])
    house_style = st.selectbox("House Style", options=['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'])
    sale_condition = st.selectbox("Sale Condition", options=['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'])

if st.button("Predict Price"):
  # Create input DataFrame with the collected features
  input_data = {
      'OverallQual': overall_qual,
      'TotalSF': total_sf,
      'YearBuilt': year_built,
      'Neighborhood': neighborhood,
      'FullBath': ful_bath,
      'GarageArea': garage_area,
      'GarageCars': garage_cars,
      'LotArea': lot_area,
      'GrLivArea': gr_liv_area
  }
  
  if show_advanced:
    input_data.update({
        'BedroomAbvGr': bedroom_abv_gr,
        'KitchenQual': kitchen_qual,
        'ExterQual': exter_qual,
        'BsmtQual': bsmt_qual,
        'GarageFinish': garage_finish,
        'PavedDrive': paved_drive,
        'CentralAir': central_air,
        'HouseStyle': house_style,
        'SaleCondition': sale_condition
    })
  
  input_data = pd.DataFrame([input_data])

  try:
    # Pipeline handles everything: cleaning, engineering, encoding, prediction
    prediction, explanation = pipeline._predict_with_explanation(input_data)

    st.success(f"**Predicted Sale Price: ${prediction[0]:,.0f}**")

    st.subheader("Key Features Impacting the Price:")
    for explanation_item in explanation:
      st.markdown(f"- **{explanation_item['feature']}**: *{explanation_item['contribution']}* by *${abs(explanation_item['impact']):.0f}*")
  except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your input values.")