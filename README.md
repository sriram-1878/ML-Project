# Customer Churn Prediction Using Machine Learning

## Project overview
This project demonstrates how machine learning can estimate whether a customer is likely to leave a service. It includes a Streamlit web app, a sample dataset, data visualizations, model evaluation, and an interactive prediction form.

## Project structure
- `app.py` - Streamlit application and machine-learning workflow
- `data/customer_churn_sample.csv` - generated sample dataset (1,200 rows)
- `requirements.txt` - Python dependencies

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract the ZIP file.
3. Open Command Prompt in the extracted project folder.
4. Install the dependencies:

   `python -m pip install -r requirements.txt`

5. Start the application:

   `python -m streamlit run app.py`

6. Open the local URL shown in the terminal (usually http://localhost:8501).

## Use your own dataset
Use the upload option in the sidebar to upload a CSV file. The file must contain a target column named `churn`, where `0` means the customer stayed and `1` means the customer churned. Other columns are treated as input features. Use enough rows and include both target classes.

The included sample dataset uses these features:
- `tenure`
- `monthly_charges`
- `contract`
- `internet_service`
- `tech_support`
- `payment_method`
- `paperless_billing`
- `senior_citizen`
- `churn` (target)

## Machine-learning workflow
1. Load customer data.
2. Separate input features and the churn target.
3. Impute missing values and encode categorical features.
4. Split data into training and testing sets.
5. Train a Random Forest classifier.
6. Display test accuracy and a confusion matrix.
7. Predict churn probability for a customer entered in the app.

## Important note
The included dataset is synthetic and intended for learning and demonstration. Its metrics do not represent real business performance. For a meaningful project evaluation, use a suitable, permissioned real-world dataset and document its source.
