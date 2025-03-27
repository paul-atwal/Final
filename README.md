# NFL 4th Down Decision Model

This project builds a machine learning-powered decision model to evaluate and recommend 4th down decisions (Go for it, Field Goal, or Punt) in the NFL.

## Project Structure

- `src/`: Core source code, including all models and decision simulation logic.
- `notebooks/`: Data exploration.
- `test/`: Scripts for testing each model and simulation component.
- `data/`: Includes processed data used for modeling and predictions.
- `report/`: Contains the project report and outline.
- `analysis/`: SHAP-based analysis of model feature importance.
- `app.py`: Streamlit app for interactive demo.

## Models Used

- **Go & Field Goal Models**: Logistic Regression (predict success probability)
- **Punt Model**: Linear Regression (predict expected opponent yardline)
- **Win Probability Model**: XGBoost Regression (predict WP for game situations)

## Running the Streamlit App

https://fourthdownmodel.streamlit.app/

To launch the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Notes

Ensure the `.pkl` files are in the root `data/` directory.

## SHAP Analysis

A script in `analysis/` shows feature importance using SHAP to interpret the Go/Field Goal model decisions.

## Credits

Created as part of a project for CMPT 419 on human-centered and data-centric AI. Inspired by the nflfastR dataset and existing fourth down decision tools.
