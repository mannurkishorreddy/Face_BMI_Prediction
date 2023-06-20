import streamlit as st
import pandas as pd
from PIL import Image

def main():
    st.title("Image and BMI Scores Viewer")
    
    # Read CSV file
    df = pd.read_csv("predictions.csv")

    for index, row in df.iterrows():
        actual_bmi = row["Actual BMI"]
        predicted_bmi = row["Predicted BMI"]
        image_path = row["Image File"]

        # Display actual and predicted BMI scores
        st.write(f"Actual BMI: {actual_bmi}")
        st.write(f"Predicted BMI: {predicted_bmi}")

        # Load and display image
        image = Image.open(image_path)
        st.image(image)

        # Add a separator between images
        st.markdown("---")

if __name__ == "__main__":
    main()
