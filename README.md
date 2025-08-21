Overview

CardioSense is a web application that uses machine learning to estimate the risk of cardiovascular disease based on user-provided health data. It provides an accessible and simple way for individuals to get an early indication of potential risk factors, encouraging proactive health awareness.


Features
 - User-friendly form for entering health metrics such as age, blood pressure, cholesterol levels, and more
 - Machine learning model that predicts the probability of cardiovascular disease risk
 - Responsive design for desktop and mobile screens
 - Clean and modern UI with consistent input fields and focus states
 - Gradient header with branding and tagline
 - Disclaimer section to remind users the tool is for educational purposes only



Tech Stack
 - Frontend: HTML, CSS, JavaScript
 - Backend: Flask (Python)
 - Machine Learning: Scikit-learn
 - Styling: Custom CSS with responsive design principles



Installation
 - Clone the repository:
    git clone https://github.com/atreyishukla/CardioSense.git
    cd cardiovascular-risk-predictor
 - Create and activate a virtual environment:
    python -m venv venv
    source venv/bin/activate   # On macOS/Linux  
    venv\Scripts\activate      # On Windows  
 - Install the dependencies:
    pip install -r requirements.txt
 - Run the Flask application:
    flask run
 - Open your browser and go to:
    http://127.0.0.1:5000



Usage
 - Fill out the input fields with your health information.
 - Submit the form to get your cardiovascular disease risk prediction.
 - Review the disclaimer before interpreting the results.



Disclaimer

This tool is intended for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
