import pickle
import kagglehub
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

encoder = OneHotEncoder(handle_unknown='ignore')

# ----------------------------
# Load Dataset
# ----------------------------
path = kagglehub.dataset_download("alphiree/cardiovascular-diseases-risk-prediction-dataset")
#print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/CVD_cleaned.csv")

# Drop unnecessary columns
for col in ['Unnamed: 0', 'id']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Encode target
df['Heart_Disease'] = df['Heart_Disease'].map({'No': 0, 'Yes': 1})
#print('')
#print(df['Heart_Disease'].value_counts())

# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=22, stratify=df['Heart_Disease'])

#print(train.shape)
#print(test.shape)

# ----------------------------
# Split X and y
# ----------------------------
X_train = train.drop("Heart_Disease", axis=1)
y_train = train["Heart_Disease"].copy()
X_test = test.drop("Heart_Disease", axis=1)
y_test = test["Heart_Disease"].copy()

# ----------------------------
# Preprocessing Pipelines
# ----------------------------
cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore', drop='first'))
num_pipeline = make_pipeline(
    FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
    StandardScaler()
)
agecat_pipeline = make_pipeline(OrdinalEncoder())
genhealth_pipeline = make_pipeline(
    OrdinalEncoder(categories=[['Poor','Fair','Good','Very Good','Excellent']])
)
checkup_pipeline = make_pipeline(
    OrdinalEncoder(categories=[[
        'Within the past year','Within the past 2 years',
        'Within the past 5 years','5 or more years ago','Never'
    ]])
)

# Column assignments
num_pipe_col = [
    'Height_(cm)','Weight_(kg)','BMI','Alcohol_Consumption',
    'Fruit_Consumption','Green_Vegetables_Consumption','FriedPotato_Consumption'
]
cat_pipe_col = ['Arthritis','Depression','Diabetes','Exercise','Other_Cancer','Sex',
                'Skin_Cancer','Smoking_History']

preprocessing = ColumnTransformer([
    ('Categorical', cat_pipeline, cat_pipe_col),
    ('Age_Category', agecat_pipeline, ['Age_Category']),
    ('Checkup', checkup_pipeline, ['Checkup']),
    ('Gen_health', genhealth_pipeline, ['General_Health']),
    ('Numerical', num_pipeline, num_pipe_col),
], remainder='passthrough')

# ----------------------------
#  Cross-Validation with Logistic Regression
# ----------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=22)

log_reg = LogisticRegression(max_iter=10000, random_state=22)

model_pipeline = make_pipeline(
    preprocessing,
    SMOTE(random_state=22),
    log_reg
)

scores = cross_val_score(model_pipeline, X_train, y_train, scoring='f1', cv=kf, n_jobs=1)
#print('------------------------------------------------------------')
#print(f'The mean F1 score for Logistic Regression is {np.mean(scores)}')

# Fit final model
final_model = model_pipeline
final_model.fit(X_train, y_train)

# ----------------------------
# Prediction Functions
# ----------------------------
def risk_category(prob):
    """Convert probability into risk category."""
    if prob < 25:
        return "Low Risk"
    elif prob < 50:
        return "Moderate Risk"
    elif prob < 75:
        return "High Risk"
    else:
        return "Very High Risk"

def predict_risk(
    Age_Category, General_Health, Checkup, Sex, Smoking_History,
    Arthritis, Diabetes, Depression, Other_Cancer, Skin_Cancer, Exercise,
    Height_cm, Weight_kg, BMI, Alcohol_Consumption, Fruit_Consumption,
    Green_Vegetables_Consumption, FriedPotato_Consumption
):
    user_data = pd.DataFrame([{
        'Age_Category': Age_Category,
        'General_Health': General_Health,
        'Checkup': Checkup,
        'Sex': Sex,
        'Smoking_History': Smoking_History,
        'Arthritis': Arthritis,
        'Diabetes': Diabetes,
        'Depression': Depression,
        'Other_Cancer': Other_Cancer,
        'Skin_Cancer': Skin_Cancer,
        'Exercise': Exercise,
        'Height_(cm)': Height_cm,
        'Weight_(kg)': Weight_kg,
        'BMI': BMI,
        'Alcohol_Consumption': Alcohol_Consumption,
        'Fruit_Consumption': Fruit_Consumption,
        'Green_Vegetables_Consumption': Green_Vegetables_Consumption,
        'FriedPotato_Consumption': FriedPotato_Consumption
    }])
    prob = final_model.predict_proba(user_data)[0][1] * 100
    return risk_category(prob)

'''def get_user_inputs():
    print("Please provide the following information:")

    Age_Category = input("Age Category (e.g., '18-24', '25-29', '30-34', ..., '80+'): ")
    General_Health = input("General Health (Poor/Fair/Good/Very Good/Excellent): ")
    Checkup = input("Last Medical Checkup (Within the past year / Within the past 2 years / Within the past 5 years / 5 or more years ago / Never): ")
    Sex = input("Sex (Male/Female): ")
    Smoking_History = input("Do you smoke? (Yes/No): ")
    Arthritis = input("Do you have arthritis? (Yes/No): ")
    Diabetes = input("Do you have diabetes? (Yes/No): ")
    Depression = input("Do you have depression? (Yes/No): ")
    Other_Cancer = input("History of other cancer? (Yes/No): ")
    Skin_Cancer = input("History of skin cancer? (Yes/No): ")
    Exercise = input("Do you exercise regularly? (Yes/No): ")

    Height_cm = float(input("Height (cm): "))
    Weight_kg = float(input("Weight (kg): "))
    BMI = float(input("BMI (Body Mass Index): "))
    Alcohol_Consumption = float(input("Alcohol consumption (times per week): "))
    Fruit_Consumption = float(input("Fruit consumption (times per day): "))
    Green_Vegetables_Consumption = float(input("Green vegetables consumption (times per day): "))
    FriedPotato_Consumption = float(input("Fried potato consumption (times per week): "))

    return {
        'Age_Category': Age_Category,
        'General_Health': General_Health,
        'Checkup': Checkup,
        'Sex': Sex,
        'Smoking_History': Smoking_History,
        'Arthritis': Arthritis,
        'Diabetes': Diabetes,
        'Depression': Depression,
        'Other_Cancer': Other_Cancer,
        'Skin_Cancer': Skin_Cancer,
        'Exercise': Exercise,
        'Height_(cm)': Height_cm,
        'Weight_(kg)': Weight_kg,
        'BMI': BMI,
        'Alcohol_Consumption': Alcohol_Consumption,
        'Fruit_Consumption': Fruit_Consumption,
        'Green_Vegetables_Consumption': Green_Vegetables_Consumption,
        'FriedPotato_Consumption': FriedPotato_Consumption
    }
'''
def predict_heart_disease_risk(user_inputs):
    user_data = pd.DataFrame([user_inputs])
    prob = final_model.predict_proba(user_data)[0][1] * 100
    return risk_category(prob)

# ----------------------------
# Main execution
# ----------------------------
'''if __name__ == "__main__":
    inputs = get_user_inputs()
    category = predict_heart_disease_risk(inputs)
    print(f"\nPredicted Heart Disease Risk Category: {category}")'''

with open('final_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)
