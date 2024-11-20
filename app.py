{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f4c18c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "dftrain=pd.read_csv(r\"C:\\Users\\polpi\\Desktop\\data science\\assignments\\done\\Logistic Regression\\Titanic_train.csv\")\n",
    "dftrain\n",
    "dftrain=dftrain.drop('Cabin',axis=1)\n",
    "dftrain_cleaned=dftrain.dropna()\n",
    "dftrain_cleaned"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c9e2851",
   "metadata": {},
   "outputs": [],
   "source": [
    "dftrain_cleaned=dftrain_cleaned.drop(['PassengerId','Name','Ticket'], axis=1)\n",
    "dftrain_cleaned"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c317eca5",
   "metadata": {},
   "outputs": [],
   "source": [
    "sex_type={'male':1,'female':0}\n",
    "embarked_types={'S':2,'C':1,'Q':0}\n",
    "dftrain_cleaned['Sex'] = dftrain_cleaned['Sex'].map(sex_type)\n",
    "dftrain_cleaned['Embarked'] = dftrain_cleaned['Embarked'].map(embarked_types)\n",
    "dftrain_cleaned"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "183dd503",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split as tts\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "x=dftrain_cleaned.drop('Survived', axis=1)\n",
    "y=dftrain_cleaned['Survived']\n",
    "x_train,x_test,y_train,y_test=tts(x,y,train_size=0.7,random_state=42)\n",
    "x_train.shape,y_train.shape,x_test.shape,y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "261e6347",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "model = LogisticRegression(max_iter=10000)\n",
    "model.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af323a81",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "joblib.dump(model, \"logistic_regression_titanic_model.pkl\")\n",
    "model = joblib.load(\"logistic_regression_titanic_model.pkl\")\n",
    "\n",
    "sex_type = {'Male': 1, 'Female': 0}\n",
    "embarked_types = {'Southampton': 2, 'Cherbourg': 1, 'Queenstown': 0}\n",
    "st.title(\"Titanic Survival Prediction\")\n",
    "st.write(\"Enter passenger details to predict survival:\")\n",
    "\n",
    "# Inputs for the user\n",
    "pclass = st.selectbox(\"Passenger Class\", [1, 2, 3], index=2)\n",
    "sex = st.selectbox(\"Sex\", list(sex_type.keys()))\n",
    "age = st.slider(\"Age\", min_value=0, max_value=100, value=30)\n",
    "sibsp = st.number_input(\"Number of Siblings/Spouses Aboard\", min_value=0, max_value=10, value=0)\n",
    "parch = st.number_input(\"Number of Parents/Children Aboard\", min_value=0, max_value=10, value=0)\n",
    "fare = st.number_input(\"Fare Paid\", min_value=0.0, max_value=1000.0, value=50.0)\n",
    "embarked = st.selectbox(\"Port of Embarkation\", list(embarked_types.keys()))\n",
    "\n",
    "# A prediction button\n",
    "if st.button(\"Predict Survival\"):\n",
    "    input_data = np.array([\n",
    "        pclass,\n",
    "        sex_type[sex],\n",
    "        age,\n",
    "        sibsp,\n",
    "        parch,\n",
    "        fare,\n",
    "        embarked_types[embarked]\n",
    "    ]).reshape(1, -1)\n",
    "\n",
    "\n",
    "    prediction = model.predict(input_data)\n",
    "    prediction_proba = model.predict_proba(input_data)\n",
    "\n",
    "\n",
    "    if prediction[0] == 1:\n",
    "        st.success(\"The passenger is predicted to survive.\")\n",
    "    else:\n",
    "        st.error(\"The passenger is predicted not to survive.\")\n",
    "\n",
    "    st.write(f\"Prediction Confidence: {prediction_proba[0][1]:.2f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
