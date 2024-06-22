
# S-Kare Skin Cancer Detection 

## Overview

This repository contains the code for a comprehensive skin cancer detection. The application leverages advanced NLP techniques and image analysis to provide an efficient and user-friendly solution for patients concerned about skin health. Key features include an NLP chatbot, image upload and analysis, and consultation booking for high-risk predictions. There are 2 Interfaces , one for Patient (App Directory) and another for Doctors(Doctor App Directory)

## Features - Patient

### NLP Chatbot
- **Engages in a conversation** with the patient to understand their skin problems using advanced NLP techniques.
- **Extracts relevant information** from the patient's responses to assist in diagnosis.

### Image Upload
- **Allows patients to upload images** of the affected skin area.
- **Supports various image formats** for user convenience.

### Patient Metadata
- **Patients can add metadata** such as age, gender, and area of localization.
- **Enhances prediction accuracy** by incorporating patient-specific details.

### Image Analysis
- **Analyzes the uploaded image** to predict the likelihood of skin cancer.
- **Uses machine learning models** trained on a diverse dataset of skin images.

## Features - Doctor Interface

### Report Generation - (Doctor App Directory)
- **Add Image and Metadata and get detailed PDF Report for diagnostic Support


## Setup & Installation

### Prerequisites:
- Python 3.8+
- Django 3.
- Pip

### Installation Steps:
1. **Clone the Repository**
    ```sh
    cd  s-kare_chatbot
    ```
2. **Install Requriement**
    ```sh
    pip install -r requirements.txt
    ```
3. **Navigate to App or Doctor App directory as per requirment.
    ```sh
    cd  App / cd Doctor App
    ```
   
5. **Run the Development Server**
    ```sh
    streamlit run app.py --server.enableXsrfProtection false
    ```

## Usage

1. **Run the server** as described in the installation steps.
2. **Access the application** at localhost.
3. **Interact with the NLP chatbot** to describe your skin condition.
4. **Upload an image** of the affected skin area.
5. **Add metadata** such as age, gender, and area of localization.
6. **Review the analysis results** to check the likelihood of skin cancer.
7. **Book a consultation** if the prediction indicates high risk.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please contact [dypfinal@gmail.com].
