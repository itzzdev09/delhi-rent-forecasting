# 🏠 Delhi House Rent Forecasting

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning project to **forecast house rental prices in Delhi** using both **LSTM (deep learning)** and **Regression (classical ML)** models.  
The dataset is augmented with **synthetic data generated via Faker**, preprocessed in **PyTorch**, and deployed with a **Streamlit app** for interactive exploration.  
Future integrations will include **Docker containerization** and **Azure DevOps CI/CD** for end-to-end automation.  

---

## 📂 Project Structure

```bash
delhi-rent-forecasting/
│── data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│── models/
│   ├── lstm_model.pth
│   ├── regression_model.pth
│   ├── scaler.pkl
│── app/
│   ├── app.py                   # Streamlit app
│── utils/
│   ├── preprocessing.py         # Cleaning, time-series conversion, faker augmentation
│   ├── model_lstm.py            # LSTM architecture
│   ├── model_regression.py      # Regression architecture
│   ├── trainer.py               # Training + evaluation loops
│── scripts/
│   ├── augment_data.py          # Uses Faker to generate synthetic Delhi data
│   ├── preprocess_data.py       # Cleans and prepares processed_data.csv
│   ├── train_lstm.py            # Trains and saves LSTM model
│   ├── train_regression.py      # Trains and saves regression model
│   ├── evaluate_models.py       # Compare performance of LSTM vs regression
│── requirements.txt
│── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/delhi-rent-forecasting.git
cd delhi-rent-forecasting
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Workflow

### 🔹 Step 1: Generate Synthetic Data
```bash
python scripts/augment_data.py
```

### 🔹 Step 2: Preprocess Data
```bash
python scripts/preprocess_data.py
```

### 🔹 Step 3: Train LSTM Model
```bash
python scripts/train_lstm.py
```

### 🔹 Step 4: Train Regression Model
```bash
python scripts/train_regression.py
```

### 🔹 Step 5: Evaluate Models
```bash
python scripts/evaluate_models.py
```

### 🔹 Step 6: Run Streamlit App
```bash
streamlit run app/app.py
```

---

## 📊 Models Used

- **LSTM (Long Short-Term Memory):** For capturing temporal dependencies in rental price trends.  
- **Linear Regression / Random Forest / XGBoost:** For baseline comparisons.  

---

## 🔮 Future Enhancements

- ✅ Add **Docker containerization** for reproducible builds  
- ✅ Set up **Azure DevOps CI/CD pipelines** for automated deployment  
- ✅ Deploy Streamlit app on **Azure Web App**  

---

## 🤝 Contributing

1. Fork the repo  
2. Create a new branch (`feature-xyz`)  
3. Commit changes  
4. Open a Pull Request  

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- [Faker](https://faker.readthedocs.io/en/master/) for synthetic data generation  
- [PyTorch](https://pytorch.org/) for model building  
- [Streamlit](https://streamlit.io/) for interactive UI  

