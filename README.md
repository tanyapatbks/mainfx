Forex Trend Prediction System
ระบบทำนายแนวโน้มราคาในตลาด Forex โดยใช้แบบจำลองการเรียนรู้ของเครื่อง (Machine Learning) เปรียบเทียบระหว่างการฝึกฝนแบบจำลองด้วยข้อมูลจากคู่สกุลเงินเดียวกับการฝึกฝนด้วยข้อมูลจากหลายคู่สกุลเงินรวมกัน (Bagging approach)
คุณสมบัติหลัก (Features)

ขั้นตอนทั้งหมด 4 ขั้นตอน:

Data Collection & Preprocessing
Feature Enhancement
Model Development
Model Evaluation & Performance Analysis


แบบจำลองที่ใช้:

CNN-LSTM (Hybrid)
Temporal Fusion Transformer (TFT)
XGBoost


คู่สกุลเงิน:

EURUSD
GBPUSD
USDJPY


เทคนิคการเลือกฟีเจอร์ (Feature Selection):

Random Forest Importance
Mutual Information
Principal Component Analysis (PCA)


การปรับค่าพารามิเตอร์ (Hyperparameter Tuning):

ใช้ Optuna สำหรับการค้นหาค่าพารามิเตอร์ที่เหมาะสมที่สุด


การวัดประสิทธิภาพ:

ผลตอบแทนรายปี (Annual Return)
อัตราส่วนของการเทรดที่ทำกำไร (Win Rate Analysis)
ประสิทธิภาพในสภาวะตลาดที่แตกต่างกัน (Different Market Conditions Testing)
เปรียบเทียบกับกลยุทธ์ซื้อและถือครอง (Bagging vs Buy & Hold)
เปรียบเทียบผลตอบแทนระหว่างสองวิธี (Single-Bagging Return Comparison)



การติดตั้ง (Installation)

โคลนโปรเจคนี้:

bashgit clone https://github.com/yourusername/forex-trend-prediction.git
cd forex-trend-prediction

ติดตั้ง dependencies:

bashpip install -r requirements.txt
การใช้งาน (Usage)
รันระบบทั้งหมด
bashpython run_pipeline.py
ด้วยการกำหนดค่าพารามิเตอร์เอง
bashpython run_pipeline.py --window-size 60 --models cnn_lstm,tft,xgboost --currency-pairs EURUSD,GBPUSD,USDJPY
รันการปรับค่าพารามิเตอร์ (Hyperparameter Tuning)
bashpython hyperparameter_tuning.py --model cnn_lstm --pair EURUSD --trials 50
โครงสร้างโปรเจค (Project Structure)
forex-trend-prediction/
├── forex_prediction.py      # ไฟล์หลักที่มีฟังก์ชันทั้งหมด
├── run_pipeline.py          # สคริปต์สำหรับรันระบบทั้งหมด
├── hyperparameter_tuning.py # สคริปต์สำหรับปรับค่าพารามิเตอร์
├── EURUSD_1H.csv            # ข้อมูล EURUSD timeframe 1 ชั่วโมง
├── GBPUSD_1H.csv            # ข้อมูล GBPUSD timeframe 1 ชั่วโมง
├── USDJPY_1H.csv            # ข้อมูล USDJPY timeframe 1 ชั่วโมง
├── logs/                    # โฟลเดอร์สำหรับเก็บ log
├── output/                  # โฟลเดอร์สำหรับเก็บผลลัพธ์
│   ├── models/              # โมเดลที่ฝึกฝนแล้ว
│   ├── results/             # ผลการประเมินโมเดล
│   ├── plots/               # กราฟและภาพการวิเคราะห์
│   └── features/            # ผลการเลือกฟีเจอร์
└── requirements.txt         # รายการ libraries ที่ต้องใช้
คำแนะนำในการใช้งาน

รันระบบแบบง่าย:

ใช้คำสั่ง python run_pipeline.py เพื่อรันขั้นตอนทั้งหมดด้วยค่าพารามิเตอร์เริ่มต้น


การกำหนดคู่สกุลเงิน:

ใช้ออปชัน --currency-pairs เพื่อระบุคู่สกุลเงินที่ต้องการ เช่น --currency-pairs EURUSD,GBPUSD


การเลือกแบบจำลอง:

ใช้ออปชัน --models เพื่อระบุแบบจำลองที่ต้องการ เช่น --models cnn_lstm,xgboost


การปรับค่าพารามิเตอร์:

ใช้คำสั่ง python hyperparameter_tuning.py เพื่อปรับค่าพารามิเตอร์ที่เหมาะสมที่สุด
กำหนดแบบจำลองที่ต้องการปรับด้วยออปชัน --model และคู่สกุลเงินด้วยออปชัน --pair


การปรับแต่งช่วงเวลาการทดสอบ:

ใช้ออปชัน --train-start, --train-end, --test-start, และ --test-end เพื่อกำหนดช่วงเวลาสำหรับการฝึกฝนและทดสอบแบบจำลอง



การดูผลลัพธ์ (Results)

ผลลัพธ์ทั้งหมดจะถูกบันทึกไว้ในโฟลเดอร์ output/
กราฟและการวิเคราะห์ผลจะอยู่ในโฟลเดอร์ output/plots/
Log ของการทำงานจะอยู่ในโฟลเดอร์ logs/
ผลการประเมินแบบจำลองจะอยู่ในไฟล์ output/results/evaluation_results.json

ข้อกำหนดทางเทคนิค (Technical Requirements)

Python 3.8 หรือสูงกว่า
TensorFlow 2.x
XGBoost
pandas, numpy, scikit-learn
matplotlib, seaborn
optuna (สำหรับการปรับค่าพารามิเตอร์)