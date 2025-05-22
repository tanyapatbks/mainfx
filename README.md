🏦 Enhanced Forex Trend Prediction System
ระบบทำนายแนวโน้มราคาในตลาด Forex ที่ได้รับการปรับปรุงใหม่ โดยใช้แบบจำลองการเรียนรู้ของเครื่อง (Machine Learning) เปรียบเทียบระหว่างการฝึกฝนแบบจำลองด้วยข้อมูลจากคู่สกุลเงินเดียวกับการฝึกฝนด้วยข้อมูลจากหลายคู่สกุลเงินรวมกัน (Bagging approach)

🆕 สิ่งที่ปรับปรุงใหม่
✨ Enhanced Features
🔧 Smart Hyperparameter Loading: อ่านค่า hyperparameters จากไฟล์ JSON อัตโนมัติ
📊 Comprehensive Visualizations: กราฟและการวิเคราะห์ที่ครอบคลุมทุกขั้นตอน
📋 HTML Reports: รายงานแบบ interactive ที่สวยงามและครบถ้วน
🎯 Enhanced Hyperparameter Tuning: การปรับค่า hyperparameters ที่ดีขึ้นพร้อม visualization
🚀 Improved Code Structure: โครงสร้างโค้ดที่ชัดเจนและลดความซ้ำซ้อน
📈 Visualization Categories
1. Data Visualization (ก่อนเทรน)
Time Series Plot: กราฟราคา Close ของปีที่ใช้ train
Correlation Heatmap: ความสัมพันธ์ระหว่าง OHLCV features
Feature Distribution: การกระจายของ features หลัก
2. Model Training Visualization
Loss Curve: กราฟ Training Loss และ Validation Loss vs Epochs
Accuracy Progression: การเปลี่ยนแปลงของ accuracy ในระหว่างการเทรน
Overfitting/Underfitting Analysis: การวิเคราะห์ปัญหาการเรียนรู้
3. Prediction Results Visualization
Predicted vs Actual: เปรียบเทียบราคาที่โมเดลทำนายกับราคาจริง
Scatter Plot: ความสัมพันธ์เชิงเส้นของค่าที่โมเดลทำนาย
Prediction Confidence: การกระจายของความมั่นใจในการทำนาย
Direction Accuracy: ความแม่นยำในการทำนายทิศทาง
4. Evaluation Metrics Visualization
Annual Return Comparison: เปรียบเทียบผลตอบแทนรายปี
Win Rate Analysis: อัตราส่วนของการเทรดที่ทำกำไร
Risk-Return Analysis: การวิเคราะห์ความสัมพันธ์ระหว่างความเสี่ยงและผลตอบแทน
Market Condition Performance: ประสิทธิภาพในสภาวะตลาดที่แตกต่างกัน
Single vs Bagging Comparison: เปรียบเทียบแบบ Single กับ Bagging
Performance Summary Table: ตารางสรุปผลงานที่ครอบคลุม
5. Feature Importance
XGBoost Feature Importance: ความสำคัญของ features จากโมเดล XGBoost
Top Features Analysis: การวิเคราะห์ features ที่สำคัญที่สุด
🚀 คุณสมบัติหลัก (Features)
📋 ขั้นตอนทั้งหมด 4 ขั้นตอน:
Data Collection & Preprocessing - รวบรวมและเตรียมข้อมูล
Feature Enhancement - ปรับปรุงและเลือก features
Model Development - พัฒนาแบบจำลอง
Model Evaluation & Performance Analysis - ประเมินและวิเคราะห์ผลงาน
🤖 แบบจำลองที่ใช้:
CNN-LSTM (Hybrid) - รวมจุดเด่นของ CNN และ LSTM
Temporal Fusion Transformer (TFT) - แบบจำลอง attention-based ล่าสุด
XGBoost - แบบจำลอง gradient boosting ที่มีประสิทธิภาพสูง
💱 คู่สกุลเงิน:
EURUSD - ยูโรต่อดอลลาร์สหรัฐ
GBPUSD - ปอนด์อังกฤษต่อดอลลาร์สหรัฐ
USDJPY - ดอลลาร์สหรัฐต่อเยนญี่ปุ่น
🎯 เทคนิคการเลือกฟีเจอร์:
Random Forest Importance - ความสำคัญจาก Random Forest
Mutual Information - ข้อมูลร่วม
Principal Component Analysis (PCA) - การวิเคราะห์องค์ประกอบหลัก
⚙️ การปรับค่าพารามิเตอร์:
Optuna Integration - ใช้ Optuna สำหรับการค้นหาค่าพารามิเตอร์ที่เหมาะสมที่สุด
Enhanced Visualization - กราฟและรายงานที่ละเอียดสำหรับการปรับค่า
Smart Parameter Loading - โหลดค่าที่ดีที่สุดอัตโนมัติ
📊 การวัดประสิทธิภาพ:
Annual Return - ผลตอบแทนรายปี
Win Rate Analysis - อัตราส่วนของการเทรดที่ทำกำไร
Risk Metrics - Maximum Drawdown, Sharpe Ratio
Market Condition Testing - ประสิทธิภาพในสภาวะตลาดที่แตกต่างกัน
Buy & Hold Comparison - เปรียบเทียบกับกลยุทธ์ซื้อและถือครอง
Single vs Bagging Analysis - เปรียบเทียบผลตอบแทนระหว่างสองวิธี
📦 การติดตั้ง (Installation)
1. โคลนโปรเจคนี้:
bash
git clone https://github.com/yourusername/enhanced-forex-trend-prediction.git
cd enhanced-forex-trend-prediction
2. ติดตั้ง dependencies:
bash
pip install -r requirements.txt
3. เตรียมข้อมูล:
วางไฟล์ข้อมูล CSV ในโฟลเดอร์หลัก:

EURUSD_1H.csv
GBPUSD_1H.csv
USDJPY_1H.csv
🎮 การใช้งาน (Usage)
🚀 รันระบบแบบเต็ม (Recommended)
bash
# รันระบบทั้งหมดพร้อม visualizations และ HTML report
python run_pipeline.py --create-report
⚙️ รันด้วยการกำหนดค่าเอง
bash
# กำหนด parameters เอง
python run_pipeline.py \
    --window-size 60 \
    --models cnn_lstm,tft,xgboost \
    --currency-pairs EURUSD,GBPUSD,USDJPY \
    --epochs 100 \
    --create-report
🔧 รันการปรับค่าพารามิเตอร์ (Enhanced)
bash
# ปรับค่าพารามิเตอร์สำหรับทุกโมเดล
python hyperparameter_tuning.py --trials 50

# ปรับค่าพารามิเตอร์สำหรับโมเดลเดียว
python hyperparameter_tuning.py --model cnn_lstm --pair EURUSD --trials 30
📊 ตัวอย่างการใช้งานขั้นสูง
bash
# รันแบบครบถ้วนพร้อมการปรับค่าพารามิเตอร์
python hyperparameter_tuning.py --trials 100
python run_pipeline.py --create-report

# รันเฉพาะ XGBoost สำหรับ EURUSD
python run_pipeline.py \
    --models xgboost \
    --currency-pairs EURUSD \
    --no-bagging \
    --create-report
📁 โครงสร้างโปรเจค (Project Structure)
enhanced-forex-trend-prediction/
├── 📜 forex_prediction.py              # ไฟล์หลักที่ปรับปรุงแล้ว
├── 🚀 run_pipeline.py                  # สคริปต์รันระบบที่ปรับปรุงแล้ว
├── 🔧 hyperparameter_tuning.py         # สคริปต์ปรับค่าพารามิเตอร์ที่ปรับปรุงแล้ว
├── 📊 EURUSD_1H.csv                    # ข้อมูล EURUSD
├── 📊 GBPUSD_1H.csv                    # ข้อมูล GBPUSD  
├── 📊 USDJPY_1H.csv                    # ข้อมูล USDJPY
├── 📂 logs/                            # โฟลเดอร์เก็บ log files
├── 📂 output/                          # โฟลเดอร์ผลลัพธ์
│   ├── 📂 models/                      # โมเดลที่ฝึกฝนแล้ว (.keras, .pkl)
│   ├── 📂 results/                     # ผลการประเมินโมเดล (.json)
│   ├── 📂 plots/                       # กราฟและภาพการวิเคราะห์ (.png)
│   ├── 📂 features/                    # ผลการเลือกฟีเจอร์ (.csv, .json)
│   ├── 📂 hyperparameter_tuning/       # ผลการปรับค่าพารามิเตอร์ (.json)
│   ├── 📂 hyperparameter_reports/      # รายงานการปรับค่าพารามิเตอร์ (.html, .png)
│   ├── 📋 comprehensive_report.html    # รายงานรวมแบบ interactive
│   └── ⚙️ enhanced_run_config.json     # การตั้งค่าที่ใช้รัน
└── 📄 requirements.txt                 # รายการ libraries ที่ต้องใช้
📊 การดูผลลัพธ์ (Results)
🎯 ไฟล์ผลลัพธ์หลัก:
📋 comprehensive_report.html: รายงานแบบ interactive ที่ครบถ้วน
📊 plots/: กราฟการวิเคราะห์ทั้งหมด (20+ ไฟล์)
📈 enhanced_final_results.json: ผลการประเมินโดยละเอียด
⚙️ hyperparameter_tuning/: ผลการปรับค่าพารามิเตอร์
🖼️ กราฟที่สำคัญ:
data_time_series_plots.png - กราฟราคาในช่วงเทรน
data_correlation_heatmaps.png - Correlation ของ OHLCV
{model}_training_history.png - Loss และ Accuracy curves
{model}_prediction_analysis.png - การวิเคราะห์การทำนาย
evaluation_metrics_comprehensive.png - เปรียบเทียบประสิทธิภาพ
single_vs_bagging_detailed_comparison.png - Single vs Bagging
risk_return_analysis.png - ความสัมพันธ์ Risk-Return
performance_summary_table.png - ตารางสรุปผลงาน
🎛️ คำแนะนำในการใช้งาน
🥇 สำหรับผู้เริ่มต้น:
bash
# รันแบบง่ายพร้อมสร้างรายงาน
python run_pipeline.py --create-report
🔧 การปรับแต่งขั้นสูง:
1. การเลือกโมเดล:
bash
# รันเฉพาะ CNN-LSTM
python run_pipeline.py --models cnn_lstm --create-report

# รันหลายโมเดล
python run_pipeline.py --models cnn_lstm,tft --create-report
2. การเลือกคู่สกุลเงิน:
bash
# รันเฉพาะ EURUSD
python run_pipeline.py --currency-pairs EURUSD --create-report

# รันหลายคู่
python run_pipeline.py --currency-pairs EURUSD,GBPUSD --create-report
3. การปรับช่วงเวลา:
bash
# กำหนดช่วงเวลาเอง
python run_pipeline.py \
    --train-start 2020-01-01 \
    --train-end 2022-12-31 \
    --test-start 2023-01-01 \
    --test-end 2023-06-30 \
    --create-report
4. การปรับค่าพารามิเตอร์:
bash
# ปรับค่าพารามิเตอร์ก่อนรันระบบ
python hyperparameter_tuning.py --trials 100
python run_pipeline.py --create-report
📈 การวิเคราะห์ผลลัพธ์:
1. เปิดดู HTML Report:
bash
# เปิดไฟล์ในเบราว์เซอร์
open output/comprehensive_report.html
2. ตรวจสอบกราฟสำคัญ:
Training History: ดู overfitting/underfitting
Prediction Analysis: ดูความแม่นยำของการทำนาย
Performance Comparison: เปรียบเทียบโมเดล
Risk-Return: ประเมินความเสี่ยง
3. วิเคราะห์ Hyperparameters:
bash
# ดูรายงานการปรับค่าพารามิเตอร์
open output/hyperparameter_reports/{pair}_{model}_hyperparameter_report.html
🔧 ข้อกำหนดทางเทคนิค (Technical Requirements)
Python 3.8+ หรือสูงกว่า
TensorFlow 2.x สำหรับ Deep Learning models
XGBoost สำหรับ Gradient Boosting
Optuna สำหรับการปรับค่าพารามิเตอร์
pandas, numpy, scikit-learn สำหรับการจัดการข้อมูล
matplotlib, seaborn สำหรับ visualization
Memory: อย่างน้อย 8GB RAM (แนะนำ 16GB+)
Storage: อย่างน้อย 5GB สำหรับผลลัพธ์
🎯 Tips & Best Practices
⚡ Performance Tips:
ใช้ GPU: ติดตั้ง TensorFlow-GPU สำหรับการเทรนที่เร็วขึ้น
Reduce Epochs: ลด epochs สำหรับการทดสอบเบื้องต้น
Select Models: เลือกเฉพาะโมเดลที่ต้องการสำหรับการทดสอบ
🔧 Troubleshooting:
Memory Issues: ลด batch_size หรือ window_size
NaN Values: ระบบจะจัดการอัตโนมัติ แต่ตรวจสอบข้อมูลต้นทาง
Long Training: ใช้ early stopping (ตั้งไว้แล้ว)
📊 Interpretation Guidelines:
Annual Return > 10%: ประสิทธิภาพดีมาก
Win Rate > 55%: การทำนายที่ดี
Max Drawdown < 5%: ความเสี่ยงที่ยอมรับได้
Sharpe Ratio > 1.0: อัตราส่วนความเสี่ยงต่อผลตอบแทนที่ดี
🤝 การพัฒนาต่อ (Future Enhancements)
🔮 แผนการพัฒนา:
 Real-time Trading Integration - เชื่อมต่อกับ broker APIs
 Advanced Ensemble Methods - รวมโมเดลหลายตัวแบบอัจฉริยะ
 Sentiment Analysis - เพิ่มการวิเคราะห์จากข่าวและ social media
 Alternative Data Sources - ข้อมูลทางเศรษฐกิจและการเมือง
 Portfolio Optimization - การจัดสรรเงินลงทุนแบบอัตโนมัติ
🛠️ Technical Improvements:
 MLflow Integration - experiment tracking
 Docker Support - containerization
 API Endpoints - REST API สำหรับการใช้งาน
 Database Integration - เก็บข้อมูลในฐานข้อมูล
 Cloud Deployment - deployment บน cloud platforms