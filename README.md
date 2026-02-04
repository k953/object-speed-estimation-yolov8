# 🚗 Real-Time Vehicle Speed Estimation using YOLOv8

A complete end-to-end computer vision project that detects vehicles in a video stream, tracks them across frames, and estimates their speed using deep learning and classical tracking techniques.

This project demonstrates practical usage of **YOLOv8**, **OpenCV**, and **multi-object tracking**, following real-world engineering practices.

---

## 🔍 Problem Statement
Estimating vehicle speed from video footage is an important task in:
- Traffic monitoring systems
- Smart city surveillance
- Intelligent transportation systems (ITS)

Traditional approaches require hardware sensors, whereas this project solves the problem **purely using vision-based techniques**.

---

## 🧠 Solution Overview
The system follows a clear multi-stage pipeline:

1. **Object Detection**  
   Vehicles are detected in each video frame using a pretrained **YOLOv8** model.

2. **Object Tracking**  
   Detected vehicles are tracked across frames using a **centroid-based tracking algorithm**, ensuring consistent object IDs.

3. **Speed Estimation**  
   Speed is estimated by measuring the displacement of each vehicle’s centroid over time.

4. **Visualization**  
   Each vehicle is displayed with:
   - Bounding box
   - Unique ID
   - Estimated speed

---

## ✨ Key Features
- Real-time vehicle detection (cars, buses, trucks)
- Lightweight centroid-based multi-object tracking
- Pixel-based speed estimation
- Clean modular Python code
- Easy to extend for real-world deployment

---

## 🛠️ Tech Stack
- **Python 3**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **NumPy**

---

## 📂 Project Structure



## ▶️ How to Run
```bash
python main.py
The system will:

Load the YOLOv8 model

Process the input video

Display detected vehicles with unique ID and estimated speed

📐 Speed Estimation Method
The speed of each vehicle is computed using the formula:

ini
Copy code
Speed = Distance (in pixels) / Time (in seconds)
Where:

Distance = Euclidean distance between object centroids in consecutive frames

Time = Time difference between consecutive frames

⚠️ Important Note
The computed speed values are relative (pixels/second).
To convert them into real-world units (km/h), camera calibration and homography are required.

🚀 Future Enhancements
Convert pixel speed to real-world km/h using camera calibration

Improve tracking robustness using Kalman Filter / Deep SORT

Handle occlusions and missed detections

Save output as a processed video file

Deploy the system as a Streamlit or Flask web application
