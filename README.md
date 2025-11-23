# 🧠 Medhya – Posture Detection & Personal Health Assistant  
*A lightweight yet highly useful ML system powered by YOLOv8 Pose + LLM*

---

## 📌 Overview  
**Medhya** is a posture-analysis and personal health assistant that combines computer vision with LLM-based medical guidance.

It detects **head–shoulder–hip alignment**, evaluates posture quality (slouching, forward head, rounded shoulders, etc.), renders a clean annotated visualization, and generates structured first-aid + exercises + a friendly clinician-style explanation.

Surprisingly, this turned out to be one of the **easiest ML projects** I’ve built in this domain — smooth, practical, and very deployable.

---

## 🔍 Features  
- Custom-trained **YOLOv8 Pose** model  
- Detects 4 key posture points → head, shoulders, hips  
- Calculates:
  - spine angle  
  - neck angle  
  - forward head offset  
  - shoulder imbalance  
  - severity score  
- Clean posture overlay (head/shoulders/hips labels + smooth curved spine)  
- LLM-powered medical feedback:
  - structured first-aid steps  
  - 2 quick exercises  
  - severity classification  
  - warm bullet-point explanation  
  - medical disclaimer  
- FastAPI backend with two endpoints:
  - `/posture` → posture detection  
  - `/chat` → general personal doctor assistant  

---

## 📂 Dataset  
I used the **Posture Keypoints Detection dataset** from Kaggle:  
🔗 https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection/data  

Dataset contains:  

```

images/
train/
val/
labels/
train/
val/

```

Each image has a matching `.txt` annotation file containing:

```

class x_center y_center width height kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis ...

```

Where:
- coordinates are **normalized (0–1)**  
- visibility flags are standard YOLO keypoint flags (0/1/2)

### 🛠 Preparing the dataset  
1. Download dataset from Kaggle  
2. Extract into your project folder  
3. Ensure structure:

```

dataset/
images/
train/
val/
labels/
train/
val/

````

4. Create `data.yaml`:

```yaml
train: dataset/images/train
val: dataset/images/val

kpt_shape: [4, 2]   # using 4 keypoints: head, shoulder, mid, hip
names: ['person']
````

---

## 🎯 Training the Model

Run YOLOv8 training:

```bash
yolo pose train model=yolov8n-pose.pt data=data.yaml epochs=50 imgsz=640
```

Weights will be saved at:

```
runs/pose/train*/weights/best.pt
```

Use this file in the backend.

---

## 🧰 Tech Stack

* **YOLOv8 Pose**
* **Python 3.11**
* **FastAPI**
* **Uvicorn**
* **OpenCV + NumPy**
* **OpenRouter API + Meta LLaMA**
* **dotenv, requests, python-multipart**

---

## 📁 Project Structure

```
PostureDetection/
│── main.py
│── requirements.txt
│── .env.example
│── README.md
│── output/                # saved annotated images
│── model/
│     └── best.pt          # your trained YOLOv8 pose model
│── dataset/
      ├── images/train
      ├── images/val
      ├── labels/train
      └── labels/val
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Create `.env`:

```
OPENROUTER_API_KEY=sk-yourkeyhere
```

Update path in `main.py`:

```python
POSE_MODEL_PATH = "model/best.pt"
```

Run server:

```bash
uvicorn main:app --reload
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

---

## 🔥 API Usage

### 🧍‍♂️ POST /posture

Upload an image → get:

* posture_metrics
* first_aid JSON
* analysis_text
* annotated image URL

Example (in Swagger):
Select file → execute.

Response sample:

```json
{
  "posture_metrics": {...},
  "first_aid": {...},
  "analysis_text": "• Your neck is slightly forward...",
  "image_url": "/output/medhya_123456.jpg"
}
```

---

### 💬 POST /chat

General personal doctor assistant (LLM powered):

```json
{
  "user_message": "I have chest discomfort"
}
```

---

## 🛠 Common Issues

**1. Unicodeescape error (Windows paths)**
Fix by using forward slashes:

```python
POSE_MODEL_PATH = "C:/Users/.../best.pt"
```

**2. No keypoints detected**
Ensure full body visible from head to hips.

**3. LLM returns messy output**
The system prompt is strict — the fallback mode still returns readable text.

**4. Slow inference on CPU**
Use smaller images; batch size = 1.

---

## 📈 Future Improvements

* Training on a **bigger, higher-quality posture dataset**
* Adding **real-time webcam mode**
* Android/iOS app integration
* More keypoints → full skeletal posture
* Deploy as a cloud API on Render / Railway / HF Spaces

---

## 📝 License

MIT License.
Feel free to fork and build on top of this.

---

## 🤝 Acknowledgements

* Kaggle dataset creators
* Ultralytics YOLO team
* OpenRouter for open LLM access

---

**Thanks for exploring Medhya — more upgrades coming soon!**

```

```

