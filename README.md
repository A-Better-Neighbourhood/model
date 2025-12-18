\# 🚗 Pothole Detection System



YOLOv8-based pothole detection API deployed on Hugging Face.



\## 🎯 Features



\- ✅ Real-time pothole detection

\- ✅ FastAPI REST API

\- ✅ Trained on multiple datasets (potholes + negative examples)

\- ✅ Deployed on Hugging Face Spaces

\- ✅ Prevents false positives (faces, buildings, etc.)



\## 🚀 Live API



\*\*API Endpoint:\*\* https://karan20p-pothole-api.hf.space



\*\*Interactive Docs:\*\* https://karan20p-pothole-api.hf.space/docs



\## 🛠️ Project Structure

```

├── training/          # Model training code (Google Colab)

├── api/               # FastAPI deployment code

├── models/            # Trained model info (download separately)

└── README.md          # This file

```



\## 📚 Datasets Used



1\. Custom Roboflow dataset

2\. Kaggle Pothole Detection Dataset

3\. Annotated Potholes Dataset

4\. Clean Roads (negative examples)

5\. Face Images (negative examples)



\## 🏋️ Training



See `training/train\_model.ipynb` for complete training code.



\*\*Requirements:\*\*

\- Google Colab with GPU

\- Roboflow API key

\- Kaggle API credentials



\## 🌐 API Usage

```python

import requests



url = "https://karan20p-pothole-api.hf.space/predict"

files = {"file": open("road\_image.jpg", "rb")}

response = requests.post(url, files=files)

print(response.json())

```



\## 📦 Local Development

```bash

cd api/

pip install -r requirements.txt

python app.py

```



\## 🚀 Deployment



Deployed on Hugging Face Spaces using Docker.



See `api/` folder for deployment configuration.





\## 🙏 Acknowledgments



\- YOLOv8 by Ultralytics

\- Roboflow for dataset management

\- Hugging Face for free hosting

