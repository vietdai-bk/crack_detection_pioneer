# Surface Crack Detection with a Wall-Climbing Robot
## 📌 Introduction

This project focuses on surface crack detection using a **wall-climbing robot** equipped with deep learning-based computer vision models. The system aims to automate structural health monitoring by detecting cracks on walls and other surfaces in real-time. It utilizes advanced object detection models such as **YOLO** and **NanoDet**, ensuring accurate and efficient crack identification for various industrial and infrastructural applications.

## 🎥 Demo Video  

[![Watch the video](https://img.youtube.com/vi/RXseryDBY3o/maxresdefault.jpg)](https://youtu.be/RXseryDBY3o)


## 📖 Table of Contents | Mục lục  

1. [📌 Introduction](#introduction)  
2. [⚙️ Installation](#installation)  
3. [📊 Training](#training)  
4. [📝 Model Evaluation](#model-evaluation)  
5. [🍓 Deploy on Raspberry Pi](#deploy-on-raspberry-pi)  
6. [🌐 API & User Interface](#api--user-interface--api)  
7. [📷 Results & Visualization](#results--visualization)  
8. [🏗️ Use Cases](#use-cases)  
9. [🤝 Contributing & Development](#contributing--development)  
10. [🙏 Acknowledgements](#acknowledgements)

## ⚙️ Installation  

To get started, install the necessary dependencies:  
```bash
pip install -r requirements.txt
```
The requirements.txt file includes the following dependencies:

```
numpy==1.26.4
opencv-python==4.10.0.84
torch==2.5.1+cu124
torchvision==0.20.1+cu124
matplotlib==3.10.0
ultralytics==8.3.59
gradio==5.17.0
tkinter
fastapi==0.115.8
uvicorn==0.34.0
```

### 📥 Installing NanoDet
Since this project uses NanoDet for crack detection, follow these steps to install it:
```
cd nanodet_clone
pip install -r requirements.txt
```
Compile and install NanoDet:
```
python setup.py develop
```
Verify the installation:
```
python tools/demo.py --config config/nanodet-m.yml --model model/nanodet_m.pth --path demo.jpg
```
If everything is set up correctly, you should see the crack detection result on demo.jpg.

## 📊 Training

This project supports training crack detection models using **YOLO** and **NanoDet**.  

***🏋️ Training YOLO Model***  

1️⃣ **Prepare the dataset** (ensure it's in YOLO format).  
2️⃣ **Modify the configuration file** (`data.yaml`) to specify dataset paths.  
3️⃣ **Start training** using the following command:
```bash
yolo train model=yolo11n.pt data=data.yaml epochs=300 imgsz=640 batch=32
```

***🏋️ Training NanoDet Model***

1️⃣ **Prepare the dataset** in COCO format or YOLO format.  
2️⃣ **Modify config/nanodet-m.yml** to match dataset paths.  
3️⃣ **Start training** with the following command:
```bash
python tools/train.py --config config/nanodet-m.yml --model model/nanodet_m.pth
```

***You can adjust the parameters for training.
Here, we use the dataset: [Crack Finder Dataset.](https://universe.roboflow.com/senior-design-1a3ye/crack-finder-bbzjj)***

***To simplify training, use the provided Jupyter Notebook:***

📌 [YOLO Training Notebook](https://colab.research.google.com/drive/1Ej5IIaYoZiq8G-XJvp8dq5Mi7aCIjczI?usp=sharing)

📌 [NanoDet Training Notebook](https://colab.research.google.com/drive/1SvJGn4m753MA_dJ-p5BUQQW3me3G3xrP?usp=sharing)

## 📝 Model Evaluation

After training, evaluate the performance of the YOLO and NanoDet models on the test dataset.

**You can also use the trained models available: [Here](https://drive.google.com/drive/folders/1MNZy7GY8FniXjPfr0P753e-TPtTk2l-E)**

***🏆 Evaluating YOLO Model***
```
yolo val model=yolo11n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=0
```
***🏆 Evaluating NanoDet Model***
```
python tools/test.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
```
***🚀 FPS Performance Testing***

To measure the inference speed (FPS - Frames Per Second) of each model, use the following scripts:

⚡ Testing YOLO FPS
```
python test_fps_yolo.py
```
This script will measure how fast the YOLO model processes images in real time.

⚡ Testing NanoDet FPS
```
python test_fps_nanodet.py
```
This script evaluates the FPS of the NanoDet model on the given hardware.

**📌 Note: FPS results depend on the hardware specifications and the model size. Running on a CPU will be slower compared to a GPU.**

## 🍓 Deploy on Raspberry Pi

This section provides steps to deploy the model on **Raspberry Pi**, including file transfer, library installation, and setting up a FastAPI service.

***📂 1. Transfer Files to Raspberry Pi***  

Copy all necessary files from the `for_raspberry` folder to your Raspberry Pi:

```bash
scp -r for_raspberry pi@<raspberry_ip>:/home/pi/crack_detection
````

Replace <raspberry_ip> with your Raspberry Pi’s IP address.

***🛠 2. Install Required Libraries***
```
pip install fastapi uvicorn
```

***🔄 3. Run as a Background Service***

Create a systemd service file (fastapi_stream.service) to run FastAPI in the background:
```
[Unit]
[Unit]
Description=Crack Detection Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 -m uvicorn api_stream_for_rasp:app --host 0.0.0.0 --port 8000
WorkingDirectory=/home/pi/crack_detection
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Save this file to /etc/systemd/system/fastapi_stream.service.

***▶️ 4. Start the Service***

Enable and start the service:
```
sudo systemctl enable fastapi_stream
sudo systemctl start fastapi_stream
```

Check service status:
```
sudo systemctl status fastapi_stream
```

Now you can view the live stream from the Raspberry Pi at `http://localhost:8000/video` 🎉.

***You can deploy the model on the Raspberry Pi to run directly; however, due to budget constraints, we only have a Raspberry Pi 4 with limited processing capabilities, so we are forced to use a personal computer to run the API instead of running it directly.***

## 🌐 API & User Interface

To run the API for crack detection, execute:
```
cd api
python main.py
```

Additionally, you can run the API with the NanoDet model, execute:
```
cd api
python nanodet_api.py
```

For the GUI application, run:
```
python app_tkinter/main.py
```

or with Gradio:
```
cd gradio_app
python app.py
```

## 📷 Results & Visualization
Results from model detection:

<p align="center"> <img src="results/crack_detection_1.png" alt="Detection 1" width="300" /> <img src="results/crack_detection_2.png" alt="Detection 2" width="300" /> <img src="results/crack_detection_3.png" alt="Detection 3" width="300" /> </p>

## 🏗️ Use Cases

**Crack Detection Pioneer** is designed to enhance structural safety and maintenance in various fields. The system can be applied in multiple real-world scenarios, including:

1. **Structural Inspections for Bridges, Buildings, and Roads**  
   - Regular inspections of infrastructure to detect cracks and prevent structural failures.  
   - Identifying early-stage deterioration, allowing for timely repairs and cost savings.  
   - Assisting civil engineers in evaluating the longevity and integrity of construction materials.  

2. **Automated Monitoring in Industrial Settings**  
   - Continuous surveillance of industrial facilities such as factories, pipelines, and storage tanks.  
   - Automated alerts for potential hazards due to structural weaknesses.  
   - Integration with existing IoT-based monitoring systems for real-time analysis.  

3. **Preventative Maintenance for Critical Infrastructures**  
   - Early crack detection in power plants, transportation networks, and water supply systems.  
   - Reducing downtime and enhancing operational efficiency by predicting maintenance needs.  
   - Ensuring compliance with safety regulations in high-risk environments such as nuclear plants and offshore platforms.  

## 🤝 Contributing & Development

We welcome contributions from the open-source community to improve **Crack Detection Pioneer**. If you're interested in contributing, follow these steps:

### 🔧 Steps to Contribute:
1. **Fork** the repository to your GitHub account.  
2. **Clone** the repository to your local machine:  
   ```bash
   git clone https://github.com/vietdai-bk/crack_detection_pioneer.git
   ```
3. **Create** a new branch for your feature or fix:
4. **Make modifications** and commit your changes:
5. **Push changes** to your forked repository:
6. **Open a Pull Request** on the original repository and describe your changes.

### 🎯 Areas for Contribution

We welcome contributions in various aspects of the project, including but not limited to:

- **Improving model accuracy and inference speed.**  
- **Optimizing deployment on low-power devices like Raspberry Pi.**  
- **Enhancing the web-based UI for better user experience.**  
- **Expanding dataset diversity for better generalization.**  
- **Writing better documentation and example notebooks.**  

We appreciate all contributions, whether it’s fixing a typo or implementing a major feature.  
Feel free to **open an issue** to discuss ideas before submitting a **pull request**! 🚀

## 🙏 Acknowledgements  

We sincerely appreciate the support and contributions that made this project possible.  

- **Open-Source Community:** This project leverages various open-source libraries and tools, and we are grateful for the developers who maintain them.  
- **Contributors:** A special thanks to everyone who has provided feedback, reported issues, or helped improve the project.  
- **Early Testers:** Your insights and suggestions have been invaluable in refining the software.  

This is a small but growing project, and every contribution counts. Thank you for being a part of it! 🚀

## 📜 License & Legal Information  

### 🔹 Project License  
This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for more details.  

### 🔹 Third-Party Licenses  
This project uses the following third-party libraries, which have their respective licenses:  

- **YOLOv11 (Ultralytics)** - [GPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)  
- **NanoDet** - [Apache 2.0 License](https://github.com/RangiLyu/nanodet/blob/main/LICENSE)  
- **FastAPI** - [MIT License](https://github.com/tiangolo/fastapi/blob/master/LICENSE)  
- **OpenCV** - [Apache 2.0 License](https://github.com/opencv/opencv/blob/master/LICENSE)  

Please review their individual licenses before use.  

## 📚 Dataset Source & Citation  

This project utilizes the **Crack Finder Dataset**, provided by **Senior Design** and available on **Roboflow Universe**.  

- **Dataset Link:** [Crack Finder Dataset](https://universe.roboflow.com/senior-design-1a3ye/crack-finder-bbzjj)  
- **License:** Please refer to the dataset's page for license details.  

If you use this dataset in your own research or project, please cite the original authors as follows:  

```bibtex
@misc{
    crack-finder-bbzjj_dataset,
    title = { Crack Finder Dataset },
    type = { Open Source Dataset },
    author = { Senior Design },
    howpublished = { \url{ https://universe.roboflow.com/senior-design-1a3ye/crack-finder-bbzjj } },
    url = { https://universe.roboflow.com/senior-design-1a3ye/crack-finder-bbzjj },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
    month = { nov },
    note = { visited on 2025-03-01 },
}
```

---

## 📢 Disclaimer  
This software is provided "as is" without warranty of any kind. The authors are not responsible for any damages resulting from the use of this software.  

