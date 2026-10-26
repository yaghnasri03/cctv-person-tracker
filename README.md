# CCTV Person Tracker



An AI-powered CCTV video analysis application that detects people in videos and helps identify a person using a reference image.



## Technologies Used



**Frontend**



* React

* TypeScript

* Vite

* Tailwind CSS



**Backend**



* Python

* FastAPI

* OpenCV

* YOLOv8

* InsightFace

* ONNX Runtime

* PyTorch and Torchvision

* FFmpeg



## Project Structure



```text

cctv-person-tracker/

├── client/                 # React frontend

├── server/                 # Python backend

│   ├── app/

│   │   ├── main.py

│   │   └── routes/

│   ├── video_processor.py

│   ├── person_matcher.py

│   └── requirements.txt

├── .gitignore

└── README.md

```



## Prerequisites



Install these tools before running the project:



* Python

* Node.js and npm

* Git

* FFmpeg



FFmpeg must be installed and available through your system PATH. Verify its installation with:



```bash

ffmpeg -version

```



## Backend Setup



Open Git Bash in the project directory and run:



```bash

cd server

python -m venv venv

source venv/Scripts/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Start the backend:



```bash

uvicorn app.main:app --reload

```



Backend URL: http://127.0.0.1:8000



API documentation: http://127.0.0.1:8000/docs



**Important:** Run the backend commands from inside the `server` directory.



## Frontend Setup



Open a second terminal in the project directory:



```bash

cd client

npm install

npm run dev

```



Open the local URL printed by Vite, usually:



http://localhost:5173



Keep both the backend and frontend running in separate terminals.



## Model Files



Large model files are excluded from GitHub.



Before running detection on another computer, ensure the required model files are available locally. Check the model paths in `video_processor.py` and `person_matcher.py` and obtain the necessary files from their legitimate sources.



## Notes



* Uploaded videos and generated output files are not intended to be committed to GitHub.

* Virtual environments and Node.js dependencies should be installed separately on each computer.

* FFmpeg must be installed on every computer that runs the backend.

* The frontend and backend must both be running for the application to work.



