# Wind Turbine Diagnostics using MongoDB Vector Search

A tongue-in-cheek demonstration of MongoDB's Vector Search capabilities for anomaly detection through sound input, using a basic handheld fan as our makeshift wind turbine. This demo enables real-time diagnosis by analyzing the emitted audio, allowing us to diagnose its condition—whether it's operating normally, stopped, or experiencing any issues.

![image](https://github.com/user-attachments/assets/9cf424df-d513-4ee3-a417-31c3dccc0ce3)
    
## Prerequisites
Install Node.js (Tested with Node.js v20.8.0)
    
## 1. MongoDB Atlas Connection
Create a file called ```.env``` in the main directory alongside the ```add_audio.py``` file and add your Atlas connection string, in the following format:
```
MONGO_CONNECTION_STRING="mongodb+srv://connectionstringfromatlas"
```
Then copy this file in to the ```nodeUI``` directory too.

## 2. Install Python Modules

Run ```python3 -m venv path/to/venv```
Run ```source path/to/venv/bin/activate```

> [!NOTE]
> Be sure to change the path in the command above.

Install the required python modules.
```
python3 -m pip install -r requirements.txt
```

> [!NOTE]
> If you get a problem when running ```pip install pyaudio``` run ```brew install portaudio```.

## 3. Record Audio Files
Run ```python3 add_audio.py```

Select the audio input by typing the relevant number and then press enter. Record each sound in sequence.

> [!NOTE]
> If you get a problem when running ```python add_audio.py``` run ```brew install wget```.

> [!TIP]
> We recommend using an external microphone and placing it very close to the fan or audio source.

## 4. Create a Search Index
Go to MongoDB Atlas and create an Atlas Search Index in the **audio** database **sounds** collection and using the content of searchindex.json
```
{
  "fields": [
    {
      "numDimensions": 2048,
      "path": "emb",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
``` 
## 5. Query the Database
Run ```python3 live_query.py``` and place your microphone next to the fan.

## 6. Run the Frontend
Switch to a new console and cd to the ```nodeUI``` directory.

Run ```npm install```

Run ```node nodeui.js```

Use a browser to open the link http://localhost:3000/

## 7. Atlas Charts
In Atlas, go to charts and click the down arrow next to **Add Dashboard** then click **Import Dashboard**.

Select the file ```Sounds.charts``` and click next.

Click on the pencil icon and ensure the database and collection match **audio** and **results**.

Click 'Save', and then 'Save'.

Click the new dashboard 'Sounds' to see analytics on the sounds that are being detected by the microphone.
