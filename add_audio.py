import pyaudio
import numpy as np

import pymongo
import json
from pymongo import MongoClient


import numpy as np
import librosa
import soundfile as sf

from panns_inference import AudioTagging

from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()

# Replace the connection string with your own.
# https://ioflood.com/blog/python-dotenv-guide-how-to-use-environment-variables-in-python/
connection_string = os.getenv('MONGO_CONNECTION_STRING')

# Create a MongoClient object
client = MongoClient(connection_string)


# Instantiate the MongoDB collection
db = client['audio']
mongodb_sounds_collection = db['sounds']
mongodb_results_collection = db['results']



# load the default model into the gpu.
model = AudioTagging(checkpoint_path=None, device='cuda') # change device to cpu if a gpu is not available



# Define the number of seconds to record
RECORD_SECONDS = 1 #0.5

# Define the sample rate and chunk size
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

# Function to normalize a vector. Normalizing a vector means adjusting the values measured in different scales to a common scale.
def normalize(v):
    # np.linalg.norm computes the vector's norm (magnitude). The norm is the total length of all vectors in a space.
    norm = np.linalg.norm(v)
    if norm == 0:
            return v

    # Return the normalized vector.
    return v / norm

# Function to get an embedding of an audio file. An embedding is a reduced-dimensionality representation of the file.
def get_embedding (audio_data):

    # Load the audio file using librosa's load function, which returns an audio time series and its corresponding sample rate.
    # a, _ = librosa.load(audio_file, sr=44100)
    # a, _ = sf.read(audio_data)

    # Convert the input list to a NumPy array
    input_array = np.array(audio_data, dtype=np.int16)

    # Scale the input array between -1 and 1
    # scaled_array = np.interp(input_array, (input_array.min(), input_array.max()), (-1, 1))
    scaled_array = np.interp(input_array, (-32768, 32767), (-1, 1))
    a = scaled_array

    # Reshape the audio time series to have an extra dimension, which is required by the model's inference function.
    query_audio = a[None, :]

    # Perform inference on the reshaped audio using the model. This returns an embedding of the audio.
    _, emb = model.inference(query_audio)

    # Normalize the embedding. This scales the embedding to have a length (magnitude) of 1, while maintaining its direction.
    normalized_v = normalize(emb[0])

    # Return the normalized embedding required for dot_product elastic similarity dense vector
    return normalized_v

def insert_mongo_results(results, mongodb_results_collection):
    # Create the results document
    entry = {"sensor":"Ralph's laptop","data_time":datetime.now(),"results":results}

    mongodb_results_collection.insert_one(entry)


def insert_mongo_sounds(audio_name,embedding,audio_file,image, mongodb_sounds_collection):
    # Create the results document
    entry = {"audio":audio_name,"emb":embedding,"audio_file":audio_file,"image":image}

    mongodb_sounds_collection.insert_one(entry)



#def knnbeta_search(embedding, mongodb_sounds_collection):
def vector_search(embedding, mongodb_sounds_collection):
    # Create the query vector
    query_vector = embedding.tolist()

    # Create the search query
#    search_query = [
#        {
#            "$search": {
#                "knnBeta": {
#                    "vector": query_vector,
#                    "path": "emb",
#                    "k": 3
#                }
#            }
#        },
#        {
#            "$project": {
#            "_id": 0,
#            "audio": 1,
#            "image": 1,
#            "audio_file": 1,
#            "score": { "$meta": "searchScore" }
#            }
#        }
#    ]

    search_query = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "emb",
                "queryVector": query_vector,
                "numCandidates": 150,
                "limit": 10
             }
        },
        {
            "$project": {
            "audio": 1,
            "image": 1,
            "audio_file": 1,
            "score": { "$meta": "vectorSearchScore"}
            }
        }
    ]

    # Perform the search query
    results = mongodb_sounds_collection.aggregate(search_query)

    return results

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
num_devices = info.get('deviceCount')

for i in range(num_devices):
    device_info = p.get_device_info_by_host_api_device_index(0, i)
    if device_info['maxInputChannels'] > 0:
        print(f"Device {i}: {device_info['name']}")

input_device = input("Which input device do you want to use?")

print(input_device)

# Initialize PyAudio
audio = pyaudio.PyAudio()

audio_description_dictionary = [
    {
      "audio": "Not running",
      "image": "https://media.tenor.com/dOAWtNUHo8sAAAAC/running-is-impossible-work-out.gif"
    },
    {
      "audio": "Running",
      "image": "https://c.tenor.com/SGujcD45Mj4AAAAd/tenor.gif"
    },
    {
      "audio": "Small hit",
      "image": "https://media.tenor.com/immNDTeuzREAAAAC/lebron-james-lebron.gif"
    },
    {
      "audio": "Big hit",
      "image": "https://media.tenor.com/nD5cv8oodb8AAAAC/reptar-city.gif"
    },
    {
      "audio": "Windy",
      "image": "https://media.tenor.com/DAFlmuvQi4kAAAAC/wind-windy.gif"
    }
  ]

for audio_description in audio_description_dictionary:
    input(f"Get ready to record '{audio_description['audio']}' - press enter when ready")

    # Open the microphone stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE,input_device_index=int(input_device))

    for j in range(1):
        stream.start_stream()
        # Record audio data from the microphone
        frames = []
        for i in range(0, int((SAMPLE_RATE / CHUNK_SIZE) * RECORD_SECONDS)):
            data = stream.read(CHUNK_SIZE)
            frames.append(data)
        # Stop the microphone stream
        stream.stop_stream()

        # Convert the recorded audio data into a NumPy array
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

        emb = get_embedding(audio_data)

        insert_mongo_sounds(f"{audio_description['audio']}",emb.tolist(),f"{j}",f"{audio_description['image']}",mongodb_sounds_collection)
    stream.close()
    print('Recorded successfully')


audio.terminate()
