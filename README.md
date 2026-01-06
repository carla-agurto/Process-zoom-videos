# **Processing Zoom videos - Acoustic Feature Extraction**
----

This repo contains two main scripts and an auxiliar script to calculate two different type of features.

## 1. Hand crafted features (OpenSmile features)
To generate csv files for the interviews (expected m4a files from zoom interviews)

```
 python3 acousticserver_pipeline.py path2inputfolder path2tempfolder path2outputfolder

```

Note: Once the script is executed, the files in the input folder and intermediate folder are deleted once the features are extracted. 

   
## 2.  Embeddings from whisper features

It was tested in python environment 3.8
It requires the ffmpeg library and to install the following libraries:

pip install -U openai-whisper
pip install pandas
pip install webrtcvad
pip install pydub

To generate csv files for the interviews, run the following line.

```
 python3 acousticserver_whisper_pipeline.py path2inputfolder path2tempfolder path2outputfolder

```
Note: Once the script is executed, the files in the input folder and intermediate folder are deleted once the features are extracted. 
