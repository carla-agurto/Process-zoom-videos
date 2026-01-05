**Processing Zoom videos - Acoustic Feature Extraction**
----
This repo contain two main scripts to calculate two different type of features.

1. Hand crafted features (OpenSmile features)
To generate csv files for the interviews (expected m4a files from zoom interviews)

```
 python3 acousticserver_pipeline.py path2inputfolder path2tempfolder path2outputfolder

```

Note: Once the script is executed, the files in the input folder and intermediate folder are deleted once the features are extracted. 
   
2.  Embedding from whisper features



**Instructions**
----
