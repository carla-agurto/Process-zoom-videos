#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:03:46 2024

@author: carlaagurto
"""

import os
import sys
import time
import argparse
from pathlib import Path
import glob
import shutil
import pandas as pd
import time
import os
from detectfirst_whisperembeddings import compare_two_files, extract_whisper_features_df

 
"""
INPUT VARIABLES
"""

"""
Definicion
"""
def snooze(snooze_time):
    """
    Function to be called to snooze for a while until new files are available.
    Parameters
    ----------
    snooze_time : int
        Time to snooze in seconds (unless there is a keyboard interrupt).
    Returns
    -------
    None.
    """
    
    print(f'No files available yet. Snoozing for {int(snooze_time/60)}'
          ' minutes')
    
    try:
        time.sleep(snooze_time)
    except KeyboardInterrupt:
        try:
            print('\nSnooze interrupted by user. Push Ctrl+C again to exit')
            time.sleep(5)
        except KeyboardInterrupt:
            print('\nExiting...')
            sys.exit(0)
        print('Resuming processing...')
        time.sleep(2)
        
        
# def resave_audio (input_file, new_input_file):
    
#     cmd = 'ffmpeg -y -i ' + "'" + input_file + "'"  + ' -ac 2 -ar 44100 ' + "'" + new_input_file + "'"
#     os.system(cmd)    
    

  

"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':
    
    # Set up parser and retrieve its fields
    parser =\
        argparse.ArgumentParser(description='Script to run acosutic whisper feature '
                                'extraction from video interviews.')
    parser.add_argument(dest='read_dir', type=str,
                        help=('Path where audios to be analyzed by this'
                              ' script are stored'))
    parser.add_argument(dest='store_dir', type=str,
                        help=('Path where spreadsheets with extracted video'
                              ' features and processing time reports will be'
                              ' stored'))
    parser.add_argument(dest='final_dir', type=str,
                        help=('Path where '))
    # parser.add_argument(dest='logs_dir', type=str,
    #                     help=('Path where log file will be stored'))    
    # parser.add_argument('--snooze', type=int, required=False, default=5, 
    #                     help=('Time to snooze (in minutes) when there are no'
    #                           ' video files available at <read_dir>'))
    
    args = parser.parse_args()
    data_dir = Path(args.read_dir)
    store_dir = Path(args.store_dir)
    final_dir2 = Path(args.final_dir)
    #logs_dir = Path(args.logs_dir)
    #snooze_time = (args.snooze) * 60
    path2results = store_dir
    
    
    log_df = pd.DataFrame(index=['1'], columns=['Error'])
   
 # Forever loop: check if files available; otherwise, sleep

#To avoid errors with the name we are going to re-save the audio files


        
    #In this part we need to check the directories in the audios_dir
        
    list_folders = [x[0] for x in os.walk(data_dir)]
    list_folders.remove(list_folders[0])
    
    if len(list_folders)== 0:
        print('done')
        #utl.snooze(snooze_time)
    
    else:
        for path2file in list_folders:
            
            #Get the name for saving the features from the folder
            sufix_name = os.path.basename(path2file)
            sufix_name1 = sufix_name.split('-')[0]
            sufix_name2 = sufix_name.split('-')[1]
            sufix_name3 = sufix_name.split('-')[2]
            sufix_name4 = sufix_name.split('-')[3]

            #create a folder with the sufix name
            print(sufix_name)
            
            os.mkdir(os.path.join(final_dir2, sufix_name))
            #update the final dir name
            final_dir = os.path.join(final_dir2, sufix_name)
            
            try:
                initial_time = time.time()
                #Now we list the files(only m4a containing the interview)
                listfiles = [ f for f in os.listdir(path2file) if f.endswith('.m4a') ]
                
                if (len(listfiles)<3) & (len(listfiles)> 0):
          
                    #Prepare input info for docker
    
                    if len(listfiles)==2:

                        print('Carla pass here')
                        
                        #Rename files
                        
                        os.rename(os.path.join(path2file,listfiles[0] ), os.path.join(path2file,'zyfirst_audiofile.m4a'))
                        os.rename(os.path.join(path2file,listfiles[1] ), os.path.join(path2file,'zysecond_audiofile.m4a'))
                        
                        
                        file1 = os.path.join('/data', 'zyfirst_audiofile.m4a') 
                        file2 = os.path.join('/data', 'zysecond_audiofile.m4a') 
                        nfiles=2
    
                        
                        
                        print('ALLFILES:', path2file, path2results, str(nfiles), file1, file2)
                        
                        folder = path2file
                        first, second, times = compare_two_files(folder)
                        
                        #extracting whisper features
                        df_s1 = extract_whisper_features_df(os.path.join(folder, first), model_name="base", chunk_sec=30, overlap_sec=15)
                        df_s1.to_csv(os.path.join(path2results, "S1.csv"), index=False)
                        
                        
                        df_s2 = extract_whisper_features_df(os.path.join(folder, second), model_name="base", chunk_sec=30, overlap_sec=15)
                        df_s2.to_csv(os.path.join(path2results, "S2.csv"), index=False)

                          
                        #Move:
                   
                        shutil.copy(os.path.join(path2results, 'S1.csv'), \
                                        os.path.join(final_dir, sufix_name1 + '-' + sufix_name2 + '-openInterview_Acoustic-WhisperFeatures_30sEmbeddings_S1-'+ sufix_name4 + '.csv'))
                                                    
                        shutil.copy(os.path.join(path2results, 'S2.csv'), \
                                        os.path.join(final_dir, sufix_name1 + '-' + sufix_name2 + '-openInterview_Acoustic-WhisperFeatures_30sEmbeddings_S2-' + sufix_name4 + '.csv'))
                                
    
                        files = os.listdir(str(path2results))
                        for f in files:
                            os.remove(os.path.join(path2results,f))
                        
                            
                           
                    else: #len files = 1
                        
                        #extracting whisper features
                        df_s1 = extract_whisper_features_df(os.path.join(folder, first), model_name="base", chunk_sec=30, overlap_sec=15)
                        df_s1.to_csv("S1.csv", index=False)
                       
    
                        shutil.copy(os.path.join(path2results, 'S1.csv'), \
                                        os.path.join(final_dir, sufix_name1 + '-' + sufix_name2 + '-openInterview_Acoustic-WhisperFeatures_30sEmbeddings_S1-'+ sufix_name4 + '.csv'))
                      
                        
                            
        
                        files = os.listdir(str(path2results))
                        for f in files:
                            os.remove(os.path.join(path2results,f))
                            
                    log_df.loc['1', 'Naudio_files'] = len(listfiles)
                    
                    
                else:
                    log_df.loc['1','Naudio_files'] = len(listfiles)
                    
                final_time = time.time()
                    
                log_df.loc['1', 'Error'] = 'No error'
                log_df.loc['1', 'Processing_time_seconds'] = final_time-initial_time; print(log_df)
                
                
                #Once processed, delete the folder
                #shutil.rmtree(path2file)
                log_df.to_csv(os.path.join(final_dir, sufix_name + 'LogFile.csv'), index=False)
                shutil.rmtree(path2file)
                
            except Exception as e:
                print('Error',e)
                #do not delete the folder
                log_df.loc['1', 'Error'] = e
                log_df.loc['1', 'Processing_time_seconds'] = 'Incomplete'
                log_df.to_csv(os.path.join(final_dir, sufix_name + 'LogFile.csv'), index = False)
                
                
                
        
                    
                        
            
