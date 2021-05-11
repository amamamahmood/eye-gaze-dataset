import subprocess
import pandas as pd
import multiprocessing
import sys
import pydicom
from scipy import ndimage
import cv2
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skvideo.io import FFmpegWriter
import ffmpeg
import glob
problematic_image_name =''
import json

#Replace with the location of the MIMIC-CXR images
original_folder_images='D:/github/eye-gaze-dataset/physionet.org/files/mimic-cxr/2.0.0/'


def create_videos(input_folder, eye_gaze_table, cases, data_type):
    '''
    This method is optional. It just creates videos of heatmaps using heatmap frames for particular eye_gaze_table.
    It can ONLY run after process_eye_gaze_table() method finishes.

    :param input_folder: Folder with saved heatmap frames (see process_eye_gaze_table())
    :param eye_gaze_table: Pandas dataframe containing the eye gaze data
    :param data_type: Type of eye gaze type: fixations, raw eye gaze
    :return: None
    '''

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
      
    # org
    org = (200, 200)
      
    # fontScale
    fontScale = 2
       
    # Blue color in BGR
    color = (0, 0, 255)
      
    # Line thickness of 2 px
    thickness = 5
    
    try:
        os.mkdir(input_folder)
    except:
        pass

    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir() ]


    for subfolder in subfolders:

        

        

        
        print('Subfolder\n',subfolder.split('/')[-1])
        files = glob.glob(os.path.join(subfolder,"*frame.png"))
        print(len(files))
        #Extract image name
        image_name = subfolder.split('\\')[-1]
        print(image_name)
        #Get file path to original image
        #case_index = cases.loc[cases['dicom_id'] == image_name].index[0]
        #file_path = cases.loc[case_index, 'path']
        case_index = cases.loc[cases['dicom_id'] == image_name]
        if case_index.empty:
            file_path = 'random.dcm'
        #print(case_index)
        else:
            file_path = cases.iloc[case_index.index[0]]['path']

        try:
            os.mkdir(os.path.join(input_folder,image_name))
        except:
            pass
        image = cv2.imread("image.png")

        

        

        try:
            last_row = eye_gaze_table.index[eye_gaze_table['DICOM_ID'] == image_name].tolist()[-1]
            full_time = eye_gaze_table["Time (in secs)"].values[last_row]
            rows = eye_gaze_table.index[eye_gaze_table['DICOM_ID'] == image_name].tolist()
            #fps = (len(files)) / full_time
            fps = 30
            
        except:
            print('Error with fps!')
            
        if (image.shape[0] % 2) > 0:
            image = np.vstack((image, np.zeros((1, image.shape[1], 3))))
        if (image.shape[1] % 2) > 0:
            image = np.hstack((image, np.zeros((image.shape[0], 1, 3))))


        ## Making baseline video:
        #base_heatmap = np.zeros([image.shape[0], image.shape[1]])
        #sigma = 150
        #base_heatmap = ndimage.gaussian_filter(base_heatmap, sigma)
        #print("Size:")
        #print(np.shape(base_heatmap))
        #plt.imsave(os.path.join(input_folder, 'base_heatmap.png'), base_heatmap)
        #n_channels = 4
        #text_base = np.zeros((image.shape[0], image.shape[1], n_channels), dtype=np.uint8)
        #plt.imsave(os.path.join(input_folder, 'text_base.png'), text_base)
        text_base = cv2.imread(os.path.join(input_folder,'text_base.png'))
        base_heatmap = cv2.imread(os.path.join(input_folder,'base_heatmap.png'))
        baseline = cv2.addWeighted(image.astype('uint8'), 0.5, base_heatmap.astype('uint8'), 0.5, 0)
        no_of_frames = round(fps* full_time) + fps
        print(no_of_frames)
        frame_array=[]
        text_array=[]
        plt.imsave(os.path.join(input_folder, 'baseline.png'), baseline)
        for j in range(no_of_frames):
            frame_array.append(baseline)
            text_array.append(text_base)
        crf = 23
                #vid_out = FFmpegWriter(os.path.join(input_folder, image_name, data_type+'.mp4'),
                 #                                 inputdict={'-r': str(fps),
                  #                                           '-s': '{}x{}'.format(image.shape[1], image.shape[0])},
                   #                               outputdict={'-r': str(fps), '-c:v': 'mpeg4', '-crf': str(crf),
                    #                                          '-preset': 'ultrafast',
                     #                                         '-pix_fmt': 'yuv420p'}, verbosity=0
                      #                            )
        out = cv2.VideoWriter(os.path.join(input_folder, image_name, data_type+'.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), fps, (image.shape[1], image.shape[0]))
        
        for i in range(len(rows)):
            start_frame = int(round(eye_gaze_table["FPOGS"].values[rows[i]],2)*fps)
            #print(start_frame)
            end_frame = int(round(eye_gaze_table["Time (in secs)"].values[rows[i]],2)*fps)
            #print(end_frame)
            
            try:
                overlay_heatmap = cv2.addWeighted(image.astype('uint8'), 0.5, cv2.imread(os.path.join(subfolder,str(i+1)+'_frame.png')).astype('uint8'), 0.5, 0)
            except:
                print('error ',cv2.imread(files[i]).astype('uint8').shape)
            for k in range(start_frame,end_frame):
                    frame_array[k]=overlay_heatmap
            #out.write(overlay_heatmap)
                    
        #### Reading transcript.json for audio transcriptions
        json_path = os.path.join(subfolder,'transcript.json')
        with open(json_path) as json_file:
            data = json.load(json_file)
        timestamps = data['time_stamped_text']
        phrase = ''
        for t in timestamps:
            begin_time = int(round(t['begin_time'],2)*fps)
            end_time = int(round(t['end_time'],2)*fps)
            phrase = t['phrase']
            print(begin_time, end_time,phrase)
            for r in range(begin_time, end_time):
                #print(r,phrase)
                img=text_array[r]
                cv2.rectangle(img,(150,150),(700,250),(0,0,0),cv2.FILLED)
                cv2.putText(img, phrase, org, font, 
                   fontScale, color, thickness)
                #(text_width, text_height) = cv2.getTextSize(phrase, font, fontScale, thickness)[0]
                # set the text start position
                #text_offset_x = 200
                #text_offset_y = 200
                # make the coords of the box with a small padding of two pixels
                #box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
                #cv2.rectangle(img, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
                #cv2.putText(img, phrase, (text_offset_x, text_offset_y), font, fontScale, color, thickness)
                cv2.waitKey(0)
                text_array[r] = img
                frame_array[r] = cv2.addWeighted(frame_array[r].astype('uint8'), 1, text_array[r].astype('uint8'), 1, 0)
                #plt.imsave(os.path.join(input_folder, str(r)+'_test_audio.png'), img)
                del img
            

        
        #### writing video to file

        for n in range(no_of_frames):
            out.write(frame_array[n])
        out.release()
        
        #video = ffmpeg.input(os.path.join(subfolder,'fixations.mp4'))
        #audio = ffmpeg.input(os.path.join(subfolder,'audio.wav'))
        #video_path = os.path.join(subfolder,'fixations_audio.mp4')
        #out = ffmpeg.output(video, audio, video_path, vcodec='copy', acodec='aac', strict='experimental')
        #out.run()
        #videofile = os.path.join(subfolder,'fixations.mp4')
        #audiofile = os.path.join(subfolder,'audio.wav')
        #outputfile = os.path.join(subfolder,'fixations_audio.mp4')
        #codec = "copy"
        #subprocess.run(f"ffmpeg -i {videofile} -i {audiofile} -c {codec} {outputfile}")                                  
        #print('Muxing Done')
def calibrate(eye_gaze_table,screen_width=1920, screen_height=1080):
    '''
    This method uses calibration image (read paper for more) and recalibrates coordinates

    :param gaze_table: pandas dataframe with eye gaze table
    :param screen_width: DO NOT CHANGE. This was used in the original eye gaze experiment.
    :param screen_height: DO NOT CHANGE. This was used in the original eye gaze experiment.
    :return:
    '''
    try:

        calibrationX=[]
        calibrationY=[]

        # Iterate through each image in the raw eye gaze spreadsheet
        for index, row in eye_gaze_table.iterrows():
            image_name = row['DICOM_ID']

            if os.path.exists(os.path.join(original_folder_images, image_name + '.dcm')) == False:
                last_row = eye_gaze_table.index[eye_gaze_table['DICOM_ID'] == image_name ].tolist()[-1]
                eyeX = eye_gaze_table['FPOGX'][last_row]* screen_width
                eyeY = eye_gaze_table['FPOGY'][last_row]* screen_height

                # Get pixel coordinates from raw eye gaze coordinates
                # eyeX = row['FPOGX'] * screen_width
                # eyeY = row['FPOGY'] * screen_height
                calibrationX.append(eyeX)
                calibrationY.append(eyeY)

        calibrationX = np.asarray(calibrationX)
        calibrationY = np.asarray(calibrationY)

        mean_X = np.mean(calibrationX)
        mean_Y = np.mean(calibrationY)

        calibratedX = screen_width//2 - mean_X
        calibratedY = screen_height//2 - mean_Y

        return calibratedX, calibratedY
    except:
        print('No calibration available')
        return .0,.0

def process_eye_gaze_table(session_table, export_folder, cases, window=0, calibration=False, sigma = 150, screen_width=1920, screen_height=1080):
    '''
    Main method to process eye gaze session table (e.g. fixations or raw eye gaze) to create heatmap frames for each coordinate.
    The frames are saved in export_folder/dicom_id
    It returns the same session table with:
        a) its eye gaze coordinates (i.e. FPOGX, FPOGY) mapped to image coordinates (i.e. X_ORIGINAL, Y_ORIGINAL).
        b) each row's heatmap frame (i.e. EYE_GAZE_HEATMAP_FRAME)

    This method also allows the user to do the following too:

    - Re-calibrate coordinates (i.e. FPOGX, FPOGY) by utilizing the calibration template (i.e. calibration_image.png)
    if available in this particular session
    - Use exponential decay as a weight in a specific window (i.e. +- heatmap frames on a given heatmap frame) for the given heatmap frame
    - Apply different sigma size when generating heatmap frames

    :param session_table: a fixation or raw eye gaze Pandas dataframe for a particular session
    :param export_folder: folder to save heatmap frames
    :param cases: the original master sheet in Pandas dataframe
    :param window: number of frames to use when applying exponential decay on a given heatmap fram
    :param calibration: flag to perform re-calibration
    :param original_folder_images: location of original dicom images downloaded from MIMIC source
    :param sigma: sigma of gaussian to apply on a given eye gaze point
    :param screen_width: screen width in pixels for monitor's resolution used in experiment. DO NOT CHANGE!
    :param screen_height: screen height in pixels for monitor's resolution used in experiment. DO NOT CHANGE!
    :return: processed session table with eye gaze coordinated mapped to original image coordinates
    '''

    session_table["X_ORIGINAL"] = ""
    session_table["Y_ORIGINAL"] = ""
    session_table["EYE_GAZE_HEATMAP_FRAME"] = ""
    previous_image_name = ''

    heatmaps = []
    counter = 1

    #Do calibration
    if calibration:
        calibratedX, calibratedY = calibrate(session_table)
    else:
        calibratedX=calibratedY=.0
    #Iterate through each image in the raw eye gaze spreadsheet
    for index, row in session_table.iterrows():
        #Get pixel coordinates from raw eye gaze coordinates and calibrate them
        eyeX = row['FPOGX']*screen_width + calibratedX
        eyeY = row['FPOGY']*screen_height + calibratedY

        #Get image name
        # print(row)
        image_name = row['DICOM_ID']
        #Get file path
        case_index = cases.loc[cases['dicom_id'] == image_name]
        if case_index.empty:
            file_path = 'random.dcm'
        #print(case_index)
        else:
            file_path = cases.iloc[case_index.index[0]]['path']
        #print(file_path)
        # print(image_name, index, session_table.shape[0], previous_image_name, image_name)
        #Condition to start a new eye gaze drawing job
        if previous_image_name != image_name:

            counter = 1

            if previous_image_name != '':

                print('Finished ', index, '/' ,session_table.shape[0], ' rows from session ',session_table['MEDIA_ID'].values[0])

                for i in range(len(heatmaps)):
                    if window != 0:
                        left_window=right_window=window
                        if i - window<0:
                            left_window = i
                        if i + window>len(heatmaps):
                            right_window = len(heatmaps)-i

                        for j in range(i-left_window,i+right_window):
                            # Use exponential decay relative to length of existing observed eye gaze
                            decay =  math.exp(-abs(i - j))
                            heatmaps[j] *= decay
                        heatmap_numpy = heatmaps[i-left_window:i+right_window]
                        current_heatmap = np.sum(heatmap_numpy, axis=0)
                    else:
                        current_heatmap = heatmaps[i]

                    plt.imsave(os.path.join(export_folder, previous_image_name, str(i) + '_frame.png'),
                                            ndimage.gaussian_filter(current_heatmap, sigma))

                heatmap = ndimage.gaussian_filter(record, sigma)

                try:
                    os.mkdir(os.path.join(export_folder, previous_image_name))
                except:
                    pass
                plt.imsave(os.path.join(export_folder, previous_image_name,'heatmap.png'), heatmap)
                ### trying to save heatmap over image
                try:
                    #Load dicom image
                    case_index2 = cases.loc[cases['dicom_id'] == previous_image_name].index[0]
                    file_path2 = cases.iloc[case_index2]['path']
                    ds = pydicom.dcmread(os.path.join(original_folder_images,file_path2))
                    image = ds.pixel_array.copy().astype(np.float)
                    image /= np.max(image)
                    image *= 255.
                    image = image.astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                except:
                    #else it is a calibration image
                    #print('not found')
                    image = cv2.imread('calibration_image.png').astype('uint8')
                plt.imsave(os.path.join(export_folder, previous_image_name,'image.png'), image)
                image1 = cv2.imread(os.path.join(export_folder, previous_image_name,'image.png'))
                image2 = cv2.imread(os.path.join(export_folder, previous_image_name,'heatmap.png'))
                try:
                    overlay_heatmap = cv2.addWeighted(image1.astype('uint8'), 0.5, image2.astype('uint8'), 0.5, 0.0)
                    plt.imsave(os.path.join(export_folder, previous_image_name,'heatmap_overlay.png'), overlay_heatmap)
                except:
                    pass
                ########
                
                heatmaps = []
                del(current_heatmap)

            if not os.path.exists(os.path.join(export_folder, image_name)):
                os.mkdir(os.path.join(export_folder, image_name))


            if os.path.exists(os.path.join(original_folder_images, file_path)) == True:
                ds = pydicom.dcmread(os.path.join(original_folder_images, file_path))

                image = ds.pixel_array.copy().astype(np.float)
                image /= np.max(image)
                image *= 255.
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                # Load metadata (top, bottom, left, right padding pixel dimensions) about the particular dicom image from the master spreadsheet
                case_index = cases.loc[cases['dicom_id'] == image_name].index[0]
                top, bottom, left, right = cases.iloc[case_index]['image_top_pad'], cases.iloc[case_index]['image_bottom_pad'], \
                                              cases.iloc[case_index]['image_left_pad'], cases.iloc[case_index]['image_right_pad']

            else:
                image = np.zeros((screen_height, screen_width,3), dtype=np.uint8)
                top, bottom, left, right = (0, 0, 0, 0)
            if (image.shape[0]%2)>0:
                image = np.vstack((image, np.zeros((1, image.shape[1], 3))))
            if (image.shape[1]%2)>0:
                image = np.hstack((image, np.zeros((image.shape[0], 1, 3))))

            record = np.zeros([image.shape[0], image.shape[1]])
            previous_image_name = image_name

        try:

            #Keep eye gazes that fall within the image
            if eyeX > left and eyeX < screen_width-right and eyeY> top and eyeY < screen_height-bottom:
                x_original = eyeX - left
                y_original = eyeY - top
            else:
                x_original = -1
                y_original = -1

            #Remap to original image coordinates
            resized_width = screen_width - left - right
            resized_height = screen_height - top - bottom
            x_original_in_pixels = int((image.shape[1]/resized_width) * x_original)
            y_original_in_pixels = int((image.shape[0]/resized_height) * y_original)


            #Create heatmap
            heatmap_image = np.zeros([image.shape[0], image.shape[1]])
            if y_original_in_pixels>0:
                record[int(y_original_in_pixels), int(x_original_in_pixels)] += 1
                heatmap_image[int(y_original_in_pixels), int(x_original_in_pixels)] = 1
            heatmaps.append(heatmap_image)

            #Also save eye gazes coordinates to the spreadsheet
            session_table.loc[index,"X_ORIGINAL"]=x_original_in_pixels
            session_table.loc[index,"Y_ORIGINAL"]=y_original_in_pixels
            session_table.loc[index,"EYE_GAZE_HEATMAP_FRAME"] = str(counter) + '_frame.png'
            counter +=1

        except:
            print(sys.exc_info()[0])

    return session_table

def concatenate_session_tables(eye_gaze_session_tables):
    '''
    Auxilary method that simply concatenates each individual session eye gaze table into a single table
    :param tables: List of Pandas dataframes of session eye gaze tables
    :return:
    '''
    final_table = []

    for i, table in enumerate(eye_gaze_session_tables):
        if i == 0:
            n_columns = len(table.columns)
            columns = table.columns
            table.columns = range(n_columns)
            final_table = table
        else:
            table.columns = range(n_columns)
            final_table = pd.concat([final_table, table], axis=0,ignore_index=True,sort=False)
    final_table.columns=columns
    return final_table

def process_fixations(experiment_name, video=False):

    print('--------> FIXATIONS <--------')

    cases = pd.read_csv('../../physionet.org/files/egd-cxr/1.0.0/master_sheet.csv')
    table = pd.read_csv('../../physionet.org/files/egd-cxr/1.0.0/fixations.csv')
    table = table.loc[table['SESSION_ID']==1]
    #table = table.loc[table['MEDIA_ID'] >=12]
    #table = table.loc[table['SESSION_ID'].between(2,5)]
    #table.drop('SESSION_ID', inplace=True, axis=1)
    print(table)
    sessions = table.groupby(['SESSION_ID'])
    
    #m = cases.dicom_id.isin(table.DICOM_ID)
    #print('mmm')
    #print(m)
    #cases = cases[m]
    #print(cases)
    #sessions = table.groupby(['SESSION_ID'])

    
    print('length of sessions')
    print(len(sessions))
    try:
        os.makedirs(experiment_name, exist_ok=True)
    except OSError as exc:
        print(exc, ' Proceeding...')
        pass

    p = multiprocessing.Pool(processes=len(sessions))
    objects = []
    for session in sessions:
        df = session[1].copy().reset_index(drop=True)
        objects.append((df, experiment_name, cases))
    eye_gaze_session_tables = p.starmap(process_eye_gaze_table, [i for i in objects])
    p.close()

    final_table = concatenate_session_tables(eye_gaze_session_tables)

    #Save experiment consolidated table
    final_table.to_csv(experiment_name+'.csv', index=False)
    #Create video files with fixation heatmaps
    if video==True:
        create_videos(experiment_name,final_table, cases, data_type='fixations')

def process_raw_eye_gaze(experiment_name, video=False):

    print('--------> RAW EYE GAZE <--------')

    cases = pd.read_csv('../../physionet.org/files/egd-cxr/1.0.0/master_sheet.csv')
    #cases = pd.read_csv('../../Resources/master_sheet.csv')
    table = pd.read_csv('../../physionet.org/files/egd-cxr/1.0.0/fixations.csv')
   # cases = cases.loc[cases['dicom_id'] == '1a3f39ce-ebe90275-9a66145a-af03360e-ee3b163b']
    table = table.loc[table['SESSION_ID'] == 1]
    print(table)
    sessions = table.groupby(['MEDIA_ID'])

    
    
    try:
        os.mkdir(experiment_name)
    except:
        pass

    p = multiprocessing.Pool(processes=len(sessions))
    objects = []
    for session in sessions:
        df = session[1].copy().reset_index(drop=True)
        
        objects.append((df, experiment_name, cases))
    eye_gaze_session_tables = p.starmap(process_eye_gaze_table, [i for i in objects])
    p.close()

    final_table = concatenate_session_tables(eye_gaze_session_tables)

    # Save experiment consolidated table
    final_table.to_csv(experiment_name + '.csv', index=False)
    #Create video files with raw eye gaze heatmaps
    if video==True:
        create_videos(experiment_name, final_table, cases, data_type='raw_eye_gaze')

if __name__ == '__main__':
    cases = pd.read_csv('../../physionet.org/files/egd-cxr/1.0.0/master_sheet.csv')
    final_table = pd.read_csv('fixation_heatmaps_session_1-5.csv')
    final_table = final_table.loc[final_table['SESSION_ID']==1]
    final_table = final_table.loc[final_table['MEDIA_ID'] <= 4]
    #FOR fixations.csv: To generate heatmap images and create videos of the heatmaps, uncomment the following line
    create_videos('fixation_heatmaps_for_video',final_table, cases, data_type='fixations')

    #process_fixations(experiment_name='fixation_heatmaps_audio', video=False)

    #The following method is required only if you want to work with the raw eye gaze data (as they come from the machine unprocessed). Please read paper for the differences.
    # process_raw_eye_gaze(experiment_name='eye_gaze_heatmaps')
