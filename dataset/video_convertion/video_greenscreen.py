# Importing all necessary libraries
import cv2
import os
import numpy as np
from datetime import datetime
from termcolor import colored
import matplotlib.pyplot as plt


# recurse over dirs, and extract frames from every video found
def read_dir_and_start_frame_extraction(path, output, img_size, folders):
    directory = os.fsencode(path)

    text = "Extracting frames from files found under: " + path
    print(colored(text, 'green'))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_file_path = create_path(path, filename)

        if filename.endswith(".h264") or filename.endswith(".mp4"):
            extract_frames_from_video(full_file_path, output, img_size, folders)
        elif os.path.isdir(full_file_path):
            read_dir_and_start_frame_extraction(full_file_path, output, img_size, folders)


def extract_frames_from_video(videoSource, output, imgSize, CLASS_CATEGORIES):
    # Read the video from specified source
    cam = cv2.VideoCapture(videoSource)

    category = videoSource.split('-')[1]
    imageOutputPath = build_output_image_path(category, output)
    currentframe = 0

    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            videoFileName = clean_filename(os.path.basename(videoSource))
            outputImageName = videoFileName + '-frame-' + str(currentframe) + '-bW.jpg'
            outputDestination = create_path(imageOutputPath, outputImageName)

            # writing the extracted images
            cv2.imwrite(outputDestination, frame)

            # manipulate extracted image
            img_ = cv2.imread(outputDestination, cv2.COLOR_BGR2RGB)
            img_copy = np.copy(img_)

            # greenscreen removal happening here
            lower_green = np.array([0, 255, 0])
            upper_green = np.array([120, 255, 100])

            # Masking
            mask = cv2.inRange(img_copy, lower_green, upper_green)
            masked_image = np.copy(img_copy)
            masked_image[mask != 0] = [0, 0, 0]
            plt.imshow(masked_image, cmap='gray')

            # Change background image
            # TODO: fix background image path
            background_image = cv2.imread('/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_4/forest.jpeg')
            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

            # Crop background image to correct size
            crop_background = background_image[0:1000, 0:1000]
            crop_background[mask == 0] = [0, 0, 0]

            final_image = crop_background + masked_image

            # Convert to gray images
            gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
            img_copy = cv2.resize(gray, (imgSize, imgSize))

            # overwrite extracted image with updated properties
            cv2.imwrite(outputDestination, img=img_copy)
            plt.imshow(img_copy, cmap='gray')

            # increasing counter so that it will
            # correctly name the file with current frame counter
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


# helper to clean the filename from its extension
def clean_filename(fileName):
    strList = fileName.split('.')
    strList.pop()

    return "".join(strList)


# helper to create joined path with OS specific delimiters
def create_path(root, path):
    return os.path.join(root, path)


# combine passed in directory name and source category
def build_output_image_path(category, directoryName):
    imageOutputPath = create_path(directoryName, category)

    try:
        # creating a folder named data
        if not os.path.exists(imageOutputPath):
            os.makedirs(imageOutputPath)

        # if not created, then raise error
    except OSError:
        print(colored('Error: Creating directory: ' + imageOutputPath, 'red'))

    return imageOutputPath


def video_to_images(input_path, output_path, folders, img_size):
    startTime = datetime.now()

    read_dir_and_start_frame_extraction(input_path, output_path, img_size, folders)

    timeToRun = datetime.now() - startTime
    print("Done in " + str(timeToRun.seconds) + " seconds.")
