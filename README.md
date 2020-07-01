# Real-time Segmentation of Road Scenes using the ICNet Architecture

1. Download or clone this repo and extract it to your hard disk
2. Navigate to the repo's directory on your hard disk using Anaconda Prompt (using cd)
3. Run **python image.py** to inference on provided sample image
4. Run **python video.py** to inference on provided sample image
5. Run **python screen.py** to inference on screen capture

# Possible errors:

1. 'Object arrays cannot be loaded when allow_pickle=False' 

To resolve this error, try downgrading your numpy version to 1.16.1. It seems to solve the problem.

**pip install numpy==1.16.1**

This version of numpy has the default value of allow_pickle as True.
