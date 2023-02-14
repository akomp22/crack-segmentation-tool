# Crack segmentation tool

The presented crack segmentation tool is designed to generate pixel-wise labaling of images of cracks in a semi-automatic manner. Its main goal is to simplify creation of a dataset to train deep learning algorithm for crack segmentation that often done fully manually. The mathematical underpin for the algorithm was developed by Remco Duits(https://www.win.tue.nl/~rduits/) and group of mathematical image analysis of Eindhoven University of Technology.
The segmentation process consists of few main steps:
1. Manual selection of two crack end-points
2. Finding of a crack path between the selected points
3. Crack edges along the retrieved path are found
4. Also crack segment can be added by manually drawing crack contur
5. Pixels between crack edges are marked as crack pixels.

The tool can be used in two ways. 
 - As an app with a graphical user interface (see tuturial bellow)
 - As python functions (see file Examples.ipynb)

The app designed to segment images from a specific folder. After segmentation JSON files for each image will be created. "check annotetions.ipynb" shows how to read annotation JSON file.

## App tuturial

#### Start the tool
1. Clone the repository or download archive and unzip it on your local machine. Run the app.py python script
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/Recording%202023-02-14%20at%2012.20.46.gif)

#### Start segmentation
1. Enter path to a folder that contain images you want to segment. 
2. Press "Start"
3. If at anny time during segmentation you decide to skip the image press "Skip" button. The empty annotation file will be saved.
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/Recording%202023-02-14%20at%2013.11.57.gif)

#### Select crack endpoints
1. Press "Select Crack End-Points" to open window for point selection. 
2. Adjust Adjust "Image size" to fit the selection window fit in your screen. 
3. Use the mouse wheel to zoom in/out around current cursor position (to drag image use sequentially zoom in/out and move cursor).
4. Press left mouse button to select point.
5. Press right mouse button to undo previouse point selection
6. Choose two points on image (crack tips)
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/Recording%202023-02-14%20at%2013.36.56.gif)

#### Crop image around crack



