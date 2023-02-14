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

## App tuturial

### Start the tool
1. Clone the repository or download archive and unzip it on your local machine. Run the app.py python script

![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/Recording%202023-02-14%20at%2012.20.46.gif)

2. Enter path to a folder that contain images you want to segment. Press "Start"
![]([https://github.com/akomp22/crack-segmentation-tool/blob/main/video/Recording%202023-02-14%20at%2012.20.46.gif](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/Recording%202023-02-14%20at%2013.11.57.gif))

3. Press "Select Crack End-Points" to open window for point selection. Adjust Adjust "Image size" to fit the selection window fit in your screen. Use the mouse wheel to zoom in/out around current cursor position (to drag image use sequentially zoom in/out and move cursor). To c
![](https://github.com/akomp22/crack-segmentation-tool/blob/main/video/Recording%202023-02-14%20at%2013.36.56.gif)
