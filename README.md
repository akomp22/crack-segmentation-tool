# Crack segmentation tool

The presented crack segmentation tool is designed to generate pixel-wise labaling of images of cracks in a semi-automatic manner. Its main goal is to simplify creation of a dataset to train deep learning algorithm for crack segmentation. The mathematical underpin for the algorithm was developed by Remco Duits(https://www.win.tue.nl/~rduits/) and group of mathematical image analysis of Eindhoven University of Technology.
The tool consists of few main steps:
1. Manual selection of two crack end-points
2. Finding of a crack path between the selected points
3. Crack edges along the retrieved path are found
4. Pixels between crack edges are marked as crack pixels.

The tool can be used in two ways. 
 - As an app with a graphical user interface is develope (see tuturial bellow)
 - As python functions (see file Examples.ipynb)

## App tuturial
