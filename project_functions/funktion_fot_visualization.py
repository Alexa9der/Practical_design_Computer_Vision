from  lib.import_libraries import *

def image_show(image=None):
    """
    Display an image using Matplotlib.

    Args:
        image (numpy.ndarray, optional): The input image to display. It should be in the BGR color format.
                                        If not provided or None, the function will return without displaying anything.

    Returns:
        None

    Displays the provided image using Matplotlib after converting it from BGR to RGB color format.
    The function also turns off axis labels for the displayed image.

    Note:
    - Ensure you have the required libraries (OpenCV and Matplotlib) imported before using this function.
    - If the provided image is in an incorrect format or there is an issue with displaying, the function will return None.
    """
    try:
        # Convert the BGR image to RGB format for display
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.axis('off')
    except:
        return None