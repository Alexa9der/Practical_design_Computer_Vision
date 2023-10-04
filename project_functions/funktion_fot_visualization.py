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
        plt.show()
    except:
        return None

def show_histogram(image):
    """
    Display histograms for the blue, green, and red color channels of an image.

    Args:
        image (numpy.ndarray): The input image for which histograms will be displayed.

    Returns:
        None

    This function calculates and displays histograms for the blue (B), green (G), and red (R) color channels
    of the input image using Matplotlib.

    Note:
    - Ensure you have the required libraries (OpenCV and Matplotlib) imported before using this function.
    - The function will display three subplots: one for each color channel (B, G, R).
    - The x-axis represents pixel intensity values (0-255), and the y-axis represents the frequency of each intensity.
    """
    ax1 = plt.subplot(311)
    plt.xlim([0, 256])
    plt.ylim([0, 1100])
    for i, col in enumerate('bgr'):  # ['b', 'g', 'r']
        # i = 0, col='b'
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        ax1 = plt.subplot(3, 1, i + 1, sharey=ax1)
        plt.plot(hist, color=col)
        
        if i < 2:
            plt.setp(ax1.get_xticklabels(), visible=False)
    
    plt.show()