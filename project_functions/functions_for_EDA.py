from  lib.import_libraries import *

def load_csv_file(file_list):
    """
    Load CSV files into pandas DataFrames and assign them as global variables.

    Args:
        file_list (list): A list of file names to load as CSV files.

    Returns:
        None

    This function iterates through the provided list of file names and loads each CSV file
    into a pandas DataFrame. The DataFrames are assigned as global variables with names
    following the pattern 'csv_<file_name_without_extension>'.

    Note:
    - Ensure you have the required library (pandas) imported before using this function.
    - The function will only load files with a '.csv' extension from the specified list.
    - Global variables with names based on the loaded file names will be created, allowing
      access to the DataFrames globally.
    """
    for file in file_list:
        if ".csv" in file:
            # Extract the base file name without extension
            name_file = file.split(".")
            
            # Read the CSV file and assign it as a global variable
            globals()[f"csv_{name_file[0]}"] = pd.read_csv(f"data/{file}")

def loading_random_data(path="data/Train", amount=5, image_size=(32, 32)):
    """
    Load random images from a specified directory path.

    Args:
        path (str, optional): The directory path from which random images will be loaded.
                             Default is "data/Train".
        amount (int, optional): The number of random images to load. Default is 5.
        image_size (tuple, optional): The desired size (height, width) for the loaded images.
                                     Default is (128, 128).

    Returns:
        numpy.ndarray: An array of loaded images with the specified image size.
                      If the directory specified by 'path' is not found, an empty array is returned.

    This function loads a specified number of random images from the specified directory.
    Each loaded image is resized to the specified image_size.

    Note:
    - Ensure you have the required libraries (OpenCV and NumPy) imported before using this function.
    - If the specified directory path does not exist, a message will be printed, and an empty array will be returned.
    """
    def load(path):
        """
        Helper function to randomly select a file from a directory.
    
        Args:
            path (str): The directory path from which to select a random file.
    
        Returns:
            str: The name of a randomly selected file from the directory.
        """
        data_from_train_folder = os.listdir(path)
        result = np.random.choice(data_from_train_folder)
        return result
    
    try:
        data = []
        for i in range(amount):
            folder_path = load(path)
            image_path = load(os.path.join(path, folder_path))
            
            full_image_path = os.path.join(path, folder_path, image_path)
            image = cv.imread(full_image_path)
            image = cv.resize(image, image_size)
            # image = image.astype('float32') / 255.0 
            
            if image is not None:
                data.append(image)
        
        return np.array(data)
    except FileNotFoundError as e:
        print(f"Directory called '{path}' not found:", e)
        return np.array([])

def histogram_equalization(image):
    """
    Apply histogram equalization to each color channel of an image and merge them.

    Args:
        image (numpy.ndarray): The input image for which histogram equalization will be applied.

    Returns:
        numpy.ndarray: The equalized image with improved contrast.

    This function applies histogram equalization to each color channel (blue, green, and red) of the input image
    using OpenCV's `equalizeHist` function. It then merges the equalized channels to create the final equalized image.

    Note:
    - Ensure you have the required library (OpenCV) imported before using this function.
    - Histogram equalization enhances the image's contrast by spreading pixel intensity values over a wider range.
    """
    # Split the image into its color channels (BGR)
    b, g, r = cv.split(image)
    
    # Apply histogram equalization to each channel
    b_eq = cv.equalizeHist(b)
    g_eq = cv.equalizeHist(g)
    r_eq = cv.equalizeHist(r)
    
    # Merge the equalized channels into a new image
    equalized_image = cv.merge((b_eq, g_eq, r_eq))

    return equalized_image

def contour_alignment(image, clipLimit=5.0, tileGridSize=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.

    Args:
        image (numpy.ndarray): The input image for which CLAHE will be applied.
        clipLimit (float, optional): Threshold for contrast limiting in CLAHE. Default is 5.0.
        tileGridSize (tuple, optional): Size of the grid for histogram equalization. Default is (8, 8).

    Returns:
        numpy.ndarray: The image with enhanced contrast using CLAHE.

    This function applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.
    CLAHE enhances local contrast by applying histogram equalization to small regions (tiles) of the image.

    Note:
    - Ensure you have the required library (OpenCV) imported before using this function.
    - CLAHE improves the image's contrast while avoiding over-amplification of noise in flat regions.
    - You can adjust the `clipLimit` and `tileGridSize` parameters for different results.
    """
    # Create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    # Convert the image to LAB color space
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    # Convert the channels to the required color depth
    l = cv.convertScaleAbs(l)
    a = cv.convertScaleAbs(a)
    b = cv.convertScaleAbs(b)

    # Apply CLAHE to the L (lightness) channel
    l2 = clahe.apply(l)

    # Merge the channels back into the LAB color space
    lab = cv.merge((l2, a, b))

    # Convert the LAB image back to BGR color space
    image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    return image

def contour_in_binarized_image(image):
    """
    Find and draw contours on a binarized version of a color image.

    Args:
        image (numpy.ndarray): The input color image on which contours will be found.

    Returns:
        numpy.ndarray: The color image with drawn contours.

    This function converts the input color image to grayscale and applies binarization to it.
    It then finds contours on the binarized image and draws those contours on the original color image.

    Note:
    - Ensure you have the required library (OpenCV) imported before using this function.
    - Contours represent the boundaries of objects or regions in the image.
    """
    # Create a copy of the input image to avoid modifying the original
    image = image.copy()
    
    # Convert the color image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply binarization to the grayscale image
    _, binary_thresh = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY)
    
    # Find contours on the binarized image
    contours, _ = cv.findContours(binary_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the original color image (black color with a line thickness of 1)
    image_with_contours = cv.drawContours(image, contours, -1, (0, 0, 0), 1)
    
    return image_with_contours

def otsu_threshold_contours(image):
    """
    Apply Otsu's thresholding and find contours on a color image.

    Args:
        image (numpy.ndarray): The input color image on which Otsu's thresholding and contour finding will be applied.

    Returns:
        numpy.ndarray: The color image with drawn contours.

    This function converts the input color image to grayscale, applies Otsu's thresholding to create a binary image,
    and then finds and draws contours on the binary image.

    Note:
    - Ensure you have the required library (OpenCV) imported before using this function.
    - Otsu's thresholding is an automatic thresholding technique to separate objects from the background.
    """

    image =  image.copy()
    gray_shapes = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to the grayscale image
    _, thresh = cv.threshold(gray_shapes, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Find contours on the thresholded image
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original color image (black color with a line thickness of 1)
    img_with_contours = cv.drawContours(image, contours, -1, (0, 0, 0), 1)
    
    return img_with_contours

def adaptive_threshold_contours(image):
    """
    Apply adaptive thresholding and find contours on a color image.

    Args:
        image (numpy.ndarray): The input color image on which adaptive thresholding and contour finding will be applied.

    Returns:
        numpy.ndarray: The color image with drawn contours.

    This function converts the input color image to grayscale, applies adaptive thresholding to create a binary image,
    and then finds and draws contours on the binary image.

    Note:
    - Ensure you have the required library (OpenCV) imported before using this function.
    - Adaptive thresholding is used to handle varying lighting conditions in an image.
    """

    image =  image.copy()
    gray_shapes = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to the grayscale image
    thresh = cv.adaptiveThreshold(gray_shapes, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 1)

    # Find contours on the thresholded image
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original color image (black color with a line thickness of 1)
    img_with_contours = cv.drawContours(image, contours, -1, (0, 0, 0), 1)

    return img_with_contours

