from  lib.import_libraries import *

def image_preprocessing(image):
    """
    Apply a series of image preprocessing techniques to enhance image quality.

    Args:
        image (numpy.ndarray): The input image to be preprocessed.

    Returns:
        numpy.ndarray: The preprocessed image with enhanced quality.

    This function applies a series of image processing techniques to improve image quality and enhance details.
    The techniques used include detail enhancement, median blur, and contrast adjustment.

    Note:
    - Ensure you have the required library (OpenCV) imported before using this function.
    - You can customize the parameters for each processing step as needed.
    """
    # Apply detail enhancement to enhance image details
    image = cv.detailEnhance(image, sigma_s=50, sigma_r=0.3)
    
    # Apply median blur to reduce noise and smooth the image
    image = cv.medianBlur(image, 3)
    
    # Apply contrast adjustment to improve image contrast
    image = cv.addWeighted(image, 1.5, image, -0.5, 0)
    
    # You can optionally add more preprocessing steps such as thresholding and contour finding
    
    return image.astype(np.float64)

def generator(train: bool, test: bool, generator_batch_size = 32, generator_image_size = (128, 128) ):
    """
    Generate data generators for training and testing.

    Args:
        train (bool): Flag indicating whether to create a training data generator.
        test (bool): Flag indicating whether to create a testing data generator.

    Returns:
        Union[ImageDataGenerator, tf.data.Dataset]: Data generators for training and testing.

    This function creates data generators for training and testing image data.
    It uses the Keras ImageDataGenerator for training and a custom generator for testing.

    Note:
    - Ensure you have the required libraries (OpenCV, pandas, TensorFlow) imported before using this function.
    - The function rescales pixel values to the range [0, 1], applies custom preprocessing, and performs data augmentation for training.
    - Testing data is loaded using a custom generator and should be provided as separate image files and labels.
    """
    
    
    # Create an instance of ImageDataGenerator for training data
    image_generator = ImageDataGenerator(
        rescale=1.0/255.0,  # Scale pixel values to the range [0, 1]
        preprocessing_function=image_preprocessing,  # Custom preprocessing function
        rotation_range=40,      # Random rotation up to 40 degrees
        width_shift_range=0.2,  # Random horizontal shift of 20% of the width
        height_shift_range=0.2, # Random vertical shift of 20% of the height
        shear_range=0.2,        # Random shear transformation
        zoom_range=0.2,         # Random zoom
        horizontal_flip=True,   # Random horizontal flip
        fill_mode='nearest'     # Fill mode after transformations
    )
    
    # Create a data generator for training data
    train_data_generator = image_generator.flow_from_directory(
        'data/Train',
        target_size=generator_image_size,
        batch_size=generator_batch_size,
        class_mode='categorical'  # For multi-class classification
    )

    def test_data_generator():
        test_data_dir = 'data/Test'
        test_data_classes = pd.read_csv("data/Test.csv", usecols=["ClassId", "Path"])
        test_image_files = os.listdir(test_data_dir)

        for image_file in test_image_files:
            image_path = os.path.join(test_data_dir, image_file)
            image = cv.imread(image_path)  # Load the image using OpenCV
            image = cv.resize(image, generator_image_size)  # Resize the image
            image = image_preprocessing(image)  # Apply custom preprocessing

            class_label = int(test_data_classes.loc[image_file == test_data_classes["Path"].str.split("/").str.get(1), "ClassId"])

            yield image, class_label
            
    # Create a tf.data.Dataset for testing data
    test_dg = tf.data.Dataset.from_generator(
        test_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(*generator_image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    test_dg = test_dg.batch(generator_batch_size)

    if train and test:
        return train_data_generator, test_dg
    elif train:
        return train_data_generator
    elif test:
        return test_dg
