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

def validation_data_generators(path_to_Train="data/Train", path_to_Validation="data/Validation"):
    """
    Split training data into training and validation sets by moving a portion of images.

    Args:
        path_to_Train (str): Path to the training data directory.
        path_to_Validation (str): Path to the validation data directory.

    Returns:
        None

    This function splits the training data into training and validation sets by moving a portion of the images.
    It creates a validation directory structure similar to the training directory.

    Note:
    - Ensure you have the required libraries (os, shutil, random) imported before using this function.
    - By default, it splits the data using a 80-20 ratio for training and validation.
    - You can customize the split ratio by modifying the 'split_ratio' variable.
    """
    name_folders = os.listdir(path_to_Train)
    
    for name_folder in name_folders:
        path_to_train_data = os.path.join(path_to_Train, name_folder)
        path_to_validation_data = os.path.join(path_to_Validation, name_folder)
    
        # Create the validation directory if it doesn't exist
        os.makedirs(path_to_validation_data, exist_ok=True)
    
        path_images = os.listdir(path_to_train_data)
    
        split_ratio = 0.2  # Customize the split ratio as needed
        num_samples = len(path_images)
        num_validation_samples = int(num_samples * split_ratio)
        
        # Randomly select validation images
        validation_image_paths = random.sample(path_images, num_validation_samples)
    
        for image_path in validation_image_paths:
            train_path = os.path.join(path_to_train_data, image_path)
            validation_path = os.path.join(path_to_validation_data, image_path)
    
            # Check if the file exists before moving it
            if os.path.exists(train_path):
                shutil.move(train_path, validation_path)
            else:
                print(f"File not found: {train_path}")

def generator(train: bool, test: bool, image_preprocessing: bool = None, generator_batch_size = 32, generator_image_size = (128, 128) ):
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
        preprocessing_function= image_preprocessing if image_preprocessing else None,  # Custom preprocessing function
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

    validation_image_generator = ImageDataGenerator(
        rescale=1.0/255.0,  
        preprocessing_function= image_preprocessing if image_preprocessing else None, 
        rotation_range=40,      
        width_shift_range=0.2,  
        height_shift_range=0.2, 
        shear_range=0.2,        
        zoom_range=0.2,         
        horizontal_flip=True,   
        fill_mode='nearest'     
    )
    
    validation_data_generator = validation_image_generator.flow_from_directory(
        'data/Validation',
        target_size=generator_image_size,
        batch_size=generator_batch_size,
        class_mode='categorical'  
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
        return train_data_generator, validation_data_generator,  test_dg
    elif train:
        return train_data_generator, validation_image_generator
    elif test:
        return test_dg
