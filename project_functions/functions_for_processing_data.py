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
    
    return image.astype(np.float32)

def edit_training_data(path_to_Train="data/Train", path_to_edited_data="data/edited_training_data"):
    """
    Preprocess and save edited training data to a new directory.

    Args:
        path_to_Train (str): Path to the original training data directory.
        path_to_edited_data (str): Path to the directory where edited training data will be saved.

    Returns:
        None

    This function preprocesses the original training data by applying custom preprocessing and saves
    the edited data to a new directory structure similar to the original training directory.

    Note:
    - Ensure you have the required libraries (os, cv2, tqdm) imported before using this function.
    - The custom preprocessing function ('image_preprocessing') is used to preprocess images.
    """
    name_folders = os.listdir(path_to_Train)
    os.makedirs(path_to_edited_data, exist_ok=True)
    
    for name_folder in tqdm(name_folders):
        path_to_train_data = os.path.join(path_to_Train, name_folder)
        path_to_new_data = os.path.join(path_to_edited_data, name_folder)
    
        os.makedirs(path_to_new_data, exist_ok=True)
        
        path_images = os.listdir(path_to_train_data)
    
        for image in path_images:
            train_path = os.path.join(path_to_train_data, image)
            new_data_path = os.path.join(path_to_new_data, image)
            
            if not os.path.exists(new_data_path):
                img = cv.imread(train_path)
                
                if img is not None:
                    # Apply custom preprocessing to the image
                    preprocessed_img = image_preprocessing(img)
                    
                    # Save the preprocessed image to the new directory
                    cv.imwrite(new_data_path, preprocessed_img)

def create_validation_data(path_to_Train="data/edited_training_data", path_to_Validation="data/validation"):
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

def generator(train: bool, validation: bool, image_preprocessing: bool = None,
              generator_batch_size=32, generator_image_size=(32, 32),
              train_data_path='data/Train', validation_data_path='data/validation'):
    """
    Generate data generators for training, validation, and testing.

    Args:
        train (bool): Flag indicating whether to create a training data generator.
        test (bool): Flag indicating whether to create a testing data generator.
        image_preprocessing (function): Custom preprocessing function for images.
        generator_batch_size (int): Batch size for data generators.
        generator_image_size (tuple): Target image size (height, width) for data generators.
        train_data_path (str): Path to the training data directory.
        validation_data_path (str): Path to the validation data directory.

    Returns:
        Tuple[ImageDataGenerator, ImageDataGenerator, tf.data.Dataset]: Data generators for training, validation, and testing.

    This function creates data generators for training, validation, and testing image data.
    It uses the Keras ImageDataGenerator for training and validation and a custom generator for testing.

    Note:
    - Ensure you have the required libraries (OpenCV, pandas, TensorFlow) imported before using this function.
    - The function rescales pixel values to the range [0, 1], applies custom preprocessing, and performs data augmentation for training.
    - Testing data is loaded using a custom generator and should be provided as separate image files and labels.
    """
    
    # Create an instance of ImageDataGenerator for training data
    image_generator = ImageDataGenerator(
        rescale=1.0/255.0,
        preprocessing_function=image_preprocessing if image_preprocessing else None,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    # Create a data generator for training data
    train_data_generator = image_generator.flow_from_directory(
        train_data_path,
        target_size=generator_image_size,
        batch_size=generator_batch_size,
        class_mode='categorical',
    )

    # Create an instance of ImageDataGenerator for validation data
    validation_image_generator = ImageDataGenerator(
        rescale=1.0/255.0,
        preprocessing_function=image_preprocessing if image_preprocessing else None,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    # Create a data generator for validation data
    validation_data_generator = validation_image_generator.flow_from_directory(
        validation_data_path,
        target_size=generator_image_size,
        batch_size=generator_batch_size,
        class_mode='categorical',
    )

    if train and validation:
        return train_data_generator, validation_data_generator
    elif train:
        return train_data_generator
    


def test_data_generator(generator_image_size = (32,32), generator_batch_size = 32 ):
    def test_generator():
        test_data_dir = 'data/Test'
        test_data_classes = pd.read_csv("data/Test.csv", usecols=["ClassId", "Path"])
        test_image_files = os.listdir(test_data_dir)
    
        for image_file in test_image_files:
            image_path = os.path.join(test_data_dir, image_file)
            image = cv.imread(image_path)  # Load the image using OpenCV
            image = cv.resize(image, generator_image_size)  # Resize the image
            image = image_preprocessing(image)  # Apply custom preprocessing
    
            class_label = int(test_data_classes.loc[image_file == test_data_classes["Path"].str.split("/").str.get(1), "ClassId"])
            
            one_hot_label = tf.one_hot(class_label, depth=43)
    
            yield image, one_hot_label
            
    # Create a tf.data.Dataset for testing data
    test_dg = tf.data.Dataset.from_generator(
        test_generator,
        output_signature=(
            tf.TensorSpec(shape=(*generator_image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(43,), dtype=tf.int32)
        )
    )
    
    test_dg = test_dg.batch(generator_batch_size)
    return test_dg



# Video preprocess and generations

def preprocess_video_file(filepath, max_frames=5, offset=0):
    cap = cv2.VideoCapture(filepath)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + offset) 
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
            frame_count += 1
        else:
            break
    cap.release()
    return np.array(frames)

def train_video_data_generation():
    classes = ['Haircut', 'GolfSwing', 'JumpingJack', 'Knitting','RockClimbingIndoor']
    filepath = r"data/UCF-101"
    num_samples = 0
    for i, class_ in enumerate(classes):
        path_to_videos =  os.listdir(os.path.join(filepath, class_))
        for video_name in path_to_videos:
            path_to_video =  os.path.join(filepath, class_, video_name)
            result = preprocess_video_file(path_to_video)
            result = result / 255.
            yield result, (i,)


def validation_video_data_generation():
    
    classes = ['Haircut', 'GolfSwing', 'JumpingJack', 'Knitting','RockClimbingIndoor']
    filepath = r"data/UCF-101"
    
    for i, class_ in enumerate(classes):
        path_to_videos =  os.listdir(os.path.join(filepath, class_))
        for video_name in path_to_videos:
            path_to_video =  os.path.join(filepath, class_, video_name)
            result = preprocess_video_file(path_to_video, offset=5 )
            result = result / 255.
            yield result, (i,)

# intermediate functions


def remove_duplicate_images_by_name(path_to_Train="data/Train"):
    """
    Remove duplicate images based on their file names in a given training data directory.

    Args:
        path_to_Train (str): Path to the training data directory.

    Returns:
        None

    This function iterates through the training data directory and removes images that have duplicate names.
    Images with names containing more than three parts are considered duplicates and are removed.

    Note:
    - Ensure you have the required libraries (os, tqdm) imported before using this function.
    - This function is designed to remove images with duplicate names to clean the training data.
    """

    for folder in tqdm(os.listdir(path_to_Train)):
        folder_path = os.path.join(path_to_Train, folder)

        for image in os.listdir(folder_path):
            parts = image.split("_")

            # Check if the image name has more than three parts (indicating a duplicate)
            if len(parts) > 3:
                image_path = os.path.join(folder_path, image)
                
                # Remove the duplicate image
                os.remove(image_path)


