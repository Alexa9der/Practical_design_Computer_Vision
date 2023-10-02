



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





def loading_random_data(path="data/Train", amount=5, image_size=(128, 128)):
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
            
            if image is not None:
                data.append(image)
        
        return np.array(data)
    except FileNotFoundError as e:
        print(f"Directory called '{path}' not found:", e)
        return np.array([])



