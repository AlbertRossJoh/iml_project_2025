import os
import numpy as np
from skimage import io
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import csv


def load_csv(test_subject, trial, folder, file):
    """
    Loads a CSV file for a specified test subject, folder, and filename. The file path differs based on 
    whether the subject is 'test_subject_0' or another subject.

    Parameters:
        test_subject (str): The identifier for the test subject (e.g., 'test_subject_0').
        folder (str): The folder within the test subject's directory (e.g., 'grid', 'circle').
        filename (str): The name of the CSV file (without the '.csv' extension) to load.

    Returns:
        data (pd.DataFrame): The loaded CSV file as a pandas DataFrame.
    """

    file_path = os.path.abspath(f'../W02/sessions/{test_subject}/{trial}/{folder}/{file}.csv')
    data = pd.read_csv(file_path)
    
    return data

def load_csv_0(test_subject, folder, filename):
    """
    Loads a CSV file for a specified test subject, folder, and filename. The file path differs based on 
    whether the subject is 'test_subject_0' or another subject.

    Parameters:
        test_subject (str): The identifier for the test subject (e.g., 'test_subject_0').
        folder (str): The folder within the test subject's directory (e.g., 'grid', 'circle').
        filename (str): The name of the CSV file (without the '.csv' extension) to load.

    Returns:
        data (pd.DataFrame): The loaded CSV file as a pandas DataFrame.
    """
    file_path = os.path.join(os.path.relpath(f'../W02/sessions/test_subject_0/gaze/' + folder), f'{filename}.csv')#f'data/test_subjects/{test_subject}/gaze/' + folder), f'{filename}.csv')

    data = pd.read_csv(file_path)
    
    return data

def create_pupil_data(json_input):
    """
    Converts pupil data from a JSON-like input into a NumPy array of pupil coordinates.

    Parameters:
        json_input (dict): A dictionary containing pupil data with 'px' and 'py' keys, 
                           where each key corresponds to a list of x and y coordinates, respectively.

    Returns:
        np.ndarray: An N x 2 array where each row represents a pair of pupil coordinates (px, py).
    """
    raw = []
    for i in range(len(json_input['px'])):
        raw.append([json_input['px'][i], json_input['py'][i]])
    return np.array(raw)

def create_position_data(json_input):
    """
    Converts position data from a JSON-like input into a NumPy array with swapped x and y coordinates.

    Parameters:
        json_input (array-like): An array-like structure containing position data, where the first column 
                                 represents the x-coordinate and the second column represents the y-coordinate.

    Returns:
        np.ndarray: An N x 2 array where the x and y coordinates are swapped (y, x).
    """
    raw = np.array(json_input)[:, [1, 0]]
    return np.array(raw)

def create_position_data_collection(json_input):
    """
    Converts position data from a JSON-like input into a NumPy array, maintaining the original order of 
    x and y coordinates.

    Parameters:
        json_input (array-like): An array-like structure containing position data, where the first column 
                                 represents the x-coordinate and the second column represents the y-coordinate.

    Returns:
        np.ndarray: An N x 2 array where each row contains a pair of x and y coordinates (x, y).
    """
    raw = np.array(json_input)[:, [0, 1]]
    return np.array(raw)


def open_img(path, idx):
    """
    Opens a single image from the specified directory and index.

    Parameters:
        path (str): The directory path where the image is located.
        idx (int): The index of the image file (assumed to be in the format '{idx}.jpg').

    Returns:
        np.ndarray: The image as a NumPy array.

    Raises:
        IOError: If the image cannot be read from the specified path.
    """
    img = io.imread(path + f'/{idx}.jpg')
    if img is None:
        raise IOError("Could not read image")
    return img

def center_crop(img, size):
    """
    Crops the center of an image to the specified size.

    Parameters:
        img (np.ndarray): The input image as a NumPy array.
        size (tuple): A tuple specifying the desired width and height of the cropped image (width, height).

    Returns:
        tuple: 
            - Cropped image as a NumPy array of the specified size.
            - A tuple containing the offsets (dx, dy) representing the amount cropped from the left and top, respectively.
    """
    width, height = size
    i_height, i_width = img.shape[:2]

    dy = (i_height-height)//2
    dx = (i_width-width)//2

    return img[dy: i_height-dy, dx: i_width-dx], (dx, dy)


def plot_error_histogram(data):
    """
    Plots a histogram of the absolute errors for both the x and y coordinates.

    Parameters:
        data (np.ndarray): An N x 2 array where the first column contains errors for the x coordinates 
                           and the second column contains errors for the y coordinates.

    Returns:
        None: Displays the histogram with separate bars for x and y errors, labeled and with a legend.
    """
    num_bins = 8
    plt.hist(data[:, 0],    num_bins,
                            edgecolor='white',
                            alpha = 0.7,
                            label="x")
    plt.hist(data[:, 1],    num_bins,
                            edgecolor='white',
                            alpha = 0.7, label="y")
    plt.xlabel("Absolute error")
    plt.ylabel("Number estimates")
    plt.legend(loc='best')
    plt.margins(x=0.01, y=0.1)



def create_image_grid_viz(dataset, image_size=(250, 250), grid_cols=6):
    """
    Creates a grid visualization of pupil detection on test images from a dataset, cropping each image to the specified size.

    Parameters:
        dataset (dict): A dictionary containing:
                        - 'images_test': A list of test images.
                        - 'pupils_test': A list of corresponding pupil coordinates.
        image_size (tuple): The desired size (width, height) of each cropped image (default: (250, 250)).
        grid_cols (int): The number of columns in the image grid (default: 6).

    Returns:
        np.ndarray: A combined image grid as a NumPy array with pupil positions marked in red.
    """
    images = dataset["images_test"]
    pupils = dataset["pupils_test"]
    n = len(images)
    grid_rows = (n // grid_cols) + (n % grid_cols > 0) 

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))

    for i, ax in enumerate(axes.flat):
        if i < n:
            img = images[i].copy()
            pupil = pupils[i]
            
            cropped, (dx, dy) = center_crop(img, image_size)
            ax.imshow(cropped)

            ax.scatter(pupil[0]-dx, pupil[1]-dy, color='red')

            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.close(fig)
    fig.canvas.draw()
    
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)

    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    
    return img



def image_and_scatter(image, data_points):
    """
    Creates a side-by-side visualization displaying the original image, pupil detection, and screen coordinates as scatter plots.

    Parameters:
        image (np.ndarray): The input image for pupil detection visualization.
        data_points (dict): A dictionary containing:
                            - 'screen_coordinates_test': N x 2 array of screen coordinates.
                            - 'pupils_n_test': N x 2 array of pupil coordinates.

    Returns:
        None: Displays the image and scatter plots of pupil and screen coordinates.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [2, 1, 1]})

    screen = data_points['screen_coordinates_test']
    pupils = data_points['pupils_n_test']
    
    axes[2].scatter(screen[:, 0], screen[:, 1], color='blue', marker='o')
    axes[2].set_xlim(0, 2000)
    axes[2].set_ylim(0, 1000)
    axes[2].set_title("Screen coordinates")
    axes[2].set_xlabel('X-axis')
    axes[2].set_ylabel('Y-axis')
    
    axes[0].imshow(image) 
    axes[0].axis('off')  
    axes[0].set_title("Pupil detection")

    axes[1].scatter(pupils[:, 0], pupils[:, 1], color='red', marker='o')
    axes[1].set_title("Mean pupil center corresponding \n to screen coordinates")
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')



def extract_frames(video_file, output_directory, frame_rate=20):
    """
    Extract frames from a video file and save them as images.

    Parameters:
        video_file (str): The path to the video file.
        output_directory (str): The directory where the frames will be saved. Note, the function will create the folder, 
                                if it does not exist.
        frame_rate (int): The number of frames to extract per second of video.
                          Default is 1 (one frame per second).
        Returns:
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cap = cv.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_file}")
        return

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)

    interval = int(fps / frame_rate)
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            output_file = os.path.join(output_directory, f"frame_{frame_count:04d}.jpg")
            cv.imwrite(output_file, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_file} to {output_directory}")


def load_images_from_folder(folder):
    """
    Load all images from a given folder and return them as a list of tuples.
    Each tuple contains the image filename and the image data.

    Parameters:
        folder (str): The path to the folder containing the images.

    Returns:
        list: A list of tuples, where each tuple contains the image filename (str) and the image data (numpy.ndarray).
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv.imread(os.path.join(folder, filename), cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((filename, img))
    return images

def split_img(img):
    """
    Split the input image into left and right halves.

    Parameters:
        img (numpy.ndarray): The image to be split.

    Returns:
        tuple: A tuple containing the left and right halves of the image (numpy.ndarray).
    """
    h, w = img.shape
    left = img[:, :int(w/2)]  # Left half of the image
    right = img[:, int(w/2):]  # Right half of the image
    #return left, right
    return img


def create_image_grid(images, centers, output_directory, grid_name='pupil_centers_grid', max_images=100):
    """
    Create a grid of images with overlay of pupil centers.
    The resulting grid is saved as a PNG file.
    
    Parameters:
        images (list of tuples): A list where each tuple contains the image filename (str) and the image data (numpy.ndarray).
        centers (list of tuples): A list of tuples representing the detected pupil centers (x, y) for each image. If no center is detected, the value should be None.
        output_directory (str): The directory where the grid image will be saved.
        side (str): Indicates which side of the image to use for the grid ('left' or 'right'). Default is 'left'.
        grid_name (str): The base name for the saved grid image. Default is 'pupil_centers_grid'.
        max_images (int): The maximum number of images to include in the grid. Default is 100.
    
    Returns:
        None
    """
    num_images = min(len(images), max_images)
    if num_images == 0:
        print(f"No images to display in the grid for {grid_name}.")
        return

    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.flatten()

    images = images[:max_images]  
    centers = centers[:max_images]  

    for ax, (image_file, img), center in zip(axs, images, centers):

        center_shifted = (center[0], center[1]) if center else None
        print("Debug", center)
        if center_shifted:
            img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.circle(img_color, (int(center_shifted[0]), int(center_shifted[1])), 5, (0, 0, 255), -1)
            ax.imshow(img_color)
            title = image_file
            ax.set_title(title)
        else:
            ax.imshow(img, cmap='gray')
            title = image_file
            ax.set_title(f'{title} (No center)')
        ax.axis('off')
    
    for i in range(len(images), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"{grid_name}.png"))
    plt.show()


def show_image(title, img, cmap='gray'):
    """
    Display an image using Matplotlib.
    
    Parameters:
        title (str): The title of the image to be displayed.
        img (numpy.ndarray): The image data to display.
        cmap (str): The color map to use for displaying the image. Default is 'gray' for grayscale images.
    
    Returns:
        None
    """
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


########## Visualize

def visualize_pupil_centers(csv_file, pattern, x_min=250, x_max=350, y_min=185, y_max=230):
    """
    Create a scatter plot of the detected pupil centers, find the top N most populated grid areas, 
    and calculate the mean center for each.

    Parameters:
        csv_file (str): The path to the CSV file containing the pupil coordinates (px, py).
        output_directory (str): The directory where the scatter plot image will be saved.
        grid_size (int): The size of each grid cell in pixels (default is 7x7).
        top_n (int): The number of top populated areas to consider (default is 10).
    """
    print(os.path.join(os.getcwd(), csv_file))

    df = pd.read_csv(os.path.join(os.getcwd(), csv_file))

    plt.figure(figsize=(8, 8))
    plt.scatter(df['px'], df['py'], c='red', marker='o', label='Pupil Centers')
    
    plt.xlim(x_min, x_max) 
    plt.ylim(y_min, y_max) 

    # Det her har vi selv tilføjet for at lave et tydeligere grid
    plt.xticks(range(x_min, x_max+1, 1))  # every 5 px
    plt.yticks(range(y_min, y_max+1, 1))  # every 5 px
    plt.grid(True, linestyle='--', alpha=0.5)  # optional: dashed light grid 

    plt.xlabel('X Coordinate (px)')
    plt.ylabel('Y Coordinate (px)')
    plt.title('Scatter Plot of Pupil Centers')
    plt.legend()
    plt.show()

def gen_data_subject_0():
    D = []

    for f in range(4):
        pupils_n = create_pupil_data(load_csv_0('test_subject_0', str(f), 'normalized_pupils'))
        pupils = create_pupil_data(load_csv_0('test_subject_0', str(f), 'pupils'))
        positions_n= create_position_data(load_csv_0('test_subject_0', str(f), 'normalized_screen_coordinates'))
        positions= create_position_data(load_csv_0('test_subject_0', str(f), 'positions'))
        images = [open_img(os.path.abspath(f'data/test_subjects/test_subject_0/gaze/{f}'), i) for i in range(len(positions)-1)]
        D.append({
            "pupils_n_train": pupils_n[:9],
            "pupils_n_test": pupils_n[9:],
            "pupils_train": pupils[:8],
            "pupils_test": pupils[8:],
            "screen_coordinates_n_train": positions_n[:9],
            "screen_coordinates_n_test": positions_n[9:],
            "screen_coordinates_train": positions[:9],
            "screen_coordinates_test": positions[9:],
            "images_train": images[:8],
            "images_test": images[8:],
        })
    return D

def gen_data_subject(test_subject, trial):
    """
    Generates and returns a list of datasets for a given test subject, containing pupil and screen coordinate data
    for different patterns ('grid', 'circle', 'line', 'random'). Each dataset includes training and testing data 
    for pupils and screen coordinates.

    Parameters:
        test_subject (str): The subject identifier (e.g., 'test_subject_1') from which data will be loaded.

    Returns:
        D (list): A list of dictionaries, each containing the following keys:
                  - 'pupils_train': N x 2 array of pupil training coordinates.
                  - 'pupils_test': N x 2 array of pupil testing coordinates.
                  - 'screen_coordinates_train': N x 2 array of screen coordinate training data.
                  - 'screen_coordinates_test': N x 2 array of screen coordinate testing data.
    """
    D = []
    p = ['grid', 'circle', 'line', 'random']
    for f in p:
        D.append({
            f"pupils_train": create_pupil_data(load_csv(test_subject, trial, f, 'mean_pupil_coordinates')),
            f"pupils_test": create_pupil_data(load_csv(test_subject, trial, f, 'mean_pupil_coordinates')),
            f"screen_coordinates_train": create_position_data_collection(load_csv(test_subject,trial, f, 'screen_coordinates')),
            f"screen_coordinates_test": create_position_data_collection(load_csv(test_subject, trial, f, 'screen_coordinates')),
        })
    return D



def save_coordinates_to_csv(S, output_directory, type):
    """
    Save 2D coordinates from a dictionary into separate CSV files in folders named after the keys.
    
    Parameters:
        S (dict): Dictionary containing 2D coordinates (arrays) with keys as folder names.
        base_folder (str): Base folder where the directories and files will be saved.
    """
    base_folder = output_directory
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    for key, coordinates in S.items():

        folder_path = os.path.join(base_folder, key)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        csv_file_path = os.path.join(folder_path, f'{type}_coordinates.csv')
        
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if type == 'screen':
                csvwriter.writerow(['sx', 'sy'])
            else:
                csvwriter.writerow(['px', 'py'])
            for point in coordinates:
                csvwriter.writerow(point)
        
        print(f"Saved {key} coordinates to {csv_file_path}")


def plot_least_squares_plane(ax, results_dict,dataset, coor):
    """
    Plots a least squares regression plane for the predicted gaze coordinates and compares it with the ground truth
    and training data. The function creates a 3D scatter plot of the training data, predictions, and ground truth, 
    along with the regression surface for either the x or y coordinates.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object on which the plot is drawn.
        results_dict (dict): A dictionary containing model coefficients, predicted values, and ground truth values.
                             Keys include:
                             - 'model_x': Coefficients for the x-coordinate model.
                             - 'model_y': Coefficients for the y-coordinate model.
                             - 'predicted': N x 2 array of predicted gaze points.
                             - 'ground_truth': N x 2 array of actual gaze points.
        dataset (dict): A dictionary containing training and testing data for pupils and screen coordinates.
                        Keys include:
                        - 'pupils_train': N x 2 array of pupil training coordinates (px, py).
                        - 'pupils_test': N x 2 array of pupil testing coordinates (px, py).
                        - 'screen_coordinates_train': N x 2 array of training screen coordinates.
                        - 'screen_coordinates_test': N x 2 array of testing screen coordinates.
        coor (str): Specifies whether to plot for the x or y coordinate ('x' or 'y').

    Returns:
        None: Displays a 3D plot of the regression surface, training data, predicted gaze points, and ground truth points.
    """
    x_train = dataset['pupils_train'][:, 0]
    y_train = dataset['pupils_train'][:, 1]
    x_test= dataset['pupils_test'][:, 0]
    y_test = dataset['pupils_test'][:, 1]
    s_x_train = dataset['screen_coordinates_train'][:, 0]
    s_y_train = dataset['screen_coordinates_train'][:, 1]
    s_x_test = dataset['screen_coordinates_test'][:, 0]
    s_y_test = dataset['screen_coordinates_test'][:, 1]

    p_test_x = results_dict['predicted'][:, 0]
    p_test_y = results_dict['predicted'][:, 1]
    s_test_x = results_dict['ground_truth'][:, 0]
    s_test_y = results_dict['ground_truth'][:, 1]
    coeffs_x = results_dict['model_x']
    coeffs_y = results_dict['model_y']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    x_grid, y_grid = np.meshgrid(np.linspace(min(x_train), max(x_train), 20), np.linspace(min(y_train), max(y_train), 20))
    if coor == 'y':
        z_grid = coeffs_y[1] * x_grid + coeffs_y[2] * y_grid + coeffs_y[0]
        # z_grid = coeffs_y[0] * x_grid + coeffs_y[1] * y_grid + coeffs_y[2]  
        z_train = x_train * coeffs_y[1]+y_train * coeffs_y[2]+coeffs_y[0]
        # z_train = x_train * coeffs_y[0]+y_train * coeffs_y[1]+coeffs_y[2]
    else:
        z_grid = coeffs_x[1] * x_grid + coeffs_x[2] * y_grid + coeffs_x[0]
        # z_grid = coeffs_x[0] * x_grid + coeffs_x[1] * y_grid + coeffs_x[2]
        z_train = x_train * coeffs_x[1]+y_train * coeffs_x[2]+coeffs_x[0]
        # z_train = x_train * coeffs_x[0]+y_train * coeffs_x[1]+coeffs_x[2]
    
    ax.scatter(x_train, y_train, z_train, color=colors[0], s=50, label='Training data')

    if coor == 'y':
        ax.scatter(x_test, y_test, p_test_y, color=colors[1], label=f'Predicted gaze')
        ax.scatter(x_test, y_test, s_test_y, color=colors[2], label='Ground truth')
    else:
        ax.scatter(x_test, y_test, p_test_x, color=colors[1], label=f'Predicted gaze')
        ax.scatter(x_test, y_test, s_test_x, color=colors[2], label='Ground truth')

    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color=colors[0])

    if coor == 'y':
        for i in range(len(p_test_x)):
            ax.plot([x_test[i], x_test[i]],  
                    [y_test[i], y_test[i]],  
                    [p_test_y[i], s_test_y[i]], 
                    color='black', linestyle='-', linewidth=1)
    else:
        for i in range(len(p_test_x)):
            ax.plot([x_test[i], x_test[i]], 
                    [y_test[i], y_test[i]], 
                    [p_test_x[i], s_test_x[i]],  
                    color='black', linestyle='-', linewidth=1)



def plot_from_results_dict(results_dict, training_data):
    """
    Plots 3D surfaces representing the least squares fit for predicted and ground truth gaze coordinates
    based on training data. The plot shows two surfaces: one for the x-coordinate and one for the y-coordinate.

    Parameters:
        results_dict (dict): A dictionary containing the model's coefficients and prediction results. 
                             Keys include:
                             - 'model_x': Coefficients for the x-coordinate model.
                             - 'model_y': Coefficients for the y-coordinate model.
                             - 'predicted': N x 2 array of predicted gaze points.
                             - 'ground_truth': N x 2 array of actual gaze points.
        training_data (dict): A dictionary containing training data for pupils and screen coordinates.
                             Keys include:
                             - 'pupils_train': N x 2 array of pupil training coordinates (px, py).
                             - 'screen_coordinates_train': N x 2 array of screen coordinates for training.

    Returns:
        None: Displays two 3D plots comparing the model's predictions and the actual ground truth for both
              x and y coordinates.
    """
    x = training_data['pupils_train'][:, 0] 
    y = training_data['pupils_train'][:, 1]
    s_x_train = training_data['screen_coordinates_train'][:, 0]
    s_y_train = training_data['screen_coordinates_train'][:, 1]

    coeffs_x = results_dict['model_x']
    coeffs_y = results_dict['model_y']
    p_test_x = results_dict['predicted'][:, 0]
    p_test_y = results_dict['predicted'][:, 1]
    s_test_x = results_dict['ground_truth'][:, 0]
    s_test_y = results_dict['ground_truth'][:, 1]


    fig = plt.figure(figsize=(16, 8)) 

    ax1 = fig.add_subplot(121, projection='3d')
    plot_least_squares_plane(ax1, results_dict,training_data, 'x')

    ax1.set_xlabel('px_train')
    ax1.set_ylabel('py_train')
    ax1.set_zlabel('x_pred = a*px+b*py+c')
    ax1.set_title(f'Prediction x')
    ax1.view_init(elev=0, azim=90) 
    ax1.set_yticklabels([])
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    plot_least_squares_plane(ax2, results_dict,training_data, 'y')
    ax2.set_xlabel('px_train')
    ax2.set_ylabel('py_train')
    ax2.set_zlabel('y_pred = a*px+b*py+c')
    ax2.set_title(f'Prediction y')
    ax2.set_xticklabels([])
    ax2.view_init(elev=0, azim=0) 
    ax2.legend()

    plt.show()

def compute_error(a, b, c, x, y, z):
    """
    Computes the sum of squared residuals (error) for a linear model with parameters a, b, and c.

    Parameters:
        a (float): Coefficient for the x variable in the model.
        b (float): Coefficient for the y variable in the model.
        c (float): Constant term in the model.
        x (np.ndarray): Array of x-coordinate data points.
        y (np.ndarray): Array of y-coordinate data points.
        z (np.ndarray): Array of z-coordinate data points (target values).

    Returns:
        float: The sum of squared residuals between the predicted values and the actual z values.
    """
    predictions = a * x + b * y + c
    error = np.sum((z - predictions) ** 2)
    return error

def plot_error_surfaces(a_fixed, b_fixed, c_fixed, x, y, z):
    """
    Plots 3D error surfaces and a 2D error curve based on varying model parameters for least squares fitting.

    Parameters:
        a_fixed (float): The fixed coefficient 'a' for one of the error surfaces.
        b_fixed (float): The fixed coefficient 'b' for one of the error surfaces.
        c_fixed (float): The fixed coefficient 'c' for the 2D error curve.
        x (np.ndarray): Array of x-coordinate data points.
        y (np.ndarray): Array of y-coordinate data points.
        z (np.ndarray): Array of z-coordinate data points (target values).

    Returns:
        None: Displays two plots - a 3D surface showing the error varying with 'a' and 'b', and a 2D curve showing
              how the error changes with 'c'.
    """

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    a_values = np.linspace(a_fixed - 10, a_fixed + 10, 50)
    b_values = np.linspace(b_fixed - 10, b_fixed + 10, 50)
    c_values = np.linspace(c_fixed - 100, c_fixed + 500, 100)
    
    a_grid, b_grid = np.meshgrid(a_values, b_values)
    error_grid_ab = np.zeros_like(a_grid)
    
    for i in range(a_grid.shape[0]):
        for j in range(a_grid.shape[1]):
            error_grid_ab[i, j] = compute_error(a_grid[i, j], b_grid[i, j], c_fixed, x, y, z)

    error_c = [compute_error(a_fixed, b_fixed, c_val, x, y, z) for c_val in c_values]

    min_error_idx = np.unravel_index(np.argmin(error_grid_ab), error_grid_ab.shape)
    min_a = a_grid[min_error_idx]
    min_b = b_grid[min_error_idx]
    min_error = error_grid_ab[min_error_idx]

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    
    surf = ax1.plot_surface(a_grid, b_grid, error_grid_ab, cmap='coolwarm', edgecolor='none')
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('Error (Sum of Squared Residuals)')
    ax1.set_title(f'Error Surface: Varying a and b (c = {c_fixed})')

    ax1.view_init(elev=30, azim=135) 
    
    ax1.scatter(min_a, min_b, min_error, color=colors[1], s=100, label='Min Error')
    ax1.text(min_a, min_b, min_error, f'a={min_a:.2f}, b={min_b:.2f}', color='black', fontsize=10)

    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Error Value')

    ax2 = fig.add_subplot(122)
    ax2.plot(c_values, error_c, color=colors[0])
    ax2.set_xlabel('c')
    ax2.set_ylabel('Error (Sum of Squared Residuals)')
    ax2.set_title(f'Error Curve: Varying c (a = {a_fixed}, b = {b_fixed})')
    
    plt.tight_layout()
    plt.show()

def plot_results(predicted, ground_truth):
    """Plot prediction results with numbers and arrows indicating prediction/ground_truth pairs.
    
    Args:
        predicted: Nx2 array of predicted points. 
        ground_truth: Nx2 array of ground_truth points.
    """
    dx = predicted[:, 0] - ground_truth[:, 0]
    dy = predicted[:, 1] - ground_truth[:, 1]
    plt.quiver(ground_truth[:, 0], ground_truth[:, 1], dx, dy, scale=1, angles='xy', scale_units='xy')
    plt.scatter(predicted[:, 0], predicted[:, 1], label="Prediction")
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="Ground truth")
    for i in range(predicted.shape[0]):
            plt.annotate(i, (predicted[i, 0], predicted[i, 1]))
            plt.annotate(i, (ground_truth[i, 0], ground_truth[i, 1]))
    plt.legend()
    plt.title('Ground truth and predictions')

def plot_error_histogram(data):
    """
    Plots a histogram of the absolute errors in the x and y coordinates.

    Parameters:
        data (np.ndarray): An N x 2 array of error values, where the first column contains errors in the x direction
                           and the second column contains errors in the y direction.

    Returns:
        None: Displays the histogram with separate bars for x and y errors, along with labels and a legend.
    """
    num_bins = 8
    plt.hist(data[:, 0], num_bins, edgecolor='white', alpha=0.7, label="x")
    plt.hist(data[:, 1], num_bins, edgecolor='white', alpha=0.7, label="y")
    plt.xlabel("Absolute error")
    plt.ylabel("Number estimates")
    plt.legend(loc='best')
    plt.margins(x=0.01, y=0.1)

def plot_results_grid(results, grid_size):
    """
    Plot results for datasets, displaying scatter plots and histograms in a grid.

    Args:
        results (list of dicts): A list where each dict contains 'predicted', 'ground_truth', and 'errors' for each dataset.
    """
    if grid_size == 1:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, axes = plt.subplots(grid_size, 3, figsize=(12, 16))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(grid_size): 
        predicted = results[i]["predicted"]
        ground_truth = results[i]["ground_truth"]
        errors = results[i]["errors"]

        if grid_size == 1:
            ax_0, ax_1, ax_2 = axes[0], axes[1], axes[2]
        else:
            ax_0, ax_1, ax_2 = axes[i, 0], axes[i, 1], axes[i, 2]
        ax_0.clear()
        ax_0.quiver(ground_truth[:, 0], ground_truth[:, 1], 
                    predicted[:, 0] - ground_truth[:, 0], 
                    predicted[:, 1] - ground_truth[:, 1], 
                    angles='xy', scale_units='xy', scale=1, color="black")
        ax_0.scatter(predicted[:, 0], predicted[:, 1], label="Prediction", color=colors[0])
        ax_0.scatter(ground_truth[:, 0], ground_truth[:, 1], label="Ground truth", color=colors[1])
        ax_0.legend()
        ax_0.set_title('Ground truth and predictions')

        ax_1.clear()
        ax_1.hist(errors[:, 0], bins=8, edgecolor='white', alpha=0.7, label="x", color=colors[0])
        ax_1.hist(errors[:, 1], bins=8, edgecolor='white', alpha=0.7, label="y", color=colors[1])
        ax_1.set_xlabel("Absolute error")
        ax_1.set_ylabel("Number estimates")
        ax_1.legend(loc='best')
        ax_1.margins(x=0.01, y=0.1)

        metrics = [results[i]['rmse'], results[i]['mae'][0], results[i]['mae'][1], results[i]['dist']]
        metric_labels = ['RMSE', 'MAE x', 'MAE y', 'Euclidean']
        ax_2.clear()
        ax_2.bar(metric_labels, metrics, color=[colors[0], colors[1], colors[2], colors[3]], alpha=0.8)
        ax_2.set_ylabel('Metric Value')
        ax_2.set_title(f'Metrics for Dataset {i+1}')
        ax_2.set_ylim(0, 1000)

    plt.tight_layout()
    plt.show()

def bar_comparison_plot(rmses_linear, rmses_quad):
    """Draw a bar chart comparing two sets of RMS errors.
    
    Args:
        rmses_linear: Results from the linear model.
        rmses_quad: Results from the quadratic model.
    """
    width = 0.3
    datasets = ["p0", "p1", "p2", "p3"]
    bx = np.arange(len(datasets))
    plt.bar(bx, rmses_linear, width, label="Linear", alpha = 0.8) 
    plt.bar(bx + width, rmses_quad, width, label="Quadratic", alpha = 0.8) 
    plt.xticks(bx + width/2, datasets)
    plt.title("RMSE for the linear and quadratic model")
    plt.legend()
    plt.show()


