"""
This file just sets the path to the directories were data is stored
Data includes:
- sky images (regular and preprocessed)
- irradiance and clear sky irradiance corresponding to images
Need to add:
- my computer = DESKTOP-A1O805T
- maybe ask quentin for some of the images to run the program
"""
# C:\Users\julia\OneDrive\Documents\Cambridge Work\Dissertation\Data\Satellite\Palaiseau_ghi_2019_07.h5


def data_images_dir(computer):

    if computer == 'DESKTOP-A1O805T':
        DATADIR = "D:/Users/julia/Documents/Cambridge Work/Dissertation/Data/Sirta/Sirta/Sky_images/sfr02_201807_201807/"  # Julian's
    elif computer=='quentin-UX330UAK-Ubuntu':
        DATADIR = "/home/quentin/Documents/Cambridge/Research_project/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Laptop
    elif computer=='Nutmeg' or computer=='Gpu_hpc':
        DATADIR = "/home/qp208/Documents/Research_project/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Nutmeg
    elif computer == 'bagnet':
        DATADIR = "/scratch3/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Bagnet
        DATADIR = "/scratch3/qp208/Datasets_2014_2019/Sirta/Sky_images/"  # Bagnet
    elif computer == 'windle' or computer == 'localhost3':
        DATADIR = "/scratch/Datasets_qp208/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Windle
        DATADIR = "/scratches/bagnet_3/qp208/Datasets_2014_2019/Sirta/Sky_images/" #Images_Y_128px/"  # Windle
    elif computer == 'mario':
        DATADIR = "/scratches/bagnet_3/qp208/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Mario
        DATADIR = "/scratches/bagnet_3/qp208/Datasets_2014_2019/Sirta/Sky_images/"


    return DATADIR


def data_preprocessed_images_dir(computer):

    if computer == 'DESKTOP-A1O805T':
        DATADIR = "D:/Users/julia/Documents/Cambridge Work/Dissertation/Data/Sirta/Sirta/Sky_images/sfr02_201807_201807/"  # Julian's
    elif computer=='quentin-UX330UAK-Ubuntu':
        DATADIR = "/home/quentin/Documents/Cambridge/Research_project/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Laptop
    elif computer=='Nutmeg' or computer=='Gpu_hpc':
        DATADIR = "/home/qp208/Documents/Research_project/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Nutmeg
    elif computer == 'bagnet':
        DATADIR = "/scratch/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Bagnet
    elif computer == 'windle' or computer == 'localhost3':
        DATADIR = "/scratch/Datasets_qp208/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Windle
        DATADIR = "/scratch/qp208/Datasets_2014_2019/Sirta/Sky_images/Images_Y_128px/"  # Windle
        #DATADIR = "/scratches/bagnet_3/qp208/Datasets_2014_2019/Sirta/Sky_images/Images_Y_128px_init/"  # Windle
    elif computer == 'mario':
        DATADIR = "/scratches/bagnet_3/qp208/Datasets/Sirta/Sky_images/sfr02_201801_201809/"  # Mario
        #DATADIR = "/scratches/bagnet_3/qp208/Datasets_2014_2019/Sirta/Sky_images/"
        DATADIR = "/scratch/qp208/Datasets_2014_2019/Sirta/Sky_images/Images_Y_128px/"  # Windle

    return DATADIR


def data_irradiance_dir(computer):

    if computer == 'DESKTOP-A1O805T':
        DATADIR_IRRADIANCE = "D:/Users/julia/Documents/Cambridge Work/Dissertation/Data/Sirta/Sirta/Irradiance/solys2/preprocessed_irradiance_data/"  # Julian's
    elif computer == 'DESKTOP-2OUOK5M':
        DATADIR_IRRADIANCE = "C:/Users/nappe/Documents/Julian/Dissertation/solys2/"  # Dad's
    elif computer == 'quentin-UX330UAK-Ubuntu':
        DATADIR_IRRADIANCE = "/home/quentin/Documents/Cambridge/Research_project/Datasets/Sirta/Irradiance/solys2/preprocessed_irradiance_data/"
    elif computer == 'Nutmeg' or computer == 'Gpu_hpc':
        DATADIR_IRRADIANCE = "/home/qp208/Documents/Research_project/Datasets/Sirta/Irradiance/solys2/preprocessed_irradiance_data/"
    elif computer == 'bagnet':
        DATADIR_IRRADIANCE = "/scratch/Datasets/Sirta/Irradiance/solys2/preprocessed_irradiance_data/"
    elif computer == 'windle' or computer == 'localhost3':
        DATADIR_IRRADIANCE = "/scratch/Datasets_qp208/Datasets/Sirta/Irradiance/solys2/preprocessed_irradiance_data/"
        DATADIR_IRRADIANCE = "/scratches/bagnet_3/qp208/Datasets_2014_2019/Sirta/Irradiance/solys2/2014_2019_preprocessed_irradiance_data/"
        #DATADIR_IRRADIANCE = "/scratch/qp208/Datasets_2014_2019/Sirta/Irradiance/solys2/2014_2019_preprocessed_irradiance_data/"
    elif computer == 'mario':
        DATADIR_IRRADIANCE = "/scratches/bagnet_3/qp208/Datasets/Sirta/Irradiance/solys2/preprocessed_irradiance_data/"
        DATADIR_IRRADIANCE = "/scratches/bagnet_3/qp208/Datasets_2014_2019/Sirta/Irradiance/solys2/2014_2019_preprocessed_irradiance_data/"
        #DATADIR_IRRADIANCE = "/scratch/qp208/Datasets_2014_2019/Sirta/Irradiance/solys2/2014_2019_preprocessed_irradiance_data/"

    return DATADIR_IRRADIANCE


def data_clear_sky_irradiance_dir(computer):

    if computer == 'DESKTOP-A1O805T':
        DATADIR_CLEAR_SKY_IRRADIANCE = "D:/Users/julia/Documents/Cambridge Work/Dissertation/Data/Sirta/Helio/"  # Julian's
    elif computer == 'DESKTOP-2OUOK5M':
        DATADIR_CLEAR_SKY_IRRADIANCE = "C:/Users/nappe/Documents/Julian/Dissertation/Helio/"  # Dad's
    elif computer == 'quentin-UX330UAK-Ubuntu':
        DATADIR_CLEAR_SKY_IRRADIANCE = "/home/quentin/Documents/Cambridge/Research_project/Datasets/Helio/"
    elif computer == 'Nutmeg' or computer == 'Gpu_hpc':
        DATADIR_CLEAR_SKY_IRRADIANCE = "/home/qp208/Documents/Research_project/Datasets/Helio/"
    elif computer == 'bagnet':
        DATADIR_CLEAR_SKY_IRRADIANCE = "/scratch/Datasets/Helio/"
    elif computer == 'windle' or computer == 'localhost3':
        DATADIR_CLEAR_SKY_IRRADIANCE = "/scratch/Datasets_qp208/Datasets/Helio/"
        DATADIR_CLEAR_SKY_IRRADIANCE = "/scratch/qp208/Datasets_2014_2019/Helio/"
    elif computer == 'mario':
        DATADIR_CLEAR_SKY_IRRADIANCE = "/scratch/qp208/Datasets_2014_2019/Helio/"

    return DATADIR_CLEAR_SKY_IRRADIANCE


def data_sirta_grid(computer):
    if computer == 'DESKTOP-A1O805T':
        DATADIR_SIRTA_GRID = 'D:/Users/julia/Documents/Cambridge Work/Dissertation/Data/Sirta/Satellite/preprocessed_irradiance_data/ghi/'
    elif computer == 'DESKTOP-2OUOK5M':
        DATADIR_SIRTA_GRID = "C:/Users/nappe/Documents/Julian/Dissertation/satellite_ghi/"  # Dad's

    return DATADIR_SIRTA_GRID

def eumetsat_sat_images(computer):
    if computer == 'DESKTOP-A1O805T':
        DATADIR_EUMETSAT_IMAGES = 'D:/Users/julia/Documents/Cambridge Work/Dissertation/Data/Sirta/Satellite/eumetsat_images/'
    elif computer == 'DESKTOP-2OUOK5M':
        DATADIR_EUMETSAT_IMAGES = "C:/Users/nappe/Documents/Julian/Dissertation/satellite_ghi/"  # Dad's

    return DATADIR_EUMETSAT_IMAGES


#def data_experiments_dir(computer):

#    if computer == 'quentin-UX330UAK-Ubuntu':
#        DATADIR_EXPERIMENTS = "/home/quentin/Documents/Cambridge/Research_project/Experiments"
#    elif computer == 'windle' or computer == 'localhost3' or computer == 'mario':
#        DATADIR_EXPERIMENTS = "/scratches/bagnet/qp208/Experiments"

#    return DATADIR_EXPERIMENTS


def destination_preprocessed_images_dir(computer):

    if computer=='quentin-UX330UAK-Ubuntu':
        DATASET_DESTINATION = "/home/quentin/Documents/Cambridge/Research_project/Datasets/Sirta/Sky_images/preprocessed_dataset/"  # Laptop
    elif computer == 'bagnet':
        DATASET_DESTINATION = "/scratch3/qp208/Datasets_2014_2019/Sirta/Sky_images/Images_Y_128px/"  # Bagnet

    return DATASET_DESTINATION