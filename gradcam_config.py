NUMBER_OF_ACTIONS = 4                       # Size of the action space
NUMBER_OF_FRAMES = 450                      # Number of frames to play before to make a gradCam saliency map
LAYER_NAME = "conv2d_8"                       # Name of the layer to calculate the gradient w.r.t
INPUT_DIM = 84                              # Dimension of the input image
MODEL_DIR = 'breakout-saves/save-00050161'  # Directory of the saved model
SAVED_MODLE = 'dqn.h5'                      # Saved model file name
VIDEO_DIR = 'movies/'                       # directory where the movie is going to be saved in