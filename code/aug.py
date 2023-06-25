import Augmentor

# Specify the path to the input image
input_image = f'C:/Users/septi/OneDrive/Documents/fsa/code/raw_data/train/dcom'

# Create a pipeline object
p = Augmentor.Pipeline(input_image)

# Add image augmentation techniques
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.zoom(probability=0.5, min_factor=1.05, max_factor=1.08)
p.flip_left_right(probability=0.5)

# Generate new images
p.sample(40)
