import Augmentor

# Specify the path to the input image
input_image = "./raw_data/train/"

# Create a pipeline object
p = Augmentor.Pipeline(input_image)

# Add image augmentation techniques
p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.flip_left_right(probability=0.5)

# Generate new images
p.sample(100)
