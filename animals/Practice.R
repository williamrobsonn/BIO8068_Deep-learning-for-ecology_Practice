#BIO8068 Data Visualisaton in Ecology ----
#Machine learning to classify images

#Starting the packages ----

library(keras)

#Setup ----

#Create project in R folder first remeber!

# list of animals to model
animal_list <- c("Butterfly", "Cow", "Elephant", "Spider")

# number of output classes (i.e. fruits)
output_n <- length(animal_list)

# image size to scale down to (original images vary but about 600 x 800 px)
img_width <- 250
img_height <- 250
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- "Training\\"
valid_image_files_path <- "Validation\\"

#Checking thar the channels are set up correctly 
length(list.files(train_image_files_path, recursive = TRUE))

valid_image_files_path #Same again for the validation pathway 

#Data generators and augmentation ----

#Never augment validation data
#Because this is a small dataset we can transpose, zoom or flip to improve accuracy

#We are going to rescale the images between 0 and 1, images usually have a value of 255

# Rescale from 255 to between zero and 1
train_data_gen = image_data_generator(
  rescale = 1/255
)

valid_data_gen <- image_data_generator(
  rescale = 1/255
)  

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = animal_list,
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = animal_list,
                                                    seed = 42)

# Check that things seem to have been read in OK
cat("Number of images per class:")

table(factor(train_image_array_gen$classes))

cat("Class labels vs index mapping")

train_image_array_gen$class_indices

#Final setup ----

#Just adding a few more things before we finalise the setup 

# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32 # Typical default, though possibly a little high given small dataset
epochs <- 10

#Design your convolutional neural network ----

#Now we define how our CNN is set 

# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 

#It is useful to check the CNN structure before compiling and running it 

print(model)

#Complile and train the model 

# Compile the model
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

#With separate classes, use categorical_crossentropy
#but for example, you had presence/absence data use binary_crossentropy for error types

# Train the model with fit_generator
history <- model %>% fit(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)

