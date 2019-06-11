# Libraries
library(keras)

# Install the Keras library
# In case that it is installed, the function rechecks that everything is still there
install_keras()

# Import dataset
mnist 		<- dataset_mnist()

# Make 28*28 grayscale images one-dimensional
x_train 	<- array_reshape(mnist$train$x,	c(60000, 28*28))
x_test 		<- array_reshape(mnist$test$x,	c(10000, 28*28))

# Rescale 256 bit to value between 1 and 0
x_train		<- x_train/256
x_test 		<- x_test/256

# Make labels that are easier to work with
y_train		<- mnist$train$y
y_test   	<- mnist$test$y

# Convert int to bin category, i.e.
# 3 to 0010000000
# 9 to 0000000001
# 0 to 1000000000
y_train 	<- to_categorical(y_train, num_classes=10)
y_test 		<- to_categorical(y_test, num_classes=10)

model <- keras_model_sequential()
model %>%
	layer_dense(units = 256, input_shape = c(784)) %>%
	layer_dense(units = 10, activation = 'softmax')
	
# To check the resulting model is what you expected, use:
#summary(model)

# When it looks ok compile it with suitable loss function, optimization procedures and performance measures
model %>% compile( 
	loss = 'categorical_crossentropy', 
	optimizer = optimizer_rmsprop(),	
	metrics = c('accuracy') 
)

# TRAINING AND EVALUATION -------------------------------------------------------------------------------------------------------------------------

history<- model %>% fit(
	x_train, y_train,
	batch_size = 128,
	epochs = 12,
	verbose = 1,
	validation_split = 0.2
)

# Evaluate the model's performance
score <- model %>% evaluate(
	x_test, y_test,
	verbose = 0
)
#--------------------------------------------------------------
# Rectified model
history<- model %>% fit(
	x_train, y_train,
	batch_size = 128,
	activation = "relu",
	epochs = 12,
	verbose = 1,
	validation_split = 0.2,
)

# Evaluate the second model's performance
score <- model %>% evaluate(
	x_test, y_test,
	verbose = 0
)

# Make 28*28 grayscale images one-dimensional
x_train 	<- array_reshape(mnist$train$x,	c(60000, 28, 28, 1))
x_test 		<- array_reshape(mnist$test$x,	c(10000, 28, 28, 1))

# Rescale 256 bit to value between 1 and 0
x_train		<- x_train/256
x_test 		<- x_test/256

# Make labels that are easier to work with
y_train		<- mnist$train$y
y_test   	<- mnist$test$y

# Convert int to bin category, i.e.
# 3 to 0010000000
# 9 to 0000000001
# 0 to 1000000000
y_train 	<- to_categorical(y_train, num_classes=10)
y_test 		<- to_categorical(y_test, num_classes=10)

model <-keras_model_sequential() %>%
	layer_conv_2d(filters = 32, kernel_size = c(3,3),
		activation = 'relu', input_shape = c(28, 28, 1)) %>%
	layer_conv_2d(filters = 64, kernel_size =  c(3,3),
		activation = 'relu') %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_flatten() %>%
	layer_dense(units = 128, activation = 'relu') %>%
	layer_dense(units = 10,   activation = 'softmax')
	
# To check the resulting model is what you expected, use:
#summary(model)

# When it looks ok compile it with suitable loss function, optimization procedures and performance measures
model %>% compile( 
	loss = 'categorical_crossentropy', 
	optimizer = optimizer_adadelta(),	
	metrics = 'accuracy'
)

history<- model %>% fit(
	x_train, y_train,
	batch_size = 128,
	epochs = 6,
	verbose = 1,
	validation_split = 0.2,
)

# Evaluate the model's performance
score <- model %>% evaluate(
	x_test, y_test,
	verbose = 1
)

model <-keras_model_sequential() %>%
	layer_conv_2d(filters = 32, kernel_size = c(3,3),
		activation = 'relu', input_shape = c(28, 28, 1)) %>%
	layer_conv_2d(filters = 64, kernel_size =  c(3,3),
		activation = 'relu') %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(rate  = 0.25) %>%
	layer_flatten() %>%
	layer_dense(units = 128, activation = 'relu') %>%
	layer_dropout(rate = 0.5) %>%
	layer_dense(units = 10,   activation = 'softmax')
	
# To check the resulting model is what you expected, use:
#summary(model)

# When it looks ok compile it with suitable loss function, optimization procedures and performance measures
model %>% compile( 
	loss = 'categorical_crossentropy', 
	optimizer = optimizer_adadelta(),	
	metrics = 'accuracy'
)

#activate model but this time only with 6 epochs 

history <- model %>% fit(
	x_train, y_train,
	batch_size = 128,
	epochs = 6,
	verbose = 1,
	validation_split = 0.2,
)

# Evaluate the model's performance
score <- model %>% evaluate(
	x_test, y_test,
	verbose = 1
)



