# Import dataset
cifar10	<- dataset_cifar10()

# Turn bitmap values to value between 1 and 0
x_train		<- cifar10$train$x/255
x_test		<- cifar10$test$x /255

# Turn labels from int into binary category label
y_train 	<- to_categorical(cifar10$train$y,	num_classes=10)
y_test 		<- to_categorical(cifar10$test$y,		num_classes=10)

model <-keras_model_sequential() %>%
	layer_conv_2d(filters = 32, kernel_size = c(3,3),
		activation = 'relu', input_shape = c(32, 32, 3), padding = "same") %>%
	layer_conv_2d(filters = 32, kernel_size =  c(3,3),
		activation = 'relu') %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(0.25) %>%
	layer_conv_2d(filters = 32, kernel_size = c(3,3),
		activation = 'relu', padding = "same") %>%
	layer_conv_2d(filters = 32, kernel_size =  c(3,3),
		activation = 'relu') %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(0.25) %>%
	layer_flatten() %>%
	layer_dense(units = 512, activation = 'relu') %>%
	layer_dropout(0.5) %>%
	layer_dense(units = 10,   activation = 'softmax')
	
# To check the resulting model is what you expected, use:
#summary(model)

# When it looks ok compile it with suitable loss function, optimization procedures and performance measures
model %>% compile( 
	loss = 'categorical_crossentropy', 
	optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),	
	metrics = 'accuracy'
)

history <- model %>% fit(
	x_train, y_train,
	batch_size = 32,
	epochs = 20,
	verbose = 1,
	validation_data = list(x_test, y_test),
	shuffle = TRUE
)

# Evaluate the model's performance
score <- model %>% evaluate(
	x_test, y_test,
	verbose = 1
)
