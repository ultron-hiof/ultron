from model.conv import create_model
from load.img.dataset import load_x_dataset, load_y_dataset

X = load_x_dataset('/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/X.pickle')
y = load_y_dataset('/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/y.pickle')
# print('Dataset loaded.\n\n')

# model = create_model(shape=X.shape[1:], conv_layers=2, layer_size=32, activation_layer='relu', dense_layers=1, dense_layer_size=512, output_classes=2, output_activation='softmax')
model = create_model(shape=X.shape[1:])
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()
# model.fit(X, y, validation_split=0.2)
