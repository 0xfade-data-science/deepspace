#import keras
#from keras import layers, Sequential
#from keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from deepspace.transformers.model.Abstract import Abstract as AbstractModel
from deepspace.DataSpace import DataSpace
from deepspace.base import Base
from deepspace.Initialize import Initialize

class NeuralNetwork(AbstractModel):
    def __init__(self, seed=Initialize.seed, denses=[],
                 learing_rate=0.01, optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'],
                 test_size=0.2, epochs=50, batch_size=32, verbose=1) :
        Base.__init__(self, '=', 50)
        AbstractModel.__init__(self)
        self.seed = seed
        self.denses = denses
        self.test_size = test_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.learing_rate = 0.01
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def get_model(self):
        return self._model
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
    def init_ds(self, ds):
        ds._model = self.get_model()
    def create_model_fit(self):
        self._model = Sequential()
        if len(self.denses) <= 0:
            self.denses = [
                # The amount of nodes (dimensions) in hidden layer should be the average of input and output layers, in this case 64.
                # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
                Dense(activation='relu',
                        input_dim=self.ds.x.shape[1], units=64),
                # Add 1st hidden layer
                Dense(32, activation='relu'),
                # Adding the output layer
                # Notice that we do not need to specify input dim.
                # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
                # We use the sigmoid because we want probability outcomes
                Dense(1, activation='sigmoid')
            ]
        for d in self.denses:
            self._model.add(d)
        #or 
        #self._model = Sequential(self.denses)

        # Create optimizer with default learning rate
        # Compile the model
        self._model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self._model.summary()

        self.ds.history = self._model.fit(self.ds.x_train, self.ds.y_train, 
                                     validation_split=self.test_size, 
                                     epochs=self.epochs, 
                                     batch_size=self.batch_size,
                                     verbose=self.verbose)




