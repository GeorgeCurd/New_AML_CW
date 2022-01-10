from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from FeatureSelection import test_normX_ANOVA_FS, train_normX_ANOVA_FS, train_normX_ETC_FS, test_normX_ETC_FS
from DataProcessing import train_Y, test_Y, train_normX, test_normX
from keras.regularizers import l1, l2
from matplotlib import pyplot
from keras.metrics import SensitivityAtSpecificity, Accuracy

# Very basic sequential model
model = Sequential()
model.add(Dense(12, input_dim=20, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(BatchNormalization(momentum=0.9))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','Recall','FalsePositives','TrueNegatives',
                                                                     'TruePositives','FalseNegatives'])
# fit the model
hist = model.fit(train_normX_ETC_FS, train_Y, epochs=25, batch_size=50,  validation_split=0.25)
# evaluate the model
scores = model.evaluate(test_normX_ETC_FS, test_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
print("\n%s: %.2f%%" % ('FAR', (scores[3]/(scores[3]+scores[4])*100)))


# plot loss
pyplot.plot(hist.history['accuracy'], label='train')
pyplot.plot(hist.history['val_accuracy'], label='test')
# limits = [ 0, 50, 0, 1]
# pyplot.axis(limits)
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy")
pyplot.legend()
pyplot.show()
