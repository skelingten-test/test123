import pandas as pd
import time
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

path = '/content/gdrive/MyDrive/cipherfiles/'
mastercsvpath = path + 'main.csv'
class_count = 6
c0 = "Blowfish"
c1 = "Caesar"
c2 = "DES"
c3 = "MD5"
c4 = "Monoalphabetic"
c5 = "RSA"

masterdf = pd.read_csv(mastercsvpath)

masterdf = masterdf.iloc[:,1:]

def returntrainedmodel(input_number, class_count):
  tf.keras.backend.clear_session()
  model = Sequential()
  model.add(Dense(4096, activation='relu',name='First_Layer', input_shape=(input_number,)))
  model.add(Dropout(rate = 0.5,name='Dropout_Layer'))
  model.add(Dense(2048, activation='relu',name='Second_Layer'))
  model.add(Dense(1024, activation='relu',name='Third_Layer'))
  model.add(Dropout(rate = 0.5,name='Dropout_Layer2'))
  model.add(Dense(512, activation='relu',name='Fourth_Layer'))
  model.add(Dense(class_count, activation='softmax',name='Fully_Connected_Layer'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model



masterdf = masterdf.sample(frac = 1)
kfold = KFold(n_splits = 10, shuffle = True)

resultcolumns =["Fold","Train Accuracy","Validation Accuracy", "Test Accuracy"]
resultdf = pd.DataFrame(columns = resultcolumns)
input_number = masterdf.shape[1] - 1
class_count = 6

count = 1
epoch = 2

for train, test in kfold.split(masterdf):
  tempdict = dict()
  tempdict["Fold"] = count
  print("@@@@@"*10)
  print("-----"*10)
  print("Split", count)
  
  count += 1
  print("-----"*10)
  print("@@@@@"*10)

  # Prepare test train data
  X_train = masterdf.iloc[train].drop(columns=['Class'])
  print(X_train.shape)
  X_test = masterdf.iloc[test]['Class']
  print(X_test.shape)
  y_train = masterdf.iloc[train].drop(columns=['Class'])
  print(y_train.shape)

  y_test = masterdf.iloc[test]['Class']
  print(y_test.shape)
  y_test = np_utils.to_categorical(y_test)
  y_train = np_utils.to_categorical(y_train)
  print(y_test.shape)

  tf.keras.backend.clear_session()

  # Train model
  history = model.fit(X_train, y_train, validation_split=0.25, verbose=1, epochs=epoch)

  # predict from test data
  y_pred = model.predict(X_test)

  # Turn predicted into proper class
  preds_classes = np.argmax(y_pred, axis=-1)
  preds_classes

  # Confusion matrix

  y_test = np.argmax(y_test, axis=-1)
  cfm = confusion_matrix(y_test, preds_classes)
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cfm, display_labels = [c0, c1, c2, c3, c4, c5])
  cm_display.plot()
  cfmname = 'Fold '+str(count)+' cfm.png'
  plt.savefig(cfmname, dpi = 300)

  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  accname = 'Fold '+str(count)+' acc.png'
  plt.savefig(accname, dpi = 300)
  
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  lossname = 'Fold '+str(count)+' loss.png'
  plt.savefig(lossname, dpi = 300)



  tempdict["Train Accuracy"] = history.history["accuracy"][epoch-1]
  tempdict["Validation Accuracy"] = history.history["val_accuracy"][epoch-1]
  tempdict["Test Accuracy"] = accuracy_score(y_test, preds_classes)
  
  resultdf = resultdf.append(tempdict, ignore_index=True)

  df_indexed = df.set_index('Technique')
  dfi.export(df_indexed_style, "seminar/df_indexed_styled.png", dpi=1280, table_conversion="selenium", max_rows=-1)





















