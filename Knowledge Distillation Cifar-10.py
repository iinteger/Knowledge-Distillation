from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import activations

np.random.seed(0)
def softer_softmax(x, axis=-1):
    ndim = K.ndim(x)
    if ndim == 1:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
    elif ndim == 2:
        return K.softmax(x / T)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)


def softmax2softer_softmax(x):
    y = []

    for i in range(len(x)):
        distilled_x = []  # x/T = 분자
        for j in range(len(x[0])):
            distilled_x.append(np.log((x[i][j]) * sum(x[i]))/T)  # 적용된 소프트맥스를 역연산하여 원래의 확률을 구하고, T 로 나누어 정규화

        softer_x = []  # softer_softmax가 적용된 x
        for j in range(len(x[0])):
            softer_x.append(np.exp(distilled_x[j])/sum(np.exp(distilled_x)))

        y.append(softer_x)  # 분자.분모
    y = np.array(y)
    return y


dataset = cifar100
dataset_name = "cifar100"
img_rows = 32
img_cols = 32
channels = 3
T = 10.0  # T-value

result_acc = []
result_loss = []
input_shape = (img_rows, img_cols, channels)
earlystopping = EarlyStopping(monitor="val_loss", patience=10)
batch_size = 512
epoch = 50

(x_train, y_train), (x_test, y_test) = dataset.load_data()
num_classes = len(np.unique(y_train))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)


def teacher_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epoch+50,
                     verbose=2,
                     validation_data=(x_test, y_test),
                     callbacks=[earlystopping])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    result_acc.append(test_acc)
    result_loss.append(test_loss)

    print('teacher Test loss:', test_loss)
    print('teacher Test accuracy:', test_acc)


    model.save("teacher.h5")
    return model, hist

# distilled model
def student_model1(teacher_model):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation=softer_softmax))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    soft_train_pred = teacher_model.predict(x_train)
    soft_train_labels = softmax2softer_softmax(soft_train_pred)
    soft_test_pred = teacher_model.predict(x_test)
    soft_test_labels = softmax2softer_softmax(soft_test_pred)

    hist1 = model.fit(x_train, soft_train_labels,
                      epochs=epoch,
                      batch_size=batch_size,
                      verbose=1,
                      validation_data=(x_test, soft_test_labels),
                      callbacks=[earlystopping])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    result_acc.append(test_acc)
    result_loss.append(test_loss)
    print('student1 softer Test loss:', test_loss)
    print('student1 softer Test accuracy:', test_acc)
    print(model.predict(x_test))

    model.layers[-1].activation = activations.softmax  # 활성화 함수 교체
    model.save("test.h5")
    model = load_model("test.h5")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    result_acc.append(test_acc)
    result_loss.append(test_loss)
    print('student1 Test loss:', test_loss)
    print('student1 Test accuracy:', test_acc)
    print(model.predict(x_test))

    model.save("student1.h5")
    return hist1


def student_model2():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hist1 = model.fit(x_train, y_train,
                      epochs=epoch,
                      batch_size=batch_size,
                      verbose=1,
                      validation_data=(x_test, y_test),
                      callbacks=[earlystopping])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    result_acc.append(test_acc)
    result_loss.append(test_loss)

    print('student2 Test loss:', test_loss)
    print('student2 Test accuracy:', test_acc)

    model.save("student2.h5")
    return hist1


teacher, teacher_hist = teacher_model()

hist1 = student_model1(teacher)  # improved

hist2 = student_model2()

print(result_acc)
print(result_loss)

plt.plot(teacher_hist.history['accuracy'])
plt.plot(hist1.history['accuracy'])
plt.plot(hist2.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['teacher', 'student1', "student2"], loc='upper left')
plt.show()

plt.plot(teacher_hist.history['val_accuracy'])
plt.plot(hist1.history['val_accuracy'])
plt.plot(hist2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['teacher', 'student1', "student2"], loc='upper left')
plt.show()
