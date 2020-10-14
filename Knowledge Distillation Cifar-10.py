from tensorflow.keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
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
        return K.softmax(x / T)  # normalize with T
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)


# 사용자 정의 활성화함수는 load가 안되기 때문에 일반 softmax 값을 T가 적용된 값으로 다시 계산
def softmax2softer_softmax(x):
    y = []

    for i in range(len(x)):
        distilled_x = []
        distilled_xAppend = distilled_x.append
        for j in range(len(x[0])):
            distilled_xAppend(np.log((x[i][j]) * sum(x[i])) / T)  # 적용된 소프트맥스를 역연산하여 원래의 확률을 구하고, T 로 나누어 정규화

        softer_x = []
        softer_xAppend = softer_x.append
        for j in range(len(x[0])):
            softer_xAppend(np.exp(distilled_x[j]) / sum(np.exp(distilled_x)))

        y.append(softer_x)
    y = np.array(y)
    return y


dataset = cifar10
dataset_name = "cifar10"
(x_train, y_train), (x_test, y_test) = dataset.load_data()
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
if len(x_train.shape) == 4:
    img_channels = x_train.shape[3]
else:
    img_channels = 1

input_shape = (img_rows, img_cols, img_channels)
T = 10.0  # T-value
batch_size = 512
epoch = 100  # 증류가 충분히 될 수 있도록 epoch가 필요
dropout_rate = 0.4
teacher_dense = 512
student_dense = 512
num_classes = len(np.unique(y_train))
result_acc = []
result_loss = []
earlystopping = EarlyStopping(monitor="val_loss", patience=10)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)


# modeling

def teacher_model():
    print("teacher training start")

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(teacher_dense))
    model.add(Dropout(dropout_rate))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epoch,
                     verbose=0,
                     validation_data=(x_test, y_test),
                     callbacks=[earlystopping])

    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    # result_acc.append(test_acc)
    # result_loss.append(test_loss)
    #
    # print('teacher Test loss:', test_loss)
    # print('teacher Test accuracy:', test_acc)

    model.save("models/teacher-" + dataset_name + ".h5")
    return model, hist


# distilled(student) model
def student_model1(teacher_model):
    print("student training start")

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(student_dense))
    model.add(Dropout(dropout_rate))
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
                      verbose=0,
                      validation_data=(x_test, soft_test_labels),
                      callbacks=[earlystopping])

    # 추론은 일반 softmax로 해야하기 때문에 변경 후 다시 load
    model.layers[-1].activation = activations.softmax
    model.save("models/student1-" + dataset_name + ".h5")
    model = load_model("models/student1-" + dataset_name + ".h5")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    result_acc.append(test_acc)
    result_loss.append(test_loss)
    print('student1 Test loss:', test_loss)
    print('student1 Test accuracy:', test_acc)

    return hist1


# normal model
def student_model2():
    print("normal model training start")

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(student_dense))
    model.add(Dropout(dropout_rate))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hist1 = model.fit(x_train, y_train,
                      epochs=epoch,
                      batch_size=batch_size,
                      verbose=0,
                      validation_data=(x_test, y_test),
                      callbacks=[earlystopping])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    result_acc.append(test_acc)
    result_loss.append(test_loss)

    print('student2 Test loss:', test_loss)
    print('student2 Test accuracy:', test_acc)

    model.save("models/student2-" + dataset_name + ".h5")
    return hist1

# main

teacher, teacher_hist = teacher_model()

hist1 = student_model1(teacher)  # improved

hist2 = student_model2()

# result

print("distilled model, normal model")
print("acc :", result_acc)
print("loss :",result_loss)

# training visualizing

# plt.plot(hist1.history['accuracy'])
# plt.plot(hist2.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['student1', "student2"], loc='upper left')
# plt.show()
#
# plt.plot(hist1.history['val_accuracy'])
# plt.plot(hist2.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['student1', "student2"], loc='upper left')
# plt.show()
#
# plt.plot(hist1.history['loss'])
# plt.plot(hist2.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['student1', "student2"], loc='upper left')
# plt.show()
#
# plt.plot(hist1.history['val_loss'])
# plt.plot(hist2.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['student1', "student2"], loc='upper left')
# plt.show()