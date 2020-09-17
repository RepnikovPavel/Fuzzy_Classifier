import struct as st
import numpy as np
import os


def __make_dataset__get_images_from_idx(path):
    """
        возращает np array shape=(num_of_image,height,width)
        1)работает конкретно под конкретный формат файла .idx3-ubyte
        2)читается все разом
    """
    # "https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1"
    file = open(path, 'rb')
    file.seek(0)
    magic = st.unpack('>4B', file.read(4))
    num_of_images = st.unpack('>I', file.read(4))[0]
    num_of_rows = st.unpack('>I', file.read(4))[0]
    num_of_column = st.unpack('>I', file.read(4))[0]
    images_array = np.zeros((num_of_images, num_of_rows, num_of_column))

    n_bytes_total = num_of_images * num_of_rows * num_of_column * 1
    images_array = np.asarray(st.unpack('>' + 'B' * n_bytes_total, file.read(n_bytes_total)), dtype=np.uint8).reshape(
        (num_of_images, num_of_rows, num_of_column))

    return images_array


def __make_dataset__get_labels_from_idx(path):
    """
        возращает np array shape=(num_of_label)
        1)работает конкретно под конкретный формат файла .idx1-ubyte
        2)читается все разом
    """
    # "https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1"
    file = open(path, 'rb')
    file.seek(0)
    magic = st.unpack('>4B', file.read(4))
    num_of_labels = st.unpack('>I', file.read(4))[0]
    images_array = np.zeros((num_of_labels))

    n_bytes_total = num_of_labels * 1
    labels_array = np.asarray(st.unpack('>' + 'B' * n_bytes_total, file.read(n_bytes_total)), dtype=np.uint8).reshape(
        (num_of_labels))

    return labels_array


def __make_dataset__build_mnist_dataset(train_data_path,
                                        train_labels_path,
                                        test_data_path,
                                        test_labels_path,
                                        record_train_data_path,
                                        record_train_labels_path,
                                        record_test_data_path,
                                        record_test_labels_path
                                        ):
    """
        результат работы: в указанные пути будут записаны датасет
        train и датасет test
        состоящие из np.array = из сэмплов (картинка, метка)  shape = (( image  ),( label ))

        1)разделение на train и test происходит на уровне датасета непосредственно из сырых данных,
        не относящегося к обучению(никаих разделений для кросс валидации, никаих
        аугментаций)

        2)в случае конкретно MNIST под train_path,test_path имеется в виду
         непосредственно имя файла(потому что MNIST лежит целиком в 4х бинарниках)
         на практике обычно дается именно директория и имена файлов нужно будет
         доставать автоматически, поэтому я выбрал именно такие названия с заделом на будущее
    """

    root = os.getcwd() + '\\'

    train_images = __make_dataset__get_images_from_idx(root + train_data_path)
    train_labels = __make_dataset__get_labels_from_idx(root + train_labels_path)
    test_images = __make_dataset__get_images_from_idx(root + test_data_path)
    test_labels = __make_dataset__get_labels_from_idx(root + test_labels_path)

    train_data_filename = record_train_data_path + '\\train_data_record'
    train_labels_filename = record_train_labels_path + '\\train_labels_record'
    test_data_filename = record_test_data_path + '\\test_data_record'
    test_labels_filename = record_test_labels_path + '\\test_labels_record'

    np.save(root + train_data_filename, train_images)
    np.save(root + train_labels_filename, train_labels)
    np.save(root + test_data_filename, test_images)
    np.save(root + test_labels_filename, test_labels)


if __name__ == '__main__':

    # прежде чем запустить - в папке проекта создайте папку row_data и папку dataset

    __make_dataset__build_mnist_dataset(
        train_data_path='row_data\\train-images.idx3-ubyte',
        train_labels_path='row_data\\train-labels.idx1-ubyte',
        test_data_path='row_data\\t10k-images.idx3-ubyte',
        test_labels_path='row_data\\t10k-labels.idx1-ubyte',

        record_train_data_path='dataset\\train',
        record_train_labels_path='dataset\\train',
        record_test_data_path='dataset\\test',
        record_test_labels_path='dataset\\test'
    )
