import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def __EDA__show_images_from_np_array(m, n, start_index, images, labels):
    """
        1)m - число картинок по вертикали
        2)n- число картинок по горизонтали
        3)start_index - c какой картинки в массиве картинко начинать рисовать

        пример:
            show_images_from_np_array(4, 8, 10, train_images, train_labels)
    """
    # 'https://pyprog.pro/mpl/mpl_axis_ticks.html'
    a = int(m)
    b = int(n)
    fig, axarr = plt.subplots(a, b, figsize=(19, 9))

    for i in range(a):
        for j in range(b):
            axarr[i, j].imshow(images[start_index + i * a + j][0])
            axarr[i, j].set_title(labels[start_index + i * a + j][0])
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])


def __EDA__load_all_public_images_and_labels(
        train_images_filename,
        train_labels_filename
):
    """
        на вход: имена файлов с расширением .npy в которых лежат картинки и метки
        на выход: список из np.array (картинки, метки)
    """
    all_train_images = np.load(train_images_filename)
    all_train_labels = np.load(train_labels_filename)
    return all_train_images, all_train_labels


def __EDA__show_random_samples(m, n, images, labels):
    """
        рисуется m*n картинок с метками в качесте title
        картинки берутся подряд друг за другом, начиная с рандомной позиции
    """

    start_index = np.random.randint(low=0, high=np.shape(images)[0] - m * n, size=1, dtype=np.uint16)
    __EDA__show_images_from_np_array(m, n, start_index, images, labels)


def __EDA__show_distribution_of_classes(train_labels):
    """
        вход: np.array с метками
        реузльтат работы: в отдельной fig риуется гитограмма распределения классов
    """
    fig, ax = plt.subplots(figsize=(19, 9))
    sns.set(color_codes=True)
    sns.distplot(train_labels)
    ax.set_title('distribution_of_classes')


def __EDA__execution_of_existing_functions(
        recorded_train_data_filename,
        recorded_train_labels_filename
):
    all_train_images, all_train_labels = __EDA__load_all_public_images_and_labels(recorded_train_data_filename,
                                                                                  recorded_train_labels_filename)

    __EDA__show_random_samples(5, 10, all_train_images, all_train_labels)
    __EDA__show_distribution_of_classes(all_train_labels)
    plt.show()

    print('executed')


if __name__ == '__main__':
    # прежде чем запускать - проделайте предыдущие шаги

    root = os.getcwd()
    __EDA__execution_of_existing_functions(
        recorded_train_data_filename=root + '\\dataset\\train\\train_data_record.npy',
        recorded_train_labels_filename=root + '\\dataset\\train\\train_labels_record.npy'
    )
