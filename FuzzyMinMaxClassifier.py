"""
в простой реализации на каждый класс есть только один гипермногоугольник, без хранения многоуголбников для каждого
входящего вектора фичей
"""

import numpy as np

# функция проверяет можно ли расширить гиперпрямоугольники для каждого класса
def HyberboxExpansionCheck(V, W, theta, X, label):
    """
    возвращает вектор, длинной = число классов, в каждой ячейке вектора - булевое значение(надо ли
    менять гиперпрямоугольник класса соответствующего ячейке этого вектора)
    """

    max_size_of_hyperbox = theta
    feature_space_shape = np.shape(V)[1]
    num_of_classes = np.shape(V)[0]
    image = X

    # проход по всем классам и проверка - нужно ли менять гиперпрямоугольник d в кажом классе
    check_vec = np.zeros(num_of_classes)
    for i in range(num_of_classes):
        # будем обновлять только те параметры, которые принадлежат соответствующему классу
        if i == label:
            # проход по всем фичам внутри класса и вычисление суммы в формуле № 3 в статье
            sum = 0
            for j in range(feature_space_shape):
                sum += np.maximum(W[i][j], image[j]) - np.minimum(V[i][j], image[j])

            if feature_space_shape * max_size_of_hyperbox >= sum:
                check_vec[i] = 1

    return check_vec


def HyberboxExpansion(V, W, X, check_vec):
    feature_space_shape = np.shape(V)[1]
    for i in range(np.shape(check_vec)[0]):
        if check_vec[i]:
            for j in range(feature_space_shape):
                V[i][j] = np.minimum(V[i][j], X[j])
                W[i][j] = np.maximum(W[i][j], X[j])


def HyperboxOverlapTestAndContraction(V, W):
    # нужно найти нежелательное пересечение и устранить
    num_of_classes = np.shape(V)[0]
    feature_space_shape = np.shape(V)[1]

    process = {}
    p = 0
    for i in range(num_of_classes):
        # проверка на пересечение каждого с каждым
        for j in range(num_of_classes):

            p = +1  # номер итерации
            process.update({p: [i, j]})
            pairs = process.values()

            # если до текущей p находилась такая же пара с точностью до замены местами i и j, то
            # дальше не идем

            flag = True

            for pair in pairs:
                if pair[0] == i and pair[1] == j:
                    flag = False
                elif pair[0] == j and pair[1] == i:
                    flag = False

            if flag:

                # тут написана проверка уже для двух векторов( как в статье )

                v1 = V[i]
                w1 = W[i]
                v2 = V[j]
                w2 = W[j]

                OverlapFactorOld = 1
                OverlapFactorNew = 1
                dN = - np.ones(
                    feature_space_shape)  # номера размерностей, которые будем менять, если есть пересечение dimensionNumber
                for k in range(feature_space_shape):

                    # проверяем все измерения гиперпрямоугольников на пересечение
                    if (v1[k] < v2[k] and v2[k] < w1[k] and w1[k] < w2[k]):  # case 1
                        OverlapFactorNew = np.min(w1[k] - v2[k], OverlapFactorOld)
                    elif (v2[k] < v1[k] and v1[k] < w2[k] and w2[k] < w1[k]):  # case 2
                        OverlapFactorNew = np.min(w2[k] - v1[k], OverlapFactorOld)
                    elif (v1[k] < v2[k] and v2[k] <= w2[k] and w2[k] < w1[k]):  # case 3
                        OverlapFactorNew = np.min(np.min(w2[k] - v1[k], w1[k] - v2[k]), OverlapFactorOld)
                    elif (v2[k] < v1[k] and v1[k] <= w1[k] and w1[k] < w2[k]):  # case 4
                        OverlapFactorNew = np.min(np.min(w1[i] - v2[i], w2[i] - v1[i]), OverlapFactorOld)
                    if (OverlapFactorOld - OverlapFactorNew > 0):
                        dN[k] = k
                        OverlapFactorOld = OverlapFactorNew

                for k in range(feature_space_shape):

                    if dN[k] != -1:
                        vj, wj = v1[dN[k]], w1[dN[k]]
                        vk, wk = v2[dN[k]], w2[dN[k]]

                        if (vj < vk and vk < wj and wj < wk):  # case 1
                            wj = (wj + vk) / 2
                            vk = wj

                        elif (vk < vj and vj < wk and wk < wj):  # case 2
                            wk = (wk + vj) / 2
                            vj = wk

                        elif (vj < vk and vk <= wk and wk < wj and (wk - vj) <= (wj - vk)):  # case 3a
                            vj = wk

                        elif (vj < vk and vk <= wk and wk < wj and (wk - vj) >= (wj - vk)):  # case 3b
                            wj = vk

                        elif (vk < vj and vj <= wj and wj < wk and (wk - vj) <= (wj - vk)):  # case 4a
                            wk = vj

                        elif (vk < vj and vj <= wj and wj < wk and (wk - vj) >= (wj - vk)):  # case 4b
                            vk = wj

                        v1[dN[k]], w1[dN[k]] = vj, wj
                        v2[dN[k]], w2[dN[k]] = vk, wk
                V[i] = v1
                W[i] = w1
                V[j] = v2
                W[j] = w2


def membership(V, W, gamma, X):
    num_of_classes = np.shape(V)[0]
    feature_space_shape = np.shape(V)[1]
    sensitivity_parameter = gamma

    distribution_of_class_membership = np.zeros(num_of_classes)
    for i in range(num_of_classes):
        sum = 0
        for j in range(feature_space_shape):
            sum += (np.maximum(0, 1 - np.maximum(0, sensitivity_parameter * np.minimum(1, X[j] - W[i][j]))) +
                    np.maximum(0, 1 - np.maximum(0, sensitivity_parameter * np.minimum(1, V[i][j] - X[j]))))

        distribution_of_class_membership[i] = sum / (2 * feature_space_shape)

    return distribution_of_class_membership


class Model(object):

    def __init__(self, num_of_classes, len_of_input_vec,theta,gamma,backup_path):
        self.backup_path = backup_path
        self.num_of_classes = num_of_classes
        self.len_of_input_vec = len_of_input_vec
        self.V = []
        self.W = []
        self.theta = theta
        self.gamma = gamma

    def train(self, images, labels, from_zero):
        # предобработка

        images = np.reshape(images, newshape=(np.shape(images)[0], 784))
        images = images / 255.0
        images = images / np.std(images) - np.mean(images)

        # сам алгоритм
        if from_zero:
            print('##########################')
            print('training from zero started')
            print('##########################')
            self.V = np.random.random((self.num_of_classes,self.len_of_input_vec))
            self.W = np.random.random((self.num_of_classes, self.len_of_input_vec))
            l = 0
            len = np.shape(images)[0]
            already_print = {}
            for i in range(np.shape(images)[0]):
                l += 1
                # 1 проверка неравенства
                check_vec = HyberboxExpansionCheck(self.V, self.W, self.theta, images[i], labels[i])

                # 2 fuzzy_intersection_and_union
                HyberboxExpansion(self.V, self.W, images[i], check_vec)

                # hyperbox overlap test
                HyperboxOverlapTestAndContraction(self.V, self.W)

                pr=int(l / len * 100)
                if pr % 10 == 0:
                    if pr not in already_print.values():
                        already_print.update({l:pr})
                        print('passed ', pr, '%')


            np.save(self.backup_path + '\\V', self.V)
            np.save(self.backup_path + '\\W', self.W)

        else:
            print('####################################')
            print('training under-trained model started')
            print('####################################')

            self.V = np.load(self.backup_path+'\\V.npy')
            self.W = np.load(self.backup_path + '\\W.npy')
            l = 0
            len = np.shape(images)[0]
            already_print ={}
            for i in range(np.shape(images)[0]):
                l += 1
                # 1 проверка неравенства
                check_vec = HyberboxExpansionCheck(self.V, self.W, self.theta, images[i], labels[i])

                # 2 fuzzy_intersection_and_union
                HyberboxExpansion(self.V, self.W, images[i], check_vec)

                # hyperbox overlap test
                HyperboxOverlapTestAndContraction(self.V, self.W)

                pr=int(l / len * 100)
                if pr % 10 == 0:
                    if pr not in already_print.values():
                        already_print.update({l:pr})
                        print('passed ', pr, '%')

            np.save(self.backup_path + '\\V', self.V)
            np.save(self.backup_path + '\\W', self.W)


    def eval(self, images, labels):
        # предобработаем картинки

        images = np.reshape(images, newshape=(np.shape(images)[0], 784))
        images = images / 255.0
        images = images / np.std(images) - np.mean(images)

        print('#####################')
        print('eval of model started')
        print('#####################')

        self.V = np.load(self.backup_path + '\\V.npy')
        self.W = np.load(self.backup_path + '\\W.npy')

        num_of_errors = 0
        l = 0
        len = np.shape(images)[0]
        already_print = {}
        for i in range(np.shape(images)[0]):
            l += 1
            distribution_of_class_membership = membership(self.V, self.W, self.gamma, images[i])
            if np.argmax(distribution_of_class_membership) != labels[i]:
                num_of_errors += 1

            print('predict:', np.argmax(distribution_of_class_membership), ' really label: ', labels[i])

            pr= int(l / len * 100)
            if pr % 10 == 0:
                if pr not in already_print.values():
                    already_print.update({l: pr})
                    print('passed ', pr, '%', ' accuracy: ', 100 - int(num_of_errors / len * 100), "%")

        print('total accuracy: ', 100- num_of_errors / np.shape(images)[0] * 100, "%")

