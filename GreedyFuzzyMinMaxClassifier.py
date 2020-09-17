"""
в случае жадного классификатора гиперпрямоугольник каждого класса = объединению прямоугольников каждого вхождения
(в простой реализации на каждый класс есть только один гипермногоугольник, без хранения многоуголбников для каждого
входящего вектора фичей)
"""

import numpy as np


# функция проверяет можно ли расширить гиперпрямоугольники для каждого класса
def HyberboxExpansionCheck(V, W, theta, X, label, count_of_X,max_count):
    """
    возвращает матрицу. первый индекс - номер класса, второй индекс- номер гиперпрямоугольника элемента этого класса
    значение по двойному индексу - надо менять данный гипермногоугольник или нет
    """

    num_of_classes = np.shape(V)[0]

    num_of_hyperbox_in_one_class = np.shape(V)[1]

    feature_space_shape = np.shape(V)[2]

    feature_vec = X

    max_size_of_dimension_in_hyperbox = theta

    check_matrix = np.zeros((num_of_classes, num_of_hyperbox_in_one_class))

    # проход по всем классам
    for i in range(num_of_classes):
        # будем обновлять только те параметры, которые принадлежат соответствующему классу
        if i == label:

            # проход по всем доступным гипермногоугольникам внутри класса
            for j in range(count_of_X):
                if j < max_count:
                    # проход по всем фичам внутри гипермногоугольника и вычисление суммы в формуле № 3 в статье
                    sum = 0
                    for k in range(feature_space_shape):
                        sum += np.maximum(W[i][j][k], feature_vec[k]) - np.minimum(V[i][j][k], feature_vec[k])

                    if feature_space_shape * max_size_of_dimension_in_hyperbox >= sum:
                        check_matrix[i][j] = 1

    return check_matrix


def HyberboxExpansion(V, W, X, check_matrix):
    num_of_classes = np.shape(V)[0]
    num_of_hyperbox_in_one_class = np.shape(V)[1]
    feature_space_shape = np.shape(V)[2]

    for i in range(num_of_classes):
        for j in range(num_of_hyperbox_in_one_class):
            if check_matrix[i][j]:
                for k in range(feature_space_shape):
                    V[i][j][k] = np.minimum(V[i][j][k], X[k])
                    W[i][j][k] = np.maximum(W[i][j][k], X[k])


def HyperboxOverlapTestAndContraction(V, W, count_of_X_vec, max_count):
    # нужно найти нежелательное пересечение каждого гипермногуогольника с гиперногоугольникам идругих классов
    # и устранить его

    num_of_classes = np.shape(V)[0]
    num_of_hyperbox_in_one_class = np.shape(V)[1]
    feature_space_shape = np.shape(V)[2]

    # map позволит не гонять в холостую много итераций
    process = {}
    p = 0
    for i in range(num_of_classes):
        # зафиксировали номер класса

        # проверка на пересечение фиксированного класса со всеми другими классами
        for j in range(num_of_classes):

            # не нужно проверять пересечение класса самим с собой и не ненужно проверять уже проверенное пересечение
            p = +1  # номер итерации
            process.update({p: [i, j]})
            pairs = process.values()
            flag = True
            for pair in pairs:
                if pair[0] == i and pair[1] == j:
                    flag = False
                elif pair[0] == j and pair[1] == i:
                    flag = False

            if flag:

                for l in range(count_of_X_vec[i]):
                    if l<max_count:
                        # тут написана проверка уже для двух гиперпрямоугольников( как в статье )
                        for m in range(count_of_X_vec[j]):
                            if m< max_count:
                                v1 = V[i][l]
                                w1 = W[i][l]
                                v2 = V[j][m]
                                w2 = W[j][m]

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
                                V[i][l] = v1
                                W[i][l] = w1
                                V[j][m] = v2
                                W[j][m] = w2


def membership(V, W, gamma, X, count_vec, max_count):
    # математически тут должна была бы произойти конкатенация всех гипермногоугольников внутри каждого класса
    # но програмно прийдется проверять в цикле по всем гипермногоугольникам множества

    num_of_classes = np.shape(V)[0]
    num_of_hyperbox_in_one_class = np.shape(V)[1]
    feature_space_shape = np.shape(V)[2]
    sensitivity_parameter = gamma

    distribution_of_class_membership = np.zeros(num_of_classes)
    for i in range(num_of_classes):
        # пройдемся по всем доступным гипермногоугольникам внутри класса, на кажом посчитаем степень принадлежности,
        # потом вернем максимальное значение

        if count_vec[i] <= max_count:
            tmp_class_memberships = np.zeros(
                count_vec[i])  # принадлежности X к классу i в каждом гипермногоугольнике класса i
        else:
            tmp_class_memberships = np.zeros(
                max_count)  # принадлежности X к классу i в каждом гипермногоугольнике класса i

        for j in range(max_count):
            if j < count_vec[i]:
                sum = 0
                for k in range(feature_space_shape):
                    sum += (np.maximum(0, 1 - np.maximum(0, sensitivity_parameter * np.minimum(1, X[k] - W[i][j][k]))) +
                            np.maximum(0, 1 - np.maximum(0, sensitivity_parameter * np.minimum(1, V[i][j][k] - X[k]))))

                tmp_class_memberships[j] = sum / (2 * feature_space_shape)

        distribution_of_class_membership[i] = np.amax(tmp_class_memberships)

    # Возвращает уже привычное распеределние классов для X
    return distribution_of_class_membership


class Model(object):

    def __init__(self, num_of_classes, max_entities_in_one_class, len_of_input_vec, theta, gamma, backup_path):
        self.backup_path = backup_path
        self.num_of_classes = num_of_classes
        self.max_count = max_entities_in_one_class
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
            self.V = np.random.random((self.num_of_classes, self.max_count, self.len_of_input_vec))
            self.W = np.random.random((self.num_of_classes, self.max_count, self.len_of_input_vec))
            l = 0
            len = np.shape(images)[0]
            already_print = {}

            count_of_X_vec = np.zeros(self.num_of_classes, dtype=np.int)

            for i in range(np.shape(images)[0]):
                l += 1

                count_of_X_vec[np.int(labels[i])] += 1

                if count_of_X_vec[np.int(labels[i])] > self.max_count:
                    continue

                # 1 проверка неравенства
                check_matrix = HyberboxExpansionCheck(self.V, self.W, self.theta, images[i], labels[i],
                                                      count_of_X_vec[np.int(labels[i])],self.max_count)

                # 2 fuzzy_intersection_and_union
                HyberboxExpansion(self.V, self.W, images[i], check_matrix)

                # hyperbox overlap test
                HyperboxOverlapTestAndContraction(self.V, self.W, count_of_X_vec,self.max_count)

                pr = int(l / len * 100)

                if pr % 2 == 0:
                    if pr not in already_print.values():
                        already_print.update({l: pr})
                        print('passed ', pr, '%')

            np.save(self.backup_path + '\\count_of_X_greedy', count_of_X_vec)
            np.save(self.backup_path + '\\V_greedy', self.V)
            np.save(self.backup_path + '\\W_greedy', self.W)

        else:
            print('####################################')
            print('training under-trained model started')
            print('####################################')

            self.V = np.load(self.backup_path + '\\V_greedy.npy')
            self.W = np.load(self.backup_path + '\\W_greedy.npy')
            count_of_X_vec = np.load(self.backup_path + '\\count_of_X_greedy.npy')

            l = 0
            len = np.shape(images)[0]
            already_print = {}
            for i in range(np.shape(images)[0]):
                l += 1

                count_of_X_vec[np.int(labels[i])] += 1
                if count_of_X_vec[np.int(labels[i])] > self.max_count:
                    continue
                # 1 проверка неравенства
                check_matrix = HyberboxExpansionCheck(self.V, self.W, self.theta, images[i], labels[i],
                                                      count_of_X_vec[np.int(labels[i])],self.max_count)

                # 2 fuzzy_intersection_and_union
                HyberboxExpansion(self.V, self.W, images[i], check_matrix)

                # hyperbox overlap test
                HyperboxOverlapTestAndContraction(self.V, self.W, count_of_X_vec,self.max_count)

                pr = int(l / len * 100)

                if pr % 2 == 0:
                    if pr not in already_print.values():
                        already_print.update({l: pr})
                        print('passed ', pr, '%')

            np.save(self.backup_path + '\\count_of_X_greedy', count_of_X_vec)
            np.save(self.backup_path + '\\V_greedy', self.V)
            np.save(self.backup_path + '\\W_greedy', self.W)

    def eval(self, images, labels):
        # предобработаем картинки

        images = np.reshape(images, newshape=(np.shape(images)[0], 784))
        images = images / 255.0
        images = images / np.std(images) - np.mean(images)

        self.V = np.load(self.backup_path + '\\V_greedy.npy')
        self.W = np.load(self.backup_path + '\\W_greedy.npy')
        count_of_X_vec = np.load(self.backup_path + '\\count_of_X_greedy.npy')

        print('#####################')
        print('eval of model started')
        print('#####################')

        num_of_errors = 0
        l = 0
        len = np.shape(images)[0]
        already_print = {}

        for i in range(np.shape(images)[0]):
            l += 1

            distribution_of_class_membership = membership(self.V, self.W, self.gamma, images[i], count_of_X_vec,
                                                          max_count=self.max_count)
            if np.argmax(distribution_of_class_membership) != labels[i]:
                num_of_errors += 1
            print('predict:', np.argmax(distribution_of_class_membership),' really label: ',labels[i])

            pr = int(l / len * 100)
            if pr % 2 == 0 and pr != 0:
                if pr not in already_print.values():
                    already_print.update({l: pr})
                    print('passed ', pr, '%', ' accuracy: ', 100 - int(num_of_errors / l * 100), "%")


        print('total accuracy: ', 100 - num_of_errors / np.shape(images)[0] * 100, "%")
