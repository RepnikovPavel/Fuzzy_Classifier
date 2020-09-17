"""
    1)ко всем функциям добавляется название __training_loop__ в качетсве си-реализации namespace
    2)с помощью флагов можно тренировать отдельно взятый алгоритм или несколько алгоритомв.
    то же самое относится и к воспроизведению отчетов о тренировке алгоритма/ алгоритмов( в виде графиков )

"""

import FuzzyMinMaxClassifier as classifier
import GreedyFuzzyMinMaxClassifier as greedy_classifier

import os
import numpy as np
import time


def __training_loop__load_all_public_images_and_labels(
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


def execution_of_existing_functions(config):
    # грузим дату целиком
    pub_images, pub_labels = __training_loop__load_all_public_images_and_labels(config['recorded_train_data_filename'],
                                                                                config[
                                                                                    'recorded_train_labels_filename'])

    private_images, private_labels = __training_loop__load_all_public_images_and_labels(
        config['recorded_test_data_filename'],
        config[
            'recorded_test_labels_filename'])

    algorithms_config = config['algorithms']
    save_model_path = config['train_loop_path']
    train_data_path = config['recorded_train_data_filename']
    train_labels_path = config['recorded_train_labels_filename']

    alg_id = 'fuzzy_min_max_classifier'

    if algorithms_config[alg_id]['processing']:
        alg_config = algorithms_config[alg_id]
        gamma = 0.9
        theta = 1

        if alg_config['training']:
            training_config = alg_config['training_config']
            if training_config['training_from_zero']:
                model = classifier.Model(num_of_classes=10, len_of_input_vec=784, theta=theta, gamma=gamma,
                                         backup_path=save_model_path)

                start = time.time()
                model.train(images=pub_images, labels=pub_labels, from_zero=True)
                print('total_training_time:', time.time() - start, ' sek')

            if training_config['train_an_under-trained_model']:
                model = classifier.Model(num_of_classes=10, len_of_input_vec=784, theta=theta, gamma=gamma,
                                         backup_path=save_model_path)

                start = time.time()
                model.train(images=pub_images, labels=pub_labels, from_zero=False)
                print('total_training_time:', time.time() - start, ' sek')

        if alg_config['eval_of_model']:
            model = classifier.Model(num_of_classes=10, len_of_input_vec=784, theta=theta, gamma=gamma,
                                     backup_path=save_model_path)
            start = time.time()
            model.eval(private_images[:200], private_labels[:200])
            print('total_eval_time:', time.time() - start, ' sek')

    alg_id = 'greedy_fuzzy_min_max_classifier'
    if algorithms_config[alg_id]['processing']:
        alg_config = algorithms_config[alg_id]
        max_entities = 500
        gamma = 0.9
        theta = 1

        if alg_config['training']:
            training_config = alg_config['training_config']

            if training_config['training_from_zero']:
                model = greedy_classifier.Model(num_of_classes=10, max_entities_in_one_class=max_entities,
                                                len_of_input_vec=784, theta=theta, gamma=gamma,
                                                backup_path=save_model_path)

                start = time.time()
                model.train(images=pub_images[:450], labels=pub_labels[:450], from_zero=True)
                print('total_training_time:', time.time() - start, ' sek')

            if training_config['train_an_under-trained_model']:
                model = greedy_classifier.Model(num_of_classes=10, max_entities_in_one_class=max_entities,
                                                len_of_input_vec=784, theta=theta, gamma=gamma,
                                                backup_path=save_model_path)
                start = time.time()
                model.train(images=pub_images[500:600], labels=pub_labels[500:600], from_zero=False)
                print('total_training_time:', time.time() - start, ' sek')

        if alg_config['eval_of_model']:
            model = greedy_classifier.Model(num_of_classes=10, max_entities_in_one_class=max_entities,
                                            len_of_input_vec=784, theta=theta, gamma=gamma,
                                            backup_path=save_model_path)
            start = time.time()
            model.eval(private_images[:200], private_labels[:200])
            print('total_eval_time:', time.time() - start, ' sek')


if __name__ == '__main__':
    # создайте папку training_loop для сохранения конфига и весов модели

    # прежде чем запускать - проделайте предыдущие шаги

    root = os.getcwd()

    # все расчеты на одном ядре процессора

    execution_of_existing_functions(
        config={
            'recorded_train_data_filename': root + '\\dataset\\train\\train_data_record.npy',
            'recorded_train_labels_filename': root + '\\dataset\\train\\train_labels_record.npy',
            'recorded_test_data_filename': root + '\\dataset\\test\\test_data_record.npy',
            'recorded_test_labels_filename': root + '\\dataset\\test\\test_labels_record.npy',

            'train_loop_path': root + '\\training_loop',
            'algorithms': {
                'fuzzy_min_max_classifier': {'processing': False,
                                             'training': False,
                                             'training_config': {
                                                 'training_from_zero': True, # 200 sek
                                                 'train_an_under-trained_model': False},
                                             'eval_of_model': True   # 900 sek (all private data)
                                             },# total accuracy примерно 45 проц
                'greedy_fuzzy_min_max_classifier': {'processing': True,
                                                    'training': False,
                                                    'training_config': {
                                                        'training_from_zero': True, #  43sek  - 500 примеров
                                                        'train_an_under-trained_model': False},
                                                    'eval_of_model': True   # 4 sek per sample при 500 примерах
                                                    } #  на 500 тренировчных примерах total accuracy примерно 74.5 проц
            }
        }
    )
