from PIL import Image
import numpy as np
import os
import math
import random


image_dir = 'data/image_2/'
label_dir = 'data/label_2/'
dynamic_label_dir = 'data/dynamic_label_2/'
training_image_dir = 'data/training_image_2/'
validation_image_dir = 'data/validation_image_2/'


def read_label(data_name):
    label_file = open('{0}{1}.txt'.format(label_dir, data_name), 'r')
    items = [line.strip().split(' ') for line in label_file.readlines()]
    label = [[item[0], item[4], item[5], item[6], item[7]] for item in items]

    dynamic_label_path = '{0}{1}.txt'.format(dynamic_label_dir, data_name)

    if os.path.exists(dynamic_label_path):
        dynamic_label_file = open(dynamic_label_path, 'r')
        dynamic_label = \
            [line.strip() for line in dynamic_label_file.readlines()]

        for i in range(len(dynamic_label)):
            label[i].append(dynamic_label[i])
    else:
        for i in range(len(label)):
            label[i].append('n')

    return label


def crop_image():
    if not os.path.exists(training_image_dir):
        os.makedirs(training_image_dir)

    if not os.path.exists(validation_image_dir):
        os.makedirs(validation_image_dir)

    data_names = sorted([f.split('.')[0] for f in os.listdir(image_dir)])

    for data_name in data_names:
        label = read_label(data_name)
        image = Image.open('{0}{1}.png'.format(image_dir, data_name))

        count = 0

        if random.random() < 0.9:
            cropped_image_dir = training_image_dir
        else:
            cropped_image_dir = validation_image_dir

        for item in label:
            if item[0] == 'Car' or item[0] == 'Van' or item[0] == 'Truck':
                center_x = round((float(item[1]) + float(item[3])) / 2.0)
                center_y = round((float(item[2]) + float(item[4])) / 2.0)
                width = math.ceil(float(item[3]) - float(item[1]))
                height = math.ceil(float(item[4]) - float(item[2]))

                cropped_lenght = max(width, height) + 4

                left = center_x - cropped_lenght / 2
                top = center_y - cropped_lenght / 2
                right = center_x + cropped_lenght / 2
                bottom = center_y + cropped_lenght / 2

                cropped_image = image.crop((left, top, right, bottom))

                if item[5] != 'n':
                    cropped_image_name = '{0}_{1}d.png'.format(data_name,
                                                               count)
                else:
                    cropped_image_name = '{0}_{1}s.png'.format(data_name,
                                                               count)

                cropped_image.save('{0}{1}'.format(cropped_image_dir,
                                                   cropped_image_name), 'PNG')

                count += 1


if __name__ == "__main__":
    crop_image()
