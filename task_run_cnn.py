import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

#Complete code at ...

def main(_):
    save_path = os.getcwd() + '/model/save_net.ckpt'
    model = CNN()
    x = model.getInput()
    y = model.inference(x)

    saver = tf.train.Saver()

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(global_init)
        sess.run(local_init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.start_queue_runners(sess=sess)

        for i in range(model.N_SAMPLES):
            x_val, y_val = sess.run([x, y])

            #Task 3:
            # - Apply Gausian filter to image of Homer
            # - Save image
            # - Push snapshot and image to GIT
            #...

            img_save_path = os.getcwd() + '/image_results/'

            # Save all original images from Homer
            img_org = Image.fromarray(x_val, 'RGB')
            img_name = img_save_path + 'img_' + str(i) + '_org.jpg'
            img_org.save(img_name)

            # Save all Gaussian filtered images from original images
            y_val = tf.cast(y_val, dtype=tf.uint8)
            img_blur = tf.squeeze(y_val).eval()
            img_blur = Image.fromarray(img_blur, 'L')
            img_blur_name = img_save_path + 'img_' + str(i) + '_blur.jpg'
            img_blur.save(img_blur_name)

        # Saver code
        saver.save(sess, save_path)

        coord.request_stop()
        coord.join(threads)
        print('done')

class CNN():
    def __init__(self):
        self.N_SAMPLES = 3
        self.DATASET = "record_test.tfrecord"
        self.IN_SHAPE = [303, 303, 3]
        pass

    def getInput(self):
        # Task 1:
        # Read TfRecord
        tfrecord_file_queue = tf.train.string_input_producer([self.DATASET], name='queue', num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(tfrecord_file_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'img': tf.FixedLenFeature([], tf.string)
                                           })

        img = tf.decode_raw(features['img'], tf.uint8)
        img = tf.reshape(img, self.IN_SHAPE)

        return img

    def inference(self, img):
        with tf.name_scope('conv1'):
            out = 1
            k_w = 5
            k_h = 5
            s = [1, 1, 1, 1]
            pad = 'VALID'
            w_shape = [k_w, k_h, 3, out]

            #Task 2
            # - Apply a 5x5 Gaussian filter to the input
            #...

            # Transform image to the in nn.conv2d required format
            img = tf.expand_dims(img, dim=0)
            img = tf.cast(img, tf.float32)

            # Define Gaussian Filter kernel with w_shape
            mask = [1,  4,  6,  4, 1,
                    4, 16, 26, 16, 4,
                    7, 26, 41, 26, 7,
                    4, 16, 26, 16, 4,
                    1,  4,  6,  4, 1]
            kernel = [i/(273.0*3.0) for i in mask] * w_shape[2]

            # Transform kernel to in nn.conv2d required format
            w_conv1 = tf.Variable(tf.constant(kernel, shape=w_shape, dtype=tf.float32))
            h_conv1 = tf.nn.conv2d(img, w_conv1, strides=s, padding=pad, name='h_conv1')
            # h_conv1 = b_conv1*h_conv1

        return h_conv1


if __name__ == '__main__':
    tf.app.run()
