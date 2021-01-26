from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=5000, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--test_image_height", type=int, default=180, help="height of test image")
parser.add_argument("--test_image_width", type=int, default=256, help="width of test image")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 64  # should be even number
IMAGE_HEIGHT = a.test_image_height  # should be even number
IMAGE_WIDTH = a.test_image_width  # should be even number

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def gen_conv(batch_input, out_channels, stride):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02) 
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=3, strides=(stride, stride), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels, stride):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=3, strides=(stride, stride), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*", "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*", "*.png"))
        decode = tf.image.decode_png

    print("len = ", len(input_paths))

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 1, message="image does not have 1 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH * 2, 1])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//2,:])
        b_images = preprocess(raw_input[:,width//2:,:])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        if a.mode == "train":
            # crop image
            h = r.get_shape().as_list()[0]
            w = r.get_shape().as_list()[1]
            h_offset = tf.cast(tf.floor(tf.random_uniform([1], 0, h - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            w_offset = tf.cast(tf.floor(tf.random_uniform([1], 0, w - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            r = tf.image.crop_to_bounding_box(r, h_offset[0], w_offset[0], CROP_SIZE, CROP_SIZE)
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    print(inputs_batch.get_shape().as_list())

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 64, 64, in_channels] => [batch, 32, 32, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf, 2)
        layers.append(output)

    layer_specs = [
        #(output layers, pooling)
        (a.ngf, False),     # encoder_2: [batch, 32, 32, ngf] => [batch, 32, 32, ngf ]
        (a.ngf * 2, True),  # encoder_3: [batch, 32, 32, ngf] => [batch, 16, 16, ngf * 2]
        (a.ngf * 2, False), # encoder_4: [batch, 16, 16, ngf * 2] => [batch, 16, 16, ngf * 2]
        (a.ngf * 4, True),  # encoder_5: [batch, 16, 16, ngf * 2] => [batch, 8, 8, ngf * 4]
        (a.ngf * 4, False), # encoder_6: [batch, 8, 8, ngf * 4] => [batch, 8, 8, ngf * 4]
        (a.ngf * 8, True),  # encoder_7: [batch, 8, 8, ngf * 4] => [batch, 4, 4, ngf * 8]
        (a.ngf * 8, False),  # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 4, 4, ngf * 8]
        (a.ngf * 8, True),  # encoder_9: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        (a.ngf * 8, True),  # encoder_10: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels, pooling in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            s = 2 if pooling == True else 1
            convolved = gen_conv(rectified, out_channels, s)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        #(output layers, dropout, pooling, skip layers)
        (a.ngf * 8, 0.0, True, False),      # decoder_10: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8]
        (a.ngf * 8, 0.0, True, True),       # decoder_9: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8]
        (a.ngf * 8, 0.0, False, True),      # decoder_8: [batch, 4, 4, ngf * 8 * 2] => [batch, 4, 4, ngf * 8]
        (a.ngf * 4, 0.0, True, False),      # decoder_7: [batch, 4, 4, ngf * 8] => [batch, 8, 8, ngf * 4]
        (a.ngf * 4, 0.0, False, True),      # decoder_6: [batch, 8, 8, ngf * 4 * 2] => [batch, 8, 8, ngf * 4]
        (a.ngf * 2, 0.0, True, False),      # decoder_5: [batch, 8, 8, ngf * 4] => [batch, 16, 16, ngf * 2]
        (a.ngf * 2, 0.0, False, True),      # decoder_4: [batch, 16, 16, ngf * 2 * 2] => [batch, 16, 16, ngf * 2]
        (a.ngf, 0.0, True, False),          # decoder_3: [batch, 16, 16, ngf * 2] => [batch, 32, 32, ngf]
        (a.ngf, 0.0, False, True),          # decoder_2: [batch, 32, 32, ngf * 2] => [batch, 32, 32, ngf]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout, pooling, skip) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if skip == True:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            else:
                input = layers[-1]
            rectified = tf.nn.relu(input)
            s = 2 if pooling == True else 1
            output = gen_deconv(rectified, out_channels, s)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 32, 32, ngf] => [batch, 64, 64, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = layers[-1]
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels, 2)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1] + generator_inputs    # residual


def create_model(inputs, targets):
    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    with tf.variable_scope("generator_loss"):
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_L1

    if a.mode == "train":
        with tf.variable_scope("generator_train"):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("create_model/generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([gen_loss_L1])


        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        return Model(
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )
    else:
        return Model(
            gen_loss_L1=gen_loss_L1,
            gen_grads_and_vars=None,
            outputs=outputs,
            train=None,
        )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

def save_images_test(fetches, step):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        class_name = name.split('_')[0]
        if not os.path.exists(os.path.join(a.output_dir, "images", class_name)):
            os.makedirs(os.path.join(a.output_dir, "images", class_name))
        fileset = {"name": name, "step": step}
        fileset["outputs"] = name
        filename = name + ".png"
        out_path = os.path.join(a.output_dir, "images", class_name, filename)
        contents = fetches["outputs"][i]
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    return filesets

def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def find_patch_and_padding(image_len, patch_len):
    patch_cnt = int((2 * image_len) // patch_len)
    padding = int((patch_len * (patch_cnt + 1) / 2 - image_len) / 2)
    return (patch_cnt , padding)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    if a.mode == "test":
        patch_h_cnt, padding_h = find_patch_and_padding(IMAGE_HEIGHT, CROP_SIZE)
        patch_w_cnt, padding_w = find_patch_and_padding(IMAGE_WIDTH, CROP_SIZE)

        paddings = [[0,0],[padding_h,padding_h],[padding_w,padding_w],[0,0]]
        inputs_pad = tf.pad(examples.inputs, paddings, "REFLECT")
        targets_pad = tf.pad(examples.targets, paddings, "REFLECT")

        IMAGE_PADDING_HEIGHT = IMAGE_HEIGHT + 2 * padding_h
        IMAGE_PADDING_WIDTH = IMAGE_WIDTH + 2 * padding_w
        outputs = tf.zeros([1, IMAGE_PADDING_HEIGHT, IMAGE_PADDING_WIDTH, 1], dtype=tf.float32)

        first = True
        # combine patchs into images
        for row in range(patch_h_cnt):
            for col in range(patch_w_cnt):
                row_index = int(row * CROP_SIZE / 2)
                col_index = int(col * CROP_SIZE / 2)
                if first == True:
                    with tf.variable_scope("create_model"):
                        model = create_model(tf.slice(inputs_pad, [0, row_index, col_index, 0], [1, CROP_SIZE, CROP_SIZE, 1]), tf.slice(targets_pad, [0, row_index, col_index, 0], [1, CROP_SIZE, CROP_SIZE, 1]))
                    first = False
                else:
                    with tf.variable_scope("create_model", reuse=True):
                        model = create_model(tf.slice(inputs_pad, [0, row_index, col_index, 0], [1, CROP_SIZE, CROP_SIZE, 1]), tf.slice(targets_pad, [0, row_index, col_index, 0], [1, CROP_SIZE, CROP_SIZE, 1]))
                paddings = [[0,0],[row_index, IMAGE_PADDING_HEIGHT - CROP_SIZE - row_index],[col_index, IMAGE_PADDING_WIDTH - CROP_SIZE - col_index],[0,0]]
                outputs = outputs + tf.pad(model.outputs, paddings, "CONSTANT")

        CROP_HALF = int(CROP_SIZE / 2)        
        o_11 = tf.pad(tf.slice(outputs, [0, 0, 0, 0], [1, CROP_HALF, CROP_HALF, 1]),
                        [[0,0],[0,IMAGE_PADDING_HEIGHT - CROP_HALF],[0,IMAGE_PADDING_WIDTH - CROP_HALF],[0,0]], "CONSTANT")
        o_12 = tf.pad(tf.slice(outputs, [0, 0, IMAGE_PADDING_WIDTH - CROP_HALF, 0], [1, CROP_HALF, CROP_HALF, 1]),
                        [[0,0],[0,IMAGE_PADDING_HEIGHT - CROP_HALF],[IMAGE_PADDING_WIDTH - CROP_HALF, 0],[0,0]], "CONSTANT")
        o_13 = tf.pad(tf.slice(outputs, [0, IMAGE_PADDING_HEIGHT - CROP_HALF, 0, 0], [1, CROP_HALF, CROP_HALF, 1]),
                        [[0,0],[IMAGE_PADDING_HEIGHT - CROP_HALF,0],[0,IMAGE_PADDING_WIDTH - CROP_HALF],[0,0]], "CONSTANT")
        o_14 = tf.pad(tf.slice(outputs, [0, IMAGE_PADDING_HEIGHT - CROP_HALF, IMAGE_PADDING_WIDTH - CROP_HALF, 0], [1, CROP_HALF, CROP_HALF, 1]),
                        [[0,0],[IMAGE_PADDING_HEIGHT - CROP_HALF, 0],[IMAGE_PADDING_WIDTH - CROP_HALF, 0],[0,0]], "CONSTANT")

        o_21 = tf.pad(tf.slice(outputs, [0, 0, CROP_HALF, 0], [1, CROP_HALF, IMAGE_PADDING_WIDTH - 2 * CROP_HALF, 1]),
                        [[0,0],[0, IMAGE_PADDING_HEIGHT - CROP_HALF],[CROP_HALF, CROP_HALF],[0,0]], "CONSTANT")
        o_22 = tf.pad(tf.slice(outputs, [0, CROP_HALF, 0, 0], [1, IMAGE_PADDING_HEIGHT - 2 * CROP_HALF, CROP_HALF, 1]),
                        [[0,0],[CROP_HALF, CROP_HALF],[0, IMAGE_PADDING_WIDTH - CROP_HALF],[0,0]], "CONSTANT")
        o_23 = tf.pad(tf.slice(outputs, [0, IMAGE_PADDING_HEIGHT - CROP_HALF, CROP_HALF, 0], [1, CROP_HALF, IMAGE_PADDING_WIDTH - 2 * CROP_HALF, 1]),
                        [[0,0],[IMAGE_PADDING_HEIGHT - CROP_HALF, 0],[CROP_HALF, CROP_HALF],[0,0]], "CONSTANT")
        o_24 = tf.pad(tf.slice(outputs, [0, CROP_HALF, IMAGE_PADDING_WIDTH - CROP_HALF, 0], [1, IMAGE_PADDING_HEIGHT - 2 * CROP_HALF, CROP_HALF, 1]),
                        [[0,0],[CROP_HALF, CROP_HALF],[IMAGE_PADDING_WIDTH - CROP_HALF, 0],[0,0]], "CONSTANT")
        o_4 = tf.pad(tf.slice(outputs, [0, CROP_HALF, CROP_HALF, 0], [1, IMAGE_PADDING_HEIGHT - 2 * CROP_HALF, IMAGE_PADDING_WIDTH - 2 * CROP_HALF, 1]),
                        [[0,0],[CROP_HALF, CROP_HALF],[CROP_HALF, CROP_HALF],[0,0]], "CONSTANT")

        outputs = o_11 + o_12 + o_13 + o_14 + (o_21 + o_22 + o_23 + o_24) / 2 + o_4 / 4
        outputs = tf.slice(outputs, [0, padding_h, padding_w, 0], [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        outputs = deprocess(outputs)
    else:
        with tf.variable_scope("create_model"):
            model = create_model(examples.inputs, examples.targets)
        outputs = deprocess(model.outputs)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)
    if a.mode == "train":
        for grad, var in model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images_test(results, step)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                #index_path = append_index(filesets)
            #print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
