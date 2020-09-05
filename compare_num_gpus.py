import multiprocessing
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import datetime
import sys, getopt

# TODO: try this with lambda callback
class EpochTime(tf.keras.callbacks.Callback):
    def __init__(self):
        self.start = None
        self.end = None

    def on_epoch_begin(self, epoch, logs=None):
        self.start = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        self.end = tf.timestamp()
        print("\nEpoch {} Time: {}".format(epoch, self.end - self.start))
        tf.summary.scalar('epoch time', data=(self.end - self.start), step=epoch)

def create_model(input_shape, model_arch=0):
    X_input = tf.keras.layers.Input(input_shape)

    print("model arch:", str(model_arch))

    if model_arch == 0:
        X = tf.keras.layers.Conv2D(16, (3,3), strides=(1,1), activation="relu")(X_input)
        X = tf.keras.layers.MaxPooling2D()(X)

        X = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation="relu")(X)
        X = tf.keras.layers.MaxPooling2D()(X)

        X = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation="relu")(X)
        X = tf.keras.layers.MaxPooling2D()(X)
    elif model_arch == 1:
        X = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation="relu")(X_input)
        X = tf.keras.layers.MaxPooling2D()(X)

        X = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation="relu")(X)
        X = tf.keras.layers.MaxPooling2D()(X)

        X = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation="relu")(X)
        X = tf.keras.layers.MaxPooling2D()(X)
    else:
        print("Not a legitimate model")
        sys.exit(2)

    X = tf.keras.layers.Flatten()(X)
    
    X = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer='l2')(X)
    X = tf.keras.layers.Dropout(0.4)(X)
    X = tf.keras.layers.Dense(1)(X)
    
    model = tf.keras.models.Model(inputs=X_input, outputs=X)
    
    return model

def get_data(num_gpus):

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        image = tf.image.resize(image, (224,224)) # make sure all images are the same size
        return image, label

    datasets, info = tfds.load(name="cats_vs_dogs", split=["train[:60%]","train[60%:80%]","train[80%:]"], with_info=True, as_supervised=True) # TODO: check if randomly splits
    train_data = datasets[0]
    dev_data = datasets[1]
    test_data = datasets[2]

    buffer_size = 10000 # The bigger this is, the more random the distribution
    batch_size_per_replica = 32
    batch_size = batch_size_per_replica * num_gpus

    # train_dataset = train_data.map(scale).cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) # AUTOTUNE will automatically find optimum value for my system (if I had enough memory, optimum would be to prefetch the number of batches I work on each time: 437 for train)
    # dev_dataset = dev_data.map(scale).cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_data.map(scale).cache().shuffle(buffer_size).batch(batch_size).prefetch(437)
    dev_dataset = dev_data.map(scale).cache().batch(batch_size).prefetch(146)
    test_dataset = test_data.map(scale).batch(batch_size)

    return train_dataset, dev_dataset, test_dataset

def train_model_virtual_gpus(num_gpus, model_arch):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=512) for i in range(num_gpus)]
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

            print("Virtual Device Configuration:")
            print(tf.config.get_logical_device_configuration(gpus[0]))
            

            strategy = tf.distribute.MirroredStrategy()
            print("Num replicas in sync (should be {}): {}".format(num_gpus, strategy.num_replicas_in_sync))

            train_dataset, dev_dataset, test_dataset = get_data(strategy.num_replicas_in_sync)

            logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_{}gpus".format(num_gpus)
            file_writer = tf.summary.create_file_writer(logdir + "/metrics")
            file_writer.set_as_default()
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

            with strategy.scope():
                model = create_model((224,224,3), model_arch=model_arch)
                model.compile(optimizer=tf.keras.optimizers.Adam(0.03), 
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # from_logits True because haven't applied softmax yet (this is apparently more numerically stable)
                            metrics=["accuracy"])

            print(model.summary())

            callbacks = [
                tensorboard_callback,
                EpochTime()
            ]

            model.fit(train_dataset, validation_data=dev_dataset, epochs=10, callbacks=callbacks)
            print("\n")

        except RuntimeError as e:
            print(e)

def train_model_physical_gpus(model_arch):
    strategy = tf.distribute.MirroredStrategy()
    print("Num replicas in sync: {}".format(strategy.num_replicas_in_sync))

    train_dataset, dev_dataset, test_dataset = get_data(strategy.num_replicas_in_sync)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_{}gpus_".format(strategy.num_replicas_in_sync) +"entire"
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    with strategy.scope():
        model = create_model((224,224,3), model_arch=model_arch)
        model.compile(optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                    metrics=["accuracy"])

    print(model.summary())

    callbacks = [
        tensorboard_callback,
        EpochTime()
    ]

    model.fit(train_dataset, 
            validation_data=dev_dataset, 
            epochs=25, 
            callbacks=callbacks)
    print("\n")

if __name__ == '__main__':
    split_one_gpu = False
    model_arch = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["splitonegpu", "modelarch="])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--splitonegpu":
            split_one_gpu = arg
        elif opt == "--modelarch":
            model_arch = int(arg)

    if split_one_gpu:
        num_of_gpus = [1, 2, 4, 6]
        for num_gpus in num_of_gpus:
            p = multiprocessing.Process(target=train_model_virtual_gpus, args=(num_gpus, model_arch))
            p.start()
            p.join()
    else:
        p = multiprocessing.Process(target=train_model_physical_gpus, args=(model_arch,))
        p.start()
        p.join()