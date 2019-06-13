import tensorflow as tf
from .base import BaseNet


class GAMNet(BaseNet):

    def __init__(self, input_num, input_dummy_num=0, subnet_num=10, subnet_arch=[10, 6], task="Regression",
                 activation_func=tf.tanh, batch_size=100, training_epochs=10000, lr_bp=0.001,
                 beta_threshold=0.05, tuning_epochs=500, l1_subnet=0.001, smooth_lambda=0.000001,
                 verbose=False, val_ratio=0, early_stop_thres=1000):

        super(GAMNet, self).__init__(input_num=input_num,
                                     input_dummy_num=input_dummy_num,
                                     subnet_num=input_num,
                                     subnet_arch=subnet_arch,
                                     task=task,
                                     proj_method="gam",
                                     activation_func=tf.tanh,
                                     bn_flag=True,
                                     lr_bp=lr_bp,
                                     l1_proj=0,
                                     l1_subnet=l1_subnet,
                                     smooth_lambda=smooth_lambda,
                                     batch_size=batch_size,
                                     training_epochs=training_epochs,
                                     tuning_epochs=tuning_epochs,
                                     beta_threshold=beta_threshold,
                                     verbose=verbose,
                                     val_ratio=val_ratio,
                                     early_stop_thres=early_stop_thres)

    @tf.function
    def train_step_init(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            regularization_loss = tf.math.add_n(self.output_layer.losses)
            total_loss = pred_loss + regularization_loss
            if self.smooth_lambda > 0:
                smoothness_loss = self.subnet_blocks.smooth_loss
                total_loss += smoothness_loss

        variables = list(set(self.trainable_weights).difference(set(self.proj_layer.weights)))
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

    @tf.function
    def train_step_finetune(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            total_loss = pred_loss
            if self.smooth_lambda > 0:
                smoothness_loss = self.subnet_blocks.smooth_loss
                total_loss += smoothness_loss

        variables = list(set(self.trainable_weights).difference(set(self.proj_layer.weights)))
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
