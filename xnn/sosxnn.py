import tensorflow as tf
from .base import BaseNet


class SOSxNN(BaseNet):

    def __init__(self, input_num, input_dummy_num=0, subnet_num=10, subnet_arch=[10, 6], task="Regression",
                 activation_func=tf.tanh, batch_size=100, training_epochs=10000, lr_bp=0.001, lr_cl=0.1,
                 beta_threshold=0.05, tuning_epochs=500, l1_proj=0.001, l1_subnet=0.001, smooth_lambda=0.000001,
                 verbose=False, val_ratio=0, early_stop_thres=1000):

        super(SOSxNN, self).__init__(input_num=input_num,
                                     input_dummy_num=input_dummy_num,
                                     subnet_num=subnet_num,
                                     subnet_arch=subnet_arch,
                                     task=task,
                                     proj_method="orthogonal",
                                     activation_func=tf.tanh,
                                     bn_flag=True,
                                     lr_bp=lr_bp,
                                     l1_proj=l1_proj,
                                     l1_subnet=l1_subnet,
                                     smooth_lambda=smooth_lambda,
                                     batch_size=batch_size,
                                     training_epochs=training_epochs,
                                     tuning_epochs=tuning_epochs,
                                     beta_threshold=beta_threshold,
                                     verbose=verbose,
                                     val_ratio=val_ratio,
                                     early_stop_thres=early_stop_thres)
        self.lr_cl = lr_cl

    @tf.function
    def train_step_init(self, inputs, labels):
        with tf.GradientTape() as tape_cl:
            with tf.GradientTape() as tape_bp:
                pred = self.apply(inputs, training=True)
                pred_loss = self.loss_fn(labels, pred)
                regularization_loss = tf.math.add_n(self.proj_layer.losses + self.output_layer.losses)
                cl_loss = pred_loss + regularization_loss
                bp_loss = pred_loss + regularization_loss 
                if self.smooth_lambda > 0:
                    smoothness_loss = self.subnet_blocks.smooth_loss
                    bp_loss += smoothness_loss

        grad_proj = tape_cl.gradient(cl_loss, self.proj_layer.weights)
        grad_nets = tape_bp.gradient(bp_loss, list(set(self.trainable_weights).difference(set(self.proj_layer.weights))))

        in_shape = self.proj_layer.weights[0].shape[0]
        matrix_a = (tf.matmul(grad_proj[0], tf.transpose(self.proj_layer.weights[0]))
                    - tf.matmul(self.proj_layer.weights[0], tf.transpose(grad_proj[0])))
        matrix_q = tf.matmul(tf.linalg.inv(tf.eye(in_shape) + tf.multiply(self.lr_cl / 2, matrix_a)),
                             (tf.eye(in_shape) - tf.multiply(self.lr_cl / 2, matrix_a)))
        self.proj_layer.weights[0].assign(tf.matmul(matrix_q, self.proj_layer.weights[0]))
        self.optimizer.apply_gradients(zip(grad_nets, list(set(self.trainable_weights).difference(set(self.proj_layer.weights)))))

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
