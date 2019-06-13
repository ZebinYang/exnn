# Explainable-Neural-Networks

## Installation 
```shell
pip install git+https://github.com/ajzhanghku/XNN.git
# with token
pip install git+https://5a43258969ee1a0d8a385b867fcaef19a9aebfee@github.com/ajzhanghku/XNN.git
```


## Usage
- xNN

```python
from xnn.xNN import xNN
from xnn.SOSxNN import SOSxNN
from xnn.visualizer import visualize_xnn, visualize_sosxnn

model = xNN(input_num = train_x.shape[1], input_dummy_num = 0, subnet_num = 10, subnet_layers = [10, 6], \
        task = "Regression", activation = tf.tanh, batch_size = min(1000, int(np.floor(train_x.shape[0]*0.20))), \
        training_epochs = 10000, beta_threshold = 0.05, tune_epochs = 500, lr_BP = 0.001, \
        l1_proj = 0.001, l1_subnet = 0.01, verbose = False, \
        val_ratio = 0.2, early_stop_thres = 2500)
pred_train, tr_x, tr_y, pred_val, val_x, val_y = model.fit(train_x, train_y); 
pred_test = model.predict(test_x)
visualize_xnn(simu_dir, "Demo_XNN", model, tr_x, dummy_name = None)
```


- SOSxNN
```python
model = SOSxNN(input_num = train_x.shape[1], input_dummy_num = 0, subnet_num = 10, subnet_layers = [10, 6], \
        task = "Regression", activation = tf.tanh, batch_size = min(1000, int(np.floor(train_x.shape[0]*0.20))), \
        training_epochs = 10000, beta_threshold = 0.05, tune_epochs = 500, lr_BP = 0.001, lr_CL= 0.1, \
        l1_proj = 0.001, l1_subnet = 0.01, smooth_lambda = 10**(-6), verbose = False, \
        val_ratio = 0.2, early_stop_thres = 2500)
pred_train, tr_x, tr_y, pred_val, val_x, val_y = model.fit(train_x, train_y); 
pred_test = model.predict(test_x)
visualize_sosxnn(simu_dir, "Demo_SOSxNN", model, tr_x, dummy_name = None)
```

References
----------
J. Vaughan, A. Sudjianto, E. Brahimi, J. Chen, and V. N. Nair, "Explainable
neural networks based on additive index models," The RMA
Journal, pp. 40-49, October 2018.

Yang, Zebin, Aijun Zhang, and Agus Sudjianto. "Enhancing Explainability of
Neural Networks through Architecture Constraints." 
arXiv preprint arXiv:1901.03838 (2019).
