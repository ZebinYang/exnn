Examples
===============
Here we give more example usage of this package.


GAMNet
---------------------------------------------------

.. code-block::

        # Simulation
        import numpy as np
        from xnn import GAMNet
        
        corr = 0.5
        noise_sigma = 1
        dummy_num = 0
        feature_num = 10
        test_num = 10000
        data_num = 10000

        proj_matrix = np.zeros((feature_num,4))
        proj_matrix[:7, 0] = np.array([1,0,0,0,0,0,0])
        proj_matrix[:7, 1] = np.array([0,1,0,0,0,0,0])
        proj_matrix[:7, 2] = np.array([0,0,0.5,0.5,0,0,0])
        proj_matrix[:7, 3] = np.array([0,0,0,0,0.2,0.3,0.5])

        def data_generator1(datanum, testnum, featurenum, corr, proj_matrix, noise_sigma, random_seed):
            np.random.seed(random_seed)
            u = np.random.uniform(-1, 1, [datanum + testnum, 1])
            t = np.sqrt(corr / (1 - corr))
            x = np.zeros((datanum + testnum, featurenum))
            for i in range(featurenum):
                x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum + testnum, 1]) + t * u) / (1 + t)
            y = np.reshape(2 * np.dot(x, proj_matrix[:, 0]) + 0.2 * np.exp(-4 * np.dot(x, proj_matrix[:, 1])) + \
                           3 * (np.dot(x, proj_matrix[:, 2]))**2 + 2.5 * np.sin(np.pi * np.dot(x, proj_matrix[:, 3])), [-1, 1]) + \
                      noise_sigma * np.random.normal(0, 1, [datanum + testnum, 1])
            return x, y

        X, Y = data_generator1(data_num+test_num, feature_num+dummy_num, corr, proj_matrix, noise_sigma, random_seed=0)
        scaler_x = MinMaxScaler((-1, 1)); scaler_y = MinMaxScaler((-1, 1))
        sX = scaler_x.fit_transform(X); sY = scaler_y.fit_transform(Y)
        train_x, test_x, train_y, test_y = train_test_split(sX, sY, test_size = test_num)
        
        np.random.seed(0)
        tf.random.set_seed(0)
        model = GAMNet(input_num = 10, input_dummy_num=0, subnet_arch=[10, 6], task="Regression",
                       activation_func=tf.tanh, batch_size=1000, training_epochs=5000, lr_bp=0.001,
                       beta_threshold=0.01, tuning_epochs=200, l1_subnet=0.01, smooth_lambda = 10**(-5),
                       verbose=True, val_ratio=0.2, early_stop_thres=200)
        model.fit(train_x, train_y)
        model.visualize("./", "test")


xNN
---------------------------------------------------

.. code-block::

        # Simulation
        import numpy as np
        from xnn import xNN
        
        corr = 0.5
        noise_sigma = 1
        dummy_num = 0
        feature_num = 10
        test_num = 10000
        data_num = 10000

        proj_matrix = np.zeros((feature_num,4))
        proj_matrix[:7, 0] = np.array([1,0,0,0,0,0,0])
        proj_matrix[:7, 1] = np.array([0,1,0,0,0,0,0])
        proj_matrix[:7, 2] = np.array([0,0,0.5,0.5,0,0,0])
        proj_matrix[:7, 3] = np.array([0,0,0,0,0.2,0.3,0.5])

        def data_generator1(datanum, testnum, featurenum, corr, proj_matrix, noise_sigma, random_seed):
            np.random.seed(random_seed)
            u = np.random.uniform(-1, 1, [datanum + testnum, 1])
            t = np.sqrt(corr / (1 - corr))
            x = np.zeros((datanum + testnum, featurenum))
            for i in range(featurenum):
                x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum + testnum, 1]) + t * u) / (1 + t)
            y = np.reshape(2 * np.dot(x, proj_matrix[:, 0]) + 0.2 * np.exp(-4 * np.dot(x, proj_matrix[:, 1])) + \
                           3 * (np.dot(x, proj_matrix[:, 2]))**2 + 2.5 * np.sin(np.pi * np.dot(x, proj_matrix[:, 3])), [-1, 1]) + \
                      noise_sigma * np.random.normal(0, 1, [datanum + testnum, 1])
            return x, y

        X, Y = data_generator1(data_num+test_num, feature_num+dummy_num, corr, proj_matrix, noise_sigma, random_seed=0)
        scaler_x = MinMaxScaler((-1, 1)); scaler_y = MinMaxScaler((-1, 1))
        sX = scaler_x.fit_transform(X); sY = scaler_y.fit_transform(Y)
        train_x, test_x, train_y, test_y = train_test_split(sX, sY, test_size = test_num)
        
        np.random.seed(0)
        tf.random.set_seed(0)
        model = xNN(input_num = 10, input_dummy_num=0, subnet_num=10, subnet_arch=[10, 6], task="Regression",
                       activation_func=tf.tanh, batch_size=1000, training_epochs=5000, lr_bp=0.001, 
                       beta_threshold=0.01, tuning_epochs=200, l1_proj=0.001, l1_subnet=0.001, 
                       verbose=True, val_ratio=0.2, early_stop_thres=500)
        model.fit(train_x, train_y)
        model.visualize("./", "test")


SOSxNN
---------------------------------------------------

.. code-block::

        # Simulation
        import numpy as np
        from xnn import SOSxNN
        
        corr = 0.5
        noise_sigma = 1
        dummy_num = 0
        feature_num = 10
        test_num = 10000
        data_num = 10000

        proj_matrix = np.zeros((feature_num,4))
        proj_matrix[:7, 0] = np.array([1,0,0,0,0,0,0])
        proj_matrix[:7, 1] = np.array([0,1,0,0,0,0,0])
        proj_matrix[:7, 2] = np.array([0,0,0.5,0.5,0,0,0])
        proj_matrix[:7, 3] = np.array([0,0,0,0,0.2,0.3,0.5])

        def data_generator1(datanum, testnum, featurenum, corr, proj_matrix, noise_sigma, random_seed):
            np.random.seed(random_seed)
            u = np.random.uniform(-1, 1, [datanum + testnum, 1])
            t = np.sqrt(corr / (1 - corr))
            x = np.zeros((datanum + testnum, featurenum))
            for i in range(featurenum):
                x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum + testnum, 1]) + t * u) / (1 + t)
            y = np.reshape(2 * np.dot(x, proj_matrix[:, 0]) + 0.2 * np.exp(-4 * np.dot(x, proj_matrix[:, 1])) + \
                           3 * (np.dot(x, proj_matrix[:, 2]))**2 + 2.5 * np.sin(np.pi * np.dot(x, proj_matrix[:, 3])), [-1, 1]) + \
                      noise_sigma * np.random.normal(0, 1, [datanum + testnum, 1])
            return x, y

        X, Y = data_generator1(data_num+test_num, feature_num+dummy_num, corr, proj_matrix, noise_sigma, random_seed=0)
        scaler_x = MinMaxScaler((-1, 1)); scaler_y = MinMaxScaler((-1, 1))
        sX = scaler_x.fit_transform(X); sY = scaler_y.fit_transform(Y)
        train_x, test_x, train_y, test_y = train_test_split(sX, sY, test_size = test_num)
        
        np.random.seed(0)
        tf.random.set_seed(0)
        model = SOSxNN(input_num=10, input_dummy_num=0, subnet_num=10, subnet_arch=[10, 6], task="Regression",
                       activation_func=tf.tanh, batch_size=1000, training_epochs=5000, lr_bp=0.001, lr_cl=0.1,
                       beta_threshold=0.01, tuning_epochs=0, l1_proj=0.001, l1_subnet = 0.01, smooth_lambda=10**(-5),
                       verbose=True, val_ratio=0.2, early_stop_thres=500)
        model.fit(train_x, train_y)
        model.visualize("./", "test", train_x)
