import tensorflow as tf

def batch_norm(x,momentum = 0.9,epsilon = 1e-5,train = True,name="bn"):
        return tf.layers.batch_normalization(inputs = x,\
                        momentum= momentum, \
                        epsilon= epsilon,\
                        scale=True,\
                        training= train,\
                        name=name)

def simple_cnn(x):
        conv1 = tf.layers.conv2d(inputs =x,\
                filters = 32,\
                kernel_size = [3,3],\
                padding="same",\
                activation=tf.nn.relu,\
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),\
                kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001))
        conv1 = batch_norm(conv1,name="pw_btn1")
        pool1 = tf.layers.max_pooling2d(inputs = conv1,\
                pool_size=[2,2],strides = 2)
        
        conv2 = tf.layers.conv2d(inputs = pool1,\
                filters = 64,\
                kernel_size=[3,3],\
                padding='same',\
                activation=tf.nn.relu,\
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),\
                kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))
        conv2 = batch_norm(conv2,name='pe_btn2')
        pool2 = tf.layers.max_pooling2d(inputs = conv2,\
                pool_size=[2,2],strides=2)
        
        conv3 = tf.layers.conv2d(inputs = pool2,\
                filters = 128,\
                kernel_size=[3,3],\
                padding='same',\
                activation=tf.nn.relu,\
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),\
                kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))
        conv3 = batch_norm(conv3,name='pe_btn3')
        pool3 = tf.layers.max_pooling2d(inputs = conv3,\
                pool_size=[2,2],strides=2)

      
        
        shp = pool3.get_shape() 
        print("shape:",shp)
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        re1 = tf.reshape(pool3,[-1,flattened_shape])

        dense1 = tf.layers.dense(inputs=re1,\
                             units=256,\
                             activation=tf.nn.relu,\
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),\
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

        drop1 = tf.layers.dropout(inputs = dense1,name='drop_out1',)
        dense2 = tf.layers.dense(inputs=drop1,\
                             units=128,\
                             activation=tf.nn.relu,\
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),\
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        drop2 = tf.layers.dropout(inputs = dense2,name='drop_out1')     
        logits = tf.layers.dense(inputs=drop2,\
                             units=10,\
                             activation=None,\
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),\
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred