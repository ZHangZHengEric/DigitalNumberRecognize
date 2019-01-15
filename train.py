# import data_preprocess
# images,labels = data_preprocess.loadCSVfile2()
# images,labels = data_preprocess.rotate_image(images,labels)
# data_preprocess.save_csv_as_image_files("image_data",images,labels)
import tensorflow as tf
import sys
import data_preprocess
import model
import time
import os
import sklearn.model_selection
w = 100
h = 100
c = 3
path = 'image_data'
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model_of_digital"
board_save_path = "grah/"

x = tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_ = tf.placeholder(tf.int32,shape=[None, ],name='y_')
print(y_.shape)
logits,pred = model.simple_cnn(x)
print(logits.shape)

global_step = tf.Variable(0, trainable=False)   
loss = tf.losses.sparse_softmax_cross_entropy(labels= y_,logits = logits)
tf.summary.scalar('loss',loss)

# learning_rate = tf.train.exponential_decay(0.01, global_step,2,0.99,staircase=True)
learning_rate = 0.000002
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step = global_step)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32),y_)
acc = tf.reduce_mean(tf.cast( correct_prediction,tf.float32 ))

average_loss = tf.reduce_mean(tf.cast( loss,tf.float32 ))
data_paths,labels = data_preprocess.find_all_data(path)
batch_size = 128
test_batch_size = 128
images_train,images_test ,labels_train,labels_test = sklearn.model_selection.train_test_split(data_paths,labels,test_size=0.2)
image_batch, label_batch = data_preprocess.get_batch_data(data_paths,labels,batch_size=batch_size,num_epochs=10,w=w,h=h,c=c)
test_image_batch, test_label_batch = data_preprocess.get_batch_data(images_test,labels_test,batch_size=test_batch_size,w=w,h=h,c=c,use_shuffle=False)      

n_epoch = 12
def train():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    saver = tf.train.Saver()                                                                                                                                                                            
    merged = tf.summary.merge_all()  
    writer = tf.summary.FileWriter(board_save_path+"/log",sess.graph) 
    threads = tf.train.start_queue_runners(sess, coord)
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) # 有checkpoint的话继续上一次的训练 

    if ckpt and ckpt.model_checkpoint_path: 
        saver.restore(sess,ckpt.model_checkpoint_path)

    try:
        for i in range (2001):
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            # print(image_batch_v.shape, label_batch_v,y_)
            _, result,loss_value, step = sess.run([train_op,merged, loss, global_step],
                                           feed_dict={x:image_batch_v,y_:label_batch_v})
            print("After %d training step(s),loss on training batch is %g. " % (step, loss_value))
            writer.add_summary(result,step)
            if i % 200  == 0:  
                saver.save( sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                
                accvalue = sess.run(acc,feed_dict={x:image_batch_v,y_:label_batch_v})
                print ("Iter "+ str(i) + ", Train Accuracy= " + str(accvalue))

                test_image_batch_v, test_label_batch_v = sess.run([test_image_batch, test_label_batch])
                test_loss_value = sess.run(average_loss,feed_dict={x:test_image_batch_v,y_:test_label_batch_v})
                
                print("After %d training step(s),loss on test batch is %g." % (step, test_loss_value))
                test_accvalue = sess.run(acc,feed_dict={x:test_image_batch_v,y_:test_label_batch_v})
                print ("Iter "+ str(i) + ", test Accuracy= " + str(test_accvalue))
    except tf.errors.OutOfRangeError:  
        print("done! now lets kill all the threads……")
    finally:
        coord.request_stop()
        print('all threads are asked to stop!')
        writer = tf.summary.FileWriter(board_save_path,sess.graph)
    coord.join(threads)  
    print('all threads are stopped!')  


train()