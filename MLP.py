#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


max_steps=3000
dropout=0.75
data_dir='./mnist_MLP/input_data'
log_dir='./mnist_MLP/logs/mnist_with_summaries'
mnist=input_data.read_data_sets(data_dir,one_hot=True)
sess=tf.InteractiveSession()

in_units=784
h1_units=300

with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,in_units])
    y_=tf.placeholder(tf.float32,[None,10])

with tf.name_scope('input_reshape'):
    image_shape_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shape_input,10)

with tf.name_scope('layer1'):
    w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    preactive = tf.matmul(x, w1) + b1
    tf.summary.histogram('pre_activations_layer1', preactive)
    hidden1 = tf.nn.relu(preactive)
    tf.summary.histogram('activations', hidden1)

with tf.name_scope('dropout'):
    keep_prob=tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep-probability',keep_prob)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

with tf.name_scope('layer2'):
    w2 = tf.Variable(tf.zeros([h1_units, 10]))
    b2 = tf.Variable(tf.zeros([10]))
    preactive = tf.matmul(hidden1_drop,w2)+b2
    tf.summary.histogram('pre_activations_layer2', preactive)
    y=tf.nn.softmax(preactive)
    tf.summary.histogram('activations', y)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)  # 优化函数可以更换

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter(log_dir+'/train',sess.graph)
test_writer=tf.summary.FileWriter(log_dir+'/test')
tf.global_variables_initializer().run()

def feed_dict(train):
    if train:
        xs,ys=mnist.train.next_batch(100)
        k=dropout
    else:
        xs,ys=mnist.test.images,mnist.test.labels
        k=1.0
    return {x:xs,y_:ys,keep_prob:k}

saver=tf.train.Saver()
for i in range(max_steps):
    if i % 10 ==0:
        summary,acc=sess.run([merged,accuracy],feed_dict=feed_dict(False))
        test_writer.add_summary(summary,i)
        print('accuracy at step %s:%s'%(i,acc))
    else:
        if i%100==99:
            # 这里通过trace_level参数配置运行时需要记录的信息， # tf.RunOptions.FULL_TRACE代表所有的信息
            run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # 运行时记录运行信息的proto，pb是用来序列化数据的
            run_metadata=tf.RunMetadata()
            # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
            summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True),options=run_options,run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata,'step%03d'%i)
            train_writer.add_summary(summary,i)
            saver.save(sess,log_dir+'/model.ckpt',i)
            print("Adding run metadata for ",i)
        else:
            summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True))
            train_writer.add_summary(summary,i)
acc=accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
print("test accuracy %g"%acc)
train_writer.close()
test_writer.close()

