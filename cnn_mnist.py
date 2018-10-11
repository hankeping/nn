#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

max_steps=20000
dropout=0.5
data_dir='./mnist_NN/input_data'
log_dir='./mnist_NN/logs/mnist_with_summaries'
mnist=input_data.read_data_sets(data_dir,one_hot=True)
sess=tf.InteractiveSession()

#初始化函数，以便重复使用
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)#使用随机噪声打破完全对称
    return tf.Variable(initial)

def bais_variable(shape):
    initial=tf.constant(0.1,shape=shape)#由于使用Relu，避免死亡节点，因此设置初始值为0.1
    return tf.Variable(initial)

#W是卷积的参数
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y_= tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    x_image=tf.reshape(x,[-1,28,28,1])#-1代表样本数不固定，1代表颜色通道数量
    tf.summary.image('input',x_image,10)

with tf.name_scope('layer1'):
    w_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bais_variable([32])
    preactive = conv2d(x_image,w_conv1)+b_conv1
    tf.summary.histogram('pre_activations_layer1', preactive)
    h_conv1 = tf.nn.relu(preactive)
    tf.summary.histogram('activations', h_conv1)

with tf.name_scope('max_pooling'):
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('layer2'):
    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bais_variable([64])
    preactive = conv2d(h_pool1,w_conv2)+b_conv2
    tf.summary.histogram('pre_activations_layer1', preactive)
    h_conv2= tf.nn.relu(preactive)
    tf.summary.histogram('activations', h_conv2)

with tf.name_scope('max_pooling'):
    h_pool2= max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    w_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bais_variable([1024])
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

with tf.name_scope('dropout'):
    keep_prob=tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep-probability',keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc_softmax'):
    w_fc2 = weight_variable([1024,10])
    b_fc2 = bais_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
tf.summary.scalar('cross_entropy',cross_entropy)


with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 优化函数可以更换

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter(log_dir+'/train',sess.graph)
test_writer=tf.summary.FileWriter(log_dir+'/test')
tf.global_variables_initializer().run()


saver=tf.train.Saver()
for i in range(max_steps):
    batch = mnist.train.next_batch(50)
    if i % 100 ==0:
        # 这里通过trace_level参数配置运行时需要记录的信息， # tf.RunOptions.FULL_TRACE代表所有的信息
        run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # 运行时记录运行信息的proto，pb是用来序列化数据的
        run_metadata=tf.RunMetadata()
        # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
        # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # print("step %d, training accuracy %g" % (i, train_accuracy))
        summary,_=sess.run([merged,train_step],feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0},options=run_options,run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata,'step%03d'%i)
        train_writer.add_summary(summary,i)
        saver.save(sess,log_dir+'/model.ckpt',i)
        print("Adding run metadata for ",i)
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    summary,_=sess.run([merged,train_step],feed_dict={x:batch[0],y_:batch[1],keep_prob:dropout})
    train_writer.add_summary(summary, i)

acc=accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
print("test accuracy %g"%acc)
train_writer.close()
test_writer.close()