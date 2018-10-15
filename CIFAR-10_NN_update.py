#coding=utf-8
# 前两行代码可以跳过，自行去
'''https: // github.com / tensorflow / models / tree / master / tutorials / image / cifar10
中将cifar10.py和cifar10_input.py下载下来'''
'''
git clone https://github.com/tensorflow/models.git
cd models/tutorials/image/cifar10
'''

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir='./CIFAR10/CIFAR10-data/cifar-10-batches-bin'
log_dir='./CIFAR10/log/cifar_with_summaries'

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # 对权重进行L2正则化，L1正则化会制造稀疏特征，大部分无用特征的权重会被置为0，而L2正则会让特征的权重不过大，使得特征的权重比较平均。
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# 把cifar10的数据解压到data_dir中
cifar10.maybe_download_and_extract()

#返回的是已经封装好的tensor，每次执行都会生成一个batch_size的数量的样本，该函数包含了数据增强的操作（如随机水平翻转等）
#  这行代码在Tensorflow 0.12版中会报错【TypeError: strided_slice() missing 1 required positional argument: 'strides'】我将版本升至1.0后就没有报错了
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

#对Variable变量数据的汇总函数，如统计均值，方差，以及直方图信息
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

with tf.name_scope('input'):
    #后面定义网络结构时，需要使用batch_size，因此不再使用None替代
    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])

with tf.name_scope('con1'):
    #w1设为0，意味着weight1不需要L2正则化
    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
    variable_summaries(weight1)
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
    variable_summaries(bias1)
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
with tf.name_scope('conv1_pooling'):
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')#最大池化的尺寸和步长不同，可以增加数据的丰富性
with tf.name_scope('conv1_lrn'):
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) #LRN层

with tf.name_scope('con2'):
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
    variable_summaries(weight2)
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))#bias初始化为0.1
    variable_summaries(bias2)
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
with tf.name_scope('conv2_lrn'):
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
with tf.name_scope('conv2_pooling'):
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('fcn1'):
    reshape = tf.reshape(pool2, [batch_size, -1])
    # 比如pool2总共有a×b×c个数，要reshape成batch_size行 × ???列，这里的-1就表示???列，tf.reshape会自行算出需要多少列a×b×c/batch_size
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)#希望这个全连接层不要过拟合，因此设置w1为0.004
    variable_summaries(weight3)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    variable_summaries(bias3)
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

with tf.name_scope('fcn2'):
    weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    variable_summaries(weight4)
    variable_summaries(bias4)
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

with tf.name_scope('fcn3_output'):
    weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)#标准差设置为上一个隐含层的节点数的倒数
    bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
    variable_summaries(weight5)
    variable_summaries(bias5)
    logits = tf.nn.relu(tf.matmul(local4, weight5) + bias5)#整个inference网络的输出结果，不需要softmax的映射，直接比较各类的数值大小就可以


def loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='cross_entropy_per_example')#这里softmax和cross_entropy结合使用
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        tf.add_to_collection('losses', cross_entropy_mean)#将cross_entropy加入到整体的loss中
        return tf.add_n(tf.get_collection('losses'), name='total_loss')#将全部loss求和，得到最终的loss，即entropy loss和weight的L2 loss

with tf.name_scope('total_loss'):
    loss = loss(logits, label_holder)
tf.summary.scalar('total_loss',loss)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)#获取top_k的准确率，默认是top 1

sess = tf.InteractiveSession()
merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter(log_dir+'/train',sess.graph)
test_writer=tf.summary.FileWriter(log_dir+'/test')
tf.global_variables_initializer().run()
tf.train.start_queue_runners()#使用16个线程加速


# for step in range(max_steps):
#     start_time = time.time()
#     image_batch, label_batch = sess.run([images_train, labels_train])#获取一个batch的训练数据
#     _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
#     duration = time.time() - start_time
#     if step % 10 == 0:
#         examples_per_sec = batch_size / duration
#         sec_per_batch = float(duration)
#
#         format_str = ('step %d,loss=%.2f (%.1f example/sec; %.3f sec/batch)')
#         print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
#
# num_examples = 10000
# import math
#
#
# num_iter = int(math.ceil(num_examples / batch_size))
# true_count = 0
# total_sample_count = num_iter * batch_size
# step = 0
# while step < num_iter:
#     image_batch, label_batch = sess.run([images_test, labels_test])
#     predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
#     true_count += np.sum(predictions)
#     step += 1
#
# precision = true_count / total_sample_count
# print('precision @ 1 = %.3f' % precision)

saver=tf.train.Saver()
for i in range(max_steps):
    image_batch, label_batch = sess.run([images_train, labels_train])  # 获取一个batch的训练数据
    if i % 100 ==0:
        # 这里通过trace_level参数配置运行时需要记录的信息， # tf.RunOptions.FULL_TRACE代表所有的信息
        run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # 运行时记录运行信息的proto，pb是用来序列化数据的
        run_metadata=tf.RunMetadata()
        # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
        # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # print("step %d, training accuracy %g" % (i, train_accuracy))
        summary,_=sess.run([merged,train_op],feed_dict={image_holder: image_batch, label_holder: label_batch},options=run_options,run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata,'step%03d'%i)
        train_writer.add_summary(summary,i)
        saver.save(sess,log_dir+'/model.ckpt',i)
        print("Adding run metadata for ",i)
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    summary,_=sess.run([merged,train_op],feed_dict={image_holder: image_batch, label_holder: label_batch})
    train_writer.add_summary(summary, i)

with tf.name_scope('accuracy'):
    num_examples = 10000
    import math
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
        true_count += np.sum(predictions)
        step += 1
        accuracy = true_count / total_sample_count
    print('precision @ 1 = %.3f' % accuracy)
tf.summary.scalar('accuracy',accuracy)

train_writer.close()
test_writer.close()