'''
# demonstration
  with tf.Session() as sess:
    #X = tf.transpose(I, [2,1,0]) # (r, w, b)
    #X = tf.batch_to_space_nd(X, [r], [[0,0]]) # (1, r*w, b)
    #X = tf.transpose(X, [2,1,0])

    x = np.arange(2*4*2).reshape(2, 4, 2)
    X = tf.placeholder("float32", shape=(2, 4, 2), name="X")
    Y = SubPixel1D(X, 2)
    y = sess.run(Y, feed_dict={X: x})

    print 'single-channel:'
    print 'original, element 0 (2 channels):', x[0,:,0], x[0,:,1]
           original, element 0 (2 channels): [0 2 4 6] [1 3 5 7]
    print 'rescaled, element 1:', y[0,:,0]
           rescaled, element 1: [0. 1. 2. 3. 4. 5. 6. 7.]
    print
    print 'original, element 0 (2 channels) :', x[1,:,0], x[1,:,1]
           original, element 0 (2 channels) : [ 8 10 12 14] [ 9 11 13 15]
    print 'rescaled, element 1:', y[1,:,0]
           rescaled, element 1: [ 8.  9. 10. 11. 12. 13. 14. 15.]
    print

    x = np.arange(2*4*4).reshape(2, 4, 4)
    X = tf.placeholder("float32", shape=(2, 4, 4), name="X")
    Y = SubPixel1D(X, 2)
    y = sess.run(Y, feed_dict={X: x})

    print 'multichannel:'
    print 'original, element 0 (4 channels):', x[0,:,0], x[0,:,1], x[0,:,2], x[0,:,3]
           original, element 0 (4 channels): [ 0  4  8 12] [ 1  5  9 13] [ 2  6 10 14] [ 3  7 11 15]
    print 'rescaled, element 1:', y[0,:,0], y[0,:,1]
           rescaled, element 1: [ 0.  2.  4.  6.  8. 10. 12. 14.] [ 1.  3.  5.  7.  9. 11. 13. 15.]
    print
    print 'original, element 0 (2 channels) :', x[1,:,0], x[1,:,1], x[1,:,2], x[1,:,3]
           original, element 0 (2 channels) : [16 20 24 28] [17 21 25 29] [18 22 26 30] [19 23 27 31]
    print 'rescaled, element 1:', y[1,:,0], y[1,:,1],
           rescaled, element 1: [16. 18. 20. 22. 24. 26. 28. 30.] [17. 19. 21. 23. 25. 27. 29. 31.]
    single-channel:
'''
import numpy as np
from keras.layers import Input, Conv1D, Permute, Reshape
from keras.models import Model

def range_test():
    x = np.arange(2*4*2).reshape(2, 4, 2)
    print 'in',x
    print 'x[0]',x[0]
    print 'x[0,:,:]',x[0,:,:]
    print 'x[0,:,0]',x[0,:,0]

def subpixel1D(x, r=2):
    shape = (-1, int(x.shape[-1]/r))
    y = Reshape(shape)(x)
    #y = Permute((2,1))(x)
    return y

def main():
    x = Input(shape=(None,2))
    y = subpixel1D(x)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    val = np.arange(2*4*2).reshape(2,4,2)
    print 'input', val.shape
    print val

    res = model.predict(val)
    print 'output', res.shape
    print res
    print res.shape #(2,8,1)
    print res[0,:,0] #
    print res[1,:,0] #

def main2():
    x = Input(shape=(None,4))
    y = subpixel1D(x)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    val = np.arange(2*4*4).reshape(2,4,4)
    print 'input'
    print val

    res = model.predict(val)
    print 'output'
    print res
    print res.shape #(2,8,2)
    print res[0,:,0] #
    print res[0,:,1] #
    print res[1,:,0] #
    print res[1,:,1] #


if __name__ == "__main__":
    main()

