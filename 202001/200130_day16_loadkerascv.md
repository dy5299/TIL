```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from keras import backend as K
import keras
```


```python
# matrix
X = np.array([[0,0], [0,1], [1,0], [1,1]], 'float32')
Y = np.array([[0], [1], [1], [0]], 'float32')

#model = keras.models.Sequential() #keras package에서 가져온 것임
model = tf.keras.models.Sequential()
#4개 층이 아님.
#activation function도 같이 정의해서. 총 모델 개수는 10개
model.add(tf.keras.layers.Dense(64, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation ='sigmoid',
                               name='output')) #name 설정


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

model.fit(X, Y, batch_size=1, epochs=100, verbose=0)


model.summary()

# inputs:  ['dense_input']
print('inputs: ', [input.op.name for input in model.inputs])

# outputs:  ['dense_4/Sigmoid']
print('outputs: ', [output.op.name for output in model.outputs])




model.save('xor.h5')
```

    WARNING:tensorflow:From C:\Users\student\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From C:\Users\student\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From C:\Users\student\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From C:\Users\student\Anaconda3\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From C:\Users\student\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    WARNING:tensorflow:From C:\Users\student\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 64)                192       
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_3 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_4 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    output (Dense)               (None, 1)                 65        
    =================================================================
    Total params: 12,737
    Trainable params: 12,737
    Non-trainable params: 0
    _________________________________________________________________
    inputs:  ['dense_1_input']
    outputs:  ['output_2/Sigmoid']
    


```python
model.predict(np.array([[1, 1], [0,1]]))
```




    array([[0.00554549],
           [0.9916363 ]], dtype=float32)




```python
#tensorflow version 1.0으로 강제로 내린 것임
from tensorflow.compat.v1.keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#2.0에서 동작이 잘 안 돼서 1.0에서...
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


#output node만 알면 연결된 nodes 찾을 수 있다.
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, './', 'xor.pbtxt', as_text=True) #text file
tf.train.write_graph(frozen_graph, './', 'xor.pb', as_text=False)   #binary file
```

    INFO:tensorflow:Froze 140 variables.
    INFO:tensorflow:Converted 140 variables to const ops.
    




    './xor.pb'




```python
import cv2 as cv2
net = cv2.dnn.readNetFromTensorflow('xor.pb')
layersNames = net.getLayerNames()
print(layersNames)
```

    ['dense_14/MatMul', 'dense_14/Relu', 'dense_15/MatMul', 'dense_15/Relu', 'dense_16/MatMul', 'dense_16/Relu', 'dense_17/MatMul', 'dense_17/Relu', 'output_1/MatMul', 'output_1/Sigmoid']
    


```python
net.setInput(np.array([[0, 0], [0,1], [1,0], [1,1]]))
out = net.forward(outputName='output_1/Sigmoid')
print(out)

out = net.forward()
print(out)
```

    [[0.01285968]
     [0.98897713]
     [0.98643863]
     [0.01020394]]
    [[0.01285968]
     [0.98897713]
     [0.98643863]
     [0.01020394]]
    


```python
out = net.forward(["dense_17/Relu","output_1/Sigmoid"])
print(out[0])
print(out[1])
```

    [[-0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
       9.43579853e-01 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00  6.26061499e-01  9.86249328e-01  1.07812691e+00
      -0.00000000e+00  1.08873641e+00 -0.00000000e+00  1.13770092e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  6.41309798e-01
      -0.00000000e+00  6.94978476e-01 -0.00000000e+00 -0.00000000e+00
       9.62778866e-01 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  1.05217874e+00
      -0.00000000e+00 -0.00000000e+00  1.20442903e+00  1.15223396e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
       7.78923094e-01 -0.00000000e+00  7.47449458e-01  8.34294200e-01
      -0.00000000e+00  1.22530818e+00  4.54982668e-01 -0.00000000e+00
       6.80924475e-01 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
       8.76442373e-01  1.11418808e+00 -0.00000000e+00 -0.00000000e+00]
     [-0.00000000e+00  9.68509197e-01  1.20047247e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  1.17532301e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  1.17683637e+00
      -0.00000000e+00  1.52316141e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00  1.74228102e-04 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00  6.84111476e-01 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  5.66924214e-01
       6.55445904e-02  1.53925383e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00  1.29983377e+00  8.59925210e-01  3.03025991e-02
      -0.00000000e+00  1.58803165e+00  2.47228429e-01  2.51388222e-01
       1.41620469e+00  1.05263721e-02  1.59601402e+00  1.74218103e-01
       9.49972868e-02 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00  7.02478811e-02  8.07660222e-02 -0.00000000e+00
       3.19823995e-03 -0.00000000e+00 -0.00000000e+00  1.85608912e+00
       8.32090676e-02 -0.00000000e+00  1.40081036e+00  9.60174978e-01]
     [-0.00000000e+00  6.92318916e-01  7.72142708e-01 -0.00000000e+00
       4.50837702e-01 -0.00000000e+00 -0.00000000e+00  1.28194082e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  6.50572479e-01
      -0.00000000e+00  1.26402903e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00  1.20908022e-04  7.48139247e-03
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00  1.42381430e+00  4.75444980e-02
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  9.92450416e-01
      -0.00000000e+00  1.34967184e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00  1.39463103e+00  7.99187183e-01 -0.00000000e+00
      -0.00000000e+00  1.28766513e+00 -0.00000000e+00 -0.00000000e+00
       1.19045782e+00 -0.00000000e+00  1.48148513e+00  1.30133048e-01
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  1.02177262e-04
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  1.68631208e+00
       2.89972872e-01 -0.00000000e+00  1.47673786e+00  7.20323801e-01]
     [-0.00000000e+00  3.13338265e-03  1.04170986e-01 -0.00000000e+00
       1.27545428e+00 -0.00000000e+00 -0.00000000e+00  2.25801766e-03
      -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  1.66348845e-01
      -0.00000000e+00  4.52550322e-01 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00  1.12450826e+00  1.19293797e+00  1.41848969e+00
      -0.00000000e+00  7.36234188e-01 -0.00000000e+00  1.53740299e+00
      -0.00000000e+00 -0.00000000e+00  1.06814161e-01  8.36985230e-01
      -0.00000000e+00  1.10618126e+00 -0.00000000e+00 -0.00000000e+00
       1.42105317e+00  2.22893115e-02 -0.00000000e+00 -0.00000000e+00
      -0.00000000e+00  2.39733815e-01  1.18750706e-01  1.40128136e+00
      -0.00000000e+00  1.11897796e-01  1.38936973e+00  9.00256097e-01
       3.29000056e-02 -0.00000000e+00  8.94650817e-04  1.73362449e-01
       9.56217706e-01 -0.00000000e+00  1.11092830e+00  1.00334215e+00
      -0.00000000e+00  1.17654693e+00  6.24063790e-01 -0.00000000e+00
       9.82596874e-02 -0.00000000e+00 -0.00000000e+00  5.85186407e-02
       1.04717541e+00  1.23080552e+00  8.98537412e-02  5.10090828e-01]]
    [[0.01285968]
     [0.98897713]
     [0.98643863]
     [0.01020394]]
    

# mnist 로딩
- 내가 keras로 pb로 변환해서 동작하는지 확인 필요


```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```

    Using TensorFlow backend.
    

    x_train shape: (60000, 28, 28, 1)
    60000 train samples
    10000 test samples
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/1
    60000/60000 [==============================] - 68s 1ms/step - loss: 0.2631 - accuracy: 0.9198 - val_loss: 0.0574 - val_accuracy: 0.9829
    




    <keras.callbacks.callbacks.History at 0x21679198dd8>




```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


```

    Test loss: 0.05740329907569103
    Test accuracy: 0.9829000234603882
    


```python
model.save('mnist.h5')
```


```python
# 일단 반드시 이와 같이 해야함
import tensorflow as tf 
from tensorflow import keras
model3 = tf.keras.models.load_model("mnist.h5", compile=False)
model3.save("out", save_format='tf') # 폴더명임
```

    INFO:tensorflow:Assets written to: out\assets
    


```python
#model.save("mnist2.pb", save_format='tf')


from tensorflow import keras
model3 = keras.models.load_model("mnist.h5", compile=False)

export_path = 'saved_model.pb가 저장될 디렉토리'
model3.save("", save_format='tf')

load_model 을 할 때 두 번째 인자에 compile 옵션을 안 넣으면, 내 모델이 컴파일이 안 되어 있다고 오류가 뜬다. 구글링해보니 컴파일이란 거는 학습할 준비 어쩌고 하던데 하여튼 학습을 하지 않고 추론하는 데에만 모델을 사용할 거라면, compile=False 옵션을 넣어주면 된다고 해서 넣었음.

​

그리고 export_path는 파일이 저장될 디렉토리를 지정해주어야 하는데, 디렉토리 안이 비어있어야 한다.

그리하여 이것을 실행하면 !!
[출처] Keras 모델을 TensorFlow Lite로 변환하기 (h5 -> pb -> tflite)|작성자 aainy

```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-141-3eb9fccfc8d5> in <module>
    ----> 1 model.save("mnist2.pb", save_format='tf')
    

    TypeError: save() got an unexpected keyword argument 'save_format'



```python
from tensorflow.compat.v1.keras import backend as K

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, './', 'xor.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, './', 'mnist.pb', as_text=False)
```

    INFO:tensorflow:Froze 397 variables.
    INFO:tensorflow:Converted 397 variables to const ops.
    




    './mnist.pb'




```python
import tensorflow.compat.v1 as tf
from tensorflow.core import framework

def find_all_nodes(graph_def, **kwargs):
    for node in graph_def.node:
        for key, value in kwargs.items():
            if getattr(node, key) != value:
                break
        else:
            yield node
    raise StopIteration


def find_node(graph_def, **kwargs):
    try:
        return next(find_all_nodes(graph_def, **kwargs))
    except StopIteration:
        raise ValueError(
            'no node with attributes: {}'.format(
                ', '.join("'{}': {}".format(k, v) for k, v in kwargs.items())))


def walk_node_ancestors(graph_def, node_def, exclude=set()):
    openlist = list(node_def.input)
    closelist = set()
    while openlist:
        name = openlist.pop()
        if name not in exclude:
            node = find_node(graph_def, name=name)
            openlist += list(node.input)
            closelist.add(name)
    return closelist


def remove_nodes_by_name(graph_def, node_names):
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].name in node_names:
            del graph_def.node[i]


def make_shape_node_const(node_def, tensor_values):
    node_def.op = 'Const'
    node_def.ClearField('input')
    node_def.attr.clear()
    node_def.attr['dtype'].type = framework.types_pb2.DT_INT32
    tensor = node_def.attr['value'].tensor
    tensor.dtype = framework.types_pb2.DT_INT32
    tensor.tensor_shape.dim.add()
    tensor.tensor_shape.dim[0].size = len(tensor_values)
    for value in tensor_values:
        tensor.tensor_content += value.to_bytes(4, 'little')
    output_shape = node_def.attr['_output_shapes']
    output_shape.list.shape.add()
    output_shape.list.shape[0].dim.add()
    output_shape.list.shape[0].dim[0].size = len(tensor_values)


def make_cv2_compatible(graph_def):
    # A reshape node needs a shape node as its second input to know how it
    # should reshape its input tensor.
    # When exporting a model using Keras, this shape node is computed
    # dynamically using `Shape`, `StridedSlice` and `Pack` operators.
    # Unfortunately those operators are not supported yet by the OpenCV API.
    # The goal here is to remove all those unsupported nodes and hard-code the
    # shape layer as a const tensor instead.
    for reshape_node in find_all_nodes(graph_def, op='Reshape'):

        # Get a reference to the shape node
        shape_node = find_node(graph_def, name=reshape_node.input[1])

        # Find and remove all unsupported nodes
        garbage_nodes = walk_node_ancestors(graph_def, shape_node,
                                            exclude=[reshape_node.input[0]])
        remove_nodes_by_name(graph_def, garbage_nodes)

        # Infer the shape tensor from the reshape output tensor shape
        if not '_output_shapes' in reshape_node.attr:
            raise AttributeError(
                'cannot infer the shape node value from the reshape node. '
                'Please set the `add_shapes` argument to `True` when calling '
                'the `Session.graph.as_graph_def` method.')
        output_shape = reshape_node.attr['_output_shapes'].list.shape[0]
        output_shape = [dim.size for dim in output_shape.dim]

        # Hard-code the inferred shape in the shape node
        make_shape_node_const(shape_node, output_shape[1:])
```


```python
sess = K.get_session()
graph_def = sess.graph.as_graph_def(add_shapes=True)
graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [model.output.name.split(':')[0]])
make_cv2_compatible(graph_def)

# Print the graph nodes
print('\n'.join(node.name for node in graph_def.node))

# Save the graph as a binary protobuf2 file
tf.train.write_graph(graph_def, '', 'mnist.pb', as_text=False)
```

    INFO:tensorflow:Froze 8 variables.
    INFO:tensorflow:Converted 8 variables to const ops.
    


    ---------------------------------------------------------------------------

    OverflowError                             Traceback (most recent call last)

    <ipython-input-128-fd3fc9516155> in <module>
          2 graph_def = sess.graph.as_graph_def(add_shapes=True)
          3 graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [model.output.name.split(':')[0]])
    ----> 4 make_cv2_compatible(graph_def)
          5 
          6 # Print the graph nodes
    

    <ipython-input-127-03460a74436e> in make_cv2_compatible(graph_def)
         84 
         85         # Hard-code the inferred shape in the shape node
    ---> 86         make_shape_node_const(shape_node, output_shape[1:])
    

    <ipython-input-127-03460a74436e> in make_shape_node_const(node_def, tensor_values)
         49     tensor.tensor_shape.dim[0].size = len(tensor_values)
         50     for value in tensor_values:
    ---> 51         tensor.tensor_content += value.to_bytes(4, 'little')
         52     output_shape = node_def.attr['_output_shapes']
         53     output_shape.list.shape.add()
    

    OverflowError: can't convert negative int to unsigned



```python
#net = cv2.dnn.readNetFromTensorflow('MINIST_CNN_frozen_graph.pb')
import cv2 as cv2
net = cv2.dnn.readNetFromTensorflow('saved_model.pb')
layersNames = net.getLayerNames()
print(layersNames)
```


    ---------------------------------------------------------------------------

    error                                     Traceback (most recent call last)

    <ipython-input-15-e26c50dec49a> in <module>
          1 #net = cv2.dnn.readNetFromTensorflow('MINIST_CNN_frozen_graph.pb')
          2 import cv2 as cv2
    ----> 3 net = cv2.dnn.readNetFromTensorflow('saved_model.pb')
          4 layersNames = net.getLayerNames()
          5 print(layersNames)
    

    error: OpenCV(4.1.2) C:\projects\opencv-python\opencv\modules\dnn\src\tensorflow\tf_io.cpp:42: error: (-2:Unspecified error) FAILED: ReadProtoFromBinaryFile(param_file, param). Failed to parse GraphDef file: saved_model.pb in function 'cv::dnn::ReadTFNetParamsFromBinaryFileOrDie'
    



```python
img = np.zeros((28,28), dtype=np.uint8)  #  이미지 일때는 uint8이어야 한다. 그래야 에러 없다.
blob = cv2.dnn.blobFromImage(img) #  학습할때 256으로 여기서 나누어야 한다.
print(blob.shape)
print(blob.dtype)
```

    (1, 1, 28, 28)
    float32
    


```python
#N: number of images in the batch
#H: height of the image
#W: width of the image
#C: number of channels of the image
    
#blob = blob.reshape(1, 28,28,1)  #NCHW
blob = blob.reshape(1, 1, 28,28)  #NHWC
net.setInput(blob)
out = net.forward()
print(out)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-138-8bd2c27f65a2> in <module>
          1 #blob = blob.reshape(1, 28,28,1)
    ----> 2 blob = blob.reshape(0, 1, 28,28)
          3 net.setInput(blob)
          4 out = net.forward()
          5 print(out)
    

    ValueError: cannot reshape array of size 784 into shape (0,1,28,28)


# CIFAR


```python
from keras.models import load_model
model = load_model('cifar10.h5')
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 30, 30, 32)        9248      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 15, 15, 64)        18496     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 64)        36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 6, 6, 64)          0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 6, 6, 64)          36928     
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 4, 4, 64)          36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 2, 2, 64)          0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 2, 2, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               131584    
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 276,138
    Trainable params: 276,138
    Non-trainable params: 0
    _________________________________________________________________
    


```python
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, './', 'cifar.pb', as_text=False)
tf.train.write_graph(frozen_graph, './', 'cifar.pbtxt', as_text=True)
```

    INFO:tensorflow:Froze 272 variables.
    INFO:tensorflow:Converted 272 variables to const ops.
    




    './cifar.pbtxt'




```python
net = cv2.dnn.readNetFromTensorflow('cifar.pb')
layersNames = net.getLayerNames()
print(layersNames)
```

    ['conv2d_1_2/convolution', 'conv2d_1_2/Relu', 'conv2d_2_2/convolution', 'conv2d_2_2/Relu', 'max_pooling2d_1_2/MaxPool', 'conv2d_3_2/convolution', 'conv2d_3_2/Relu', 'conv2d_4_2/convolution', 'conv2d_4_2/Relu', 'max_pooling2d_2_2/MaxPool', 'conv2d_5_2/convolution', 'conv2d_5_2/Relu', 'conv2d_6_2/convolution', 'conv2d_6_2/Relu', 'max_pooling2d_3_2/MaxPool', 'flatten_1_2/Shape', 'flatten_1_2/strided_slice', 'flatten_1_2/Prod', 'flatten_1_2/stack_6803', 'flatten_1_2/Reshape', 'dense_1_2/MatMul', 'dense_1_2/Relu', 'dense_2_2/MatMul', 'dense_2_2/Softmax']
    


```python
#from keras.utils import to_categorical
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```


```python
print(test_images.shape)
```

    (10000, 32, 32, 3)
    


```python
img = test_images[0,:,:,:] / 255
print(img.shape)

plt.imshow(img)
print(img)
```

    (32, 32, 3)
    [[[0.61960784 0.43921569 0.19215686]
      [0.62352941 0.43529412 0.18431373]
      [0.64705882 0.45490196 0.2       ]
      ...
      [0.5372549  0.37254902 0.14117647]
      [0.49411765 0.35686275 0.14117647]
      [0.45490196 0.33333333 0.12941176]]
    
     [[0.59607843 0.43921569 0.2       ]
      [0.59215686 0.43137255 0.15686275]
      [0.62352941 0.44705882 0.17647059]
      ...
      [0.53333333 0.37254902 0.12156863]
      [0.49019608 0.35686275 0.1254902 ]
      [0.46666667 0.34509804 0.13333333]]
    
     [[0.59215686 0.43137255 0.18431373]
      [0.59215686 0.42745098 0.12941176]
      [0.61960784 0.43529412 0.14117647]
      ...
      [0.54509804 0.38431373 0.13333333]
      [0.50980392 0.37254902 0.13333333]
      [0.47058824 0.34901961 0.12941176]]
    
     ...
    
     [[0.26666667 0.48627451 0.69411765]
      [0.16470588 0.39215686 0.58039216]
      [0.12156863 0.34509804 0.5372549 ]
      ...
      [0.14901961 0.38039216 0.57254902]
      [0.05098039 0.25098039 0.42352941]
      [0.15686275 0.33333333 0.49803922]]
    
     [[0.23921569 0.45490196 0.65882353]
      [0.19215686 0.4        0.58039216]
      [0.1372549  0.33333333 0.51764706]
      ...
      [0.10196078 0.32156863 0.50980392]
      [0.11372549 0.32156863 0.49411765]
      [0.07843137 0.25098039 0.41960784]]
    
     [[0.21176471 0.41960784 0.62745098]
      [0.21960784 0.41176471 0.58431373]
      [0.17647059 0.34901961 0.51764706]
      ...
      [0.09411765 0.30196078 0.48627451]
      [0.13333333 0.32941176 0.50588235]
      [0.08235294 0.2627451  0.43137255]]]
    


![png](200130_day16_loadkerascv_files/200130_day16_loadkerascv_25_1.png)



```python
o = model.predict(img.reshape(1,32,32,3))

np.argmax(o)


```




    3




```python
img = (img*255).astype(dtype='uint8')

blob = cv2.dnn.blobFromImage(img, 1/255) #  학습할때 256으로 나누었음.
print(blob.shape)
print(blob.dtype)

#why ? 보통 1,32,32,3으로 넘겨야함
```

    (1, 3, 32, 32)
    float32
    


```python
#
net.setInput(blob)
out = net.forward()
print(out)
```


    ---------------------------------------------------------------------------

    error                                     Traceback (most recent call last)

    <ipython-input-112-f34891369778> in <module>
          1 #
          2 net.setInput(blob)
    ----> 3 out = net.forward()
          4 print(out)
    

    error: OpenCV(4.1.2) C:\projects\opencv-python\opencv\modules\dnn\src\dnn.cpp:525: error: (-2:Unspecified error) Can't create layer "flatten_1_2/Shape" of type "Shape" in function 'cv::dnn::dnn4_v20190902::LayerData::getLayerInstance'
    



```python
def freeze_graph(model_dir, output_node_names):
  """Extract the sub graph defined by the output nodes and convert 
  all its variables into constant 
  Args:
      model_dir: the root folder containing the checkpoint state file
      output_node_names: a string, containing all the output node's names, 
                          comma separated
                        """
  if not tf.gfile.Exists(model_dir):
    raise AssertionError(
      "Export directory doesn't exists. Please specify an export "
      "directory: %s" % model_dir)

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  # We retrieve our checkpoint fullpath
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path
    
  # We precise the file fullname of our freezed graph
  absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
  output_graph = absolute_model_dir + "/frozen_model.pb"
  print(output_graph)

  # We clear devices to allow TensorFlow to control on which device it will load operations
  clear_devices = True

  # We start a session using a temporary fresh Graph
  with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, # The session is used to retrieve the weights
      tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
      output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

  return output_graph_def



```


```python
estimator_model = tf.keras.estimator.model_to_estimator(keras_model = model, model_dir = './k')
#k 폴더에 keras폴더에 chekc point 파일 저장함
# out이름 어덯게 알지?
freeze_graph('./k/keras/',"activation_3/Sigmoid")

#./k/keras/frozen_model.pb에 생성

""
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using the Keras model provided.
    INFO:tensorflow:Using config: {'_model_dir': './k', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x0000026EEC757D68>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
    ./k/keras/frozen_model.pb
    INFO:tensorflow:Restoring parameters from ./k/keras/keras_model.ckpt
    INFO:tensorflow:Froze 12 variables.
    INFO:tensorflow:Converted 12 variables to const ops.
    56 ops in the final graph.
    




    ''




```python
model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 64)                192       
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_3 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 64)                256       
    _________________________________________________________________
    activation_2 (Activation)    (None, 64)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 65        
    _________________________________________________________________
    activation_3 (Activation)    (None, 1)                 0         
    =================================================================
    Total params: 12,993
    Trainable params: 12,865
    Non-trainable params: 128
    _________________________________________________________________
    


```python
model.layers[7].name
```




    'activation_3'


