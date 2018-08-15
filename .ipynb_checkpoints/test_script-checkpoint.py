import multiprocessing
import time

import mxnet as mx
from mxnet import gluon, nd, autograd

from skimage.transform import resize

# Model
ctx = mx.gpu()
net = gluon.model_zoo.vision.resnet50_v2(pretrained=False, ctx=ctx)
net.initialize(mx.init.Xavier(magnitude=2.3), ctx=ctx)
net.hybridize(static_alloc=True, static_shape=True)

# Data
BATCH_SIZE = 96
def transform(x, y):
    x = resize(x.asnumpy(), (224, 224), anti_aliasing=False, mode='constant')
    x = x.transpose((2, 0, 1)).astype('float32')
    return x, y
dataset_train = gluon.data.vision.CIFAR10(train=True, transform=transform)
dataloader_train = gluon.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, last_batch="discard", num_workers=multiprocessing.cpu_count()-7)

net(nd.ones((BATCH_SIZE, 3, 224, 224), ctx=ctx))

# Loss
loss_fn = gluon.loss.SoftmaxCELoss()

# Trainer
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.001, 'momentum':0.9, 'wd':0.00001})

# Training Loop
epoch = 1
print_n = 20
tick_0 = time.time()

for e in range(epoch):
    tick = time.time()
    total_loss = nd.zeros((1,), ctx)
    for i, (data, label) in enumerate(dataloader_train):
        if i == 0:
            tick_0 = time.time()
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        # Forward pass and loss computation
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        total_loss += loss.mean()
        # Compute gradient
        loss.backward()
        
        # Update network weights
        trainer.step(data.shape[0])
       
        # Print batch metrics
        avg_loss = (total_loss / (i+1)).asscalar() # sync
        if i % print_n == 0 and i > 0:
            print('Batch [{}], Avg Loss {:.4f}, Samples/sec: {:.4f}'.format(
                i, avg_loss, data.shape[0]*(print_n)/(time.time()-tick))
            )
            tick = time.time()
            
        
        if i == 100:
            break
    avg_loss = (total_loss / (i+1)).asscalar() # sync 
    print('Epoch [{}], Avg Loss {:.4f}'.format(e, avg_loss))
    print('~Samples/Sec {:.4f}'.format(data.shape[0]*(i+1)/(time.time()-tick_0)))