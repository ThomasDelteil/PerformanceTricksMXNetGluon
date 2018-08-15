import mxnet as mx
from mxnet import gluon, nd, autograd

from skimage.transform import resize
import matplotlib.pyplot as plt

#################################################################
# Model                                                         #
#################################################################
ctx = mx.cpu()
net = gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=ctx)


#################################################################
# Data                                                          #
#################################################################
BATCH_SIZE = 8

def transform(x, y):
    x = resize(x.asnumpy(), (224, 224), anti_aliasing=False, mode='constant')
    x = x.transpose((2, 0, 1)).astype('float32')
    return x, y

dataset_train = gluon.data.vision.CIFAR10(train=True, transform=transform)
dataloader_train = gluon.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, last_batch="discard")


#################################################################
# Loss                                                         #
#################################################################
loss_fn = gluon.loss.SoftmaxCELoss()


#################################################################
# Trainer                                                       #
#################################################################
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': 0.9, 'wd': 0.00001})


#################################################################
# Metric                                                        #
#################################################################
accuracy = mx.metric.Accuracy()

#################################################################
# Training Loop                                                 #
#################################################################
epoch = 1
for e in range(epoch):
    for i, (data, label) in enumerate(dataloader_train):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        # Forward pass and loss computation
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)

        # Compute gradient
        loss.backward()

        # Update network weights
        trainer.step(data.shape[0])

        # Update metric
        accuracy.update(label, output)

#conv2d = net.features[1]
#weights = conv2d.weight.data().transpose((0, 2, 3, 1)).reshape((56,56,3)).asnumpy()
#plt.imshow(weights)
#plt.show()
