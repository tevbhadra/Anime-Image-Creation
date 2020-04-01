import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import keras

def plot_images(image_path, gen, save2file=False, samples=64, step=0):
    filename = image_path + "/sample_%d.png" % step
    noise = np.random.uniform(-1, 1, (samples, 100))
    # noise = np.random.normal(0, 1, (samples, 100))
    images = gen.predict(noise)
    images = (images * 127.5) + 127.5
    images = images.astype(np.uint8)
    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(8, 8, i + 1)
        image = images[i, :, :, :]
        image = np.reshape(image, [32, 32, 3])
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout(pad = 0.25)
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

		
def grad_visual(model, shape, epoch, p):
    out = model.get_layer(index=len(model.layers) - 1).output
    loss_val = K.mean(out[:, :, :, 0])
    grads = K.gradients(loss_val, model.layers[0].input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.layers[0].input], [loss_val, grads])
    loss_value, grads_value = iterate([np.zeros((1, shape))])
    plt.plot(range(shape), grads_value.reshape(shape))
    plt.savefig(p + "/epoch %d.png" %epoch)
    return plt


def plot_loss(des_loss, gen_loss, epoch, p):
    plt.clf()
    plt.plot(range(epoch), des_loss, color='b', alpha = 0.5, label="Discriminator Loss")
    plt.plot(range(epoch), gen_loss, color='r', alpha = 0.5, label="Generator Loss")
    plt.legend()
    # plt.show()
    plt.savefig(p + "/loss_plot.png")
    return plt

#generating sample images from the Generator
def plot_img(ct, root, gen):
    plt.clf()
    for i in range(ct):
        x123 = np.random.randint(0, 255, 100).reshape(1, 100)
        im123 = gen.predict((x123 - 127.5) / 127.5)
        plt.imshow(im123.reshape(32, 32, 3))
        plt.savefig(root + "test_samples/image_" + str(ct+1))