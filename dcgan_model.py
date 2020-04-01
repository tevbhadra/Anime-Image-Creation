from keras.layers import Dense, Conv2D, Dropout, Flatten, Reshape, BatchNormalization, Conv2DTranspose, \
    UpSampling2D, PReLU

from keras.optimizers import Adam
from keras.models import Sequential
import os
from dcgan_utils import *
from plot import *
from read_data import *


# from keras.utils import multi_gpu_model

# trying Wassertein loss - didn't work!
# def d_loss(y_true, y_pred):
#     return K.mean(y_true*y_pred)

print("Enter the root directory")
x = input()
# root = "D:/Document/06 DAAN 897 - Deep Learning/Project/GAN/"
root = x

#model creation start
def generator():
    dropout = 0.2
    dim = 16
    model = Sequential()

    model.add(Dense(256 * dim * dim, input_shape=(100,), kernel_initializer='RandomNormal'))
    model.add(PReLU())
    model.add(BatchNormalization(momentum=0.5))
    # model.add(Activation('relu'))
    model.add(Reshape((dim, dim, 256)))
    # model.add(Dropout(dropout))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(128, kernel_size=5, padding='same'))
    model.add(PReLU())
    model.add(BatchNormalization(momentum=0.5))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    model.add(Conv2DTranspose(64, kernel_size=5, padding='same'))
    model.add(PReLU())
    model.add(BatchNormalization(momentum=0.5))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    model.add(Conv2DTranspose(32, kernel_size=5, padding='same'))
    model.add(PReLU())
    model.add(BatchNormalization(momentum=0.5))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    model.add(Conv2DTranspose(3, 5, padding='same', activation='tanh'))

    return model


gen = generator()
gen.summary()


def discriminator():
    dropout = 0.2
    model = Sequential()

    model.add(Conv2D(32, 5, strides=2, input_shape=(32, 32, 3), padding='same'))
    model.add(PReLU())
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.5))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, 5, strides=2, padding='same'))
    model.add(PReLU())
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.5))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, 5, strides=2, padding='same'))
    model.add(PReLU())
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.5))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, 5, strides=2, padding='same'))
    model.add(PReLU())
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization(momentum=0.5))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    return model


dis = discriminator()
# dis.summary()
optimizer_DM = Adam(lr=0.00002, beta_1=0.5)
dis.compile(optimizer=optimizer_DM, loss='binary_crossentropy')


def stacked_G_D(G, D):
    # D.trainable = False
    model = Sequential()
    model.add(G)
    model.add(D)
    return model


stacked_G_D = stacked_G_D(gen, dis)
optimizer_AM = Adam(lr=0.00001, beta_1=0.5)
stacked_G_D.compile(optimizer=optimizer_AM, loss='binary_crossentropy')
#model creation end


#read data start
print ("Enter the path for the images")
x = input()
print ("Enter the images you want to train the model on (<14000) - ")
y = input()
train_data = readImagesShort(x, scale_size = 32, count = int(y))
train_data = np.array(train_data)
np.random.shuffle(train_data)
#read data end
print("Creating required directories....")

home_dir = root + "dcgan_output_samples/"
if (os.path.isdir(home_dir)==0):
    os.mkdir(home_dir)

allFiles = glob.glob(home_dir + "/*")
temp = [x.split("\\")[-1] for x in allFiles]
temp2 = [int(x.split(" ")[0]) for x in temp]
if len(temp2)==0:
    count = 1
else:
    count = max(temp2)+1
p = home_dir + str(count) + " run"
dir = os.mkdir(p)


def set_trainable(model, flag):
    for l in model.layers:
        l.trainable = flag
    return model

print("Calling TensorBoard callbcks....")
tb_dis = keras.callbacks.TensorBoard(
  log_dir = root+'/dis_logs/',
  histogram_freq = 0,
  batch_size = 32,
  write_graph = True,
  write_grads = True
)
tb_dis.set_model(dis)

tb_gan = keras.callbacks.TensorBoard(
  log_dir = root+'/gan_logs',
  histogram_freq = 0,
  batch_size = 32,
  write_graph = True,
  write_grads = True
)
tb_dis.set_model(stacked_G_D)


run = 0
des_loss = []
gen_loss = []

print("Intializing the training function...")

def train(data, G, D, stacked_G_D, epoch, run, plot_interval, model_save_interval):
    print ("Executing...")
    image_path = p + "/" + str(run) + " run " + str(epoch) + " epoch"
    os.mkdir(image_path)
    batch = 32
    save_interval = int(plot_interval)
    # flips = batch // 16
    factor = 0.5
    dis_train_factor = 10
    gen_train_factor = 1
    for cnt in range(epoch):
        D = set_trainable(D, True)

        for _ in range(dis_train_factor):
            legit_img, gen_noise = get_x(batch, factor, data)
            # gen_noise = np.random.normal(0, 1, (batch//2, 100))  # noise being fed to the generator
            synthetic_img = G.predict(gen_noise)
            # x_comb = np.concatenate((legit_img, synthetic_img))
            y_legit, y_fake = get_y(batch, factor)

            # flipping some results from 1 to 0 and vice versa
            # rand_num = np.random.randint(0, int(batch*factor), flips)
            # y_legit[[rand_num]] = flip_results(y_legit[[rand_num]])
            D_loss_fake = D.train_on_batch(synthetic_img, y_fake)
            D_loss_real = D.train_on_batch(legit_img, y_legit)
            tb_dis.on_batch_end(cnt, D_loss_real)
            tb_dis.on_batch_end(cnt, D_loss_fake)
        for k in range(gen_train_factor):
            D = set_trainable(D, False)
            noise = np.random.uniform(-1, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))
            G_loss = stacked_G_D.train_on_batch(noise, y_mislabled)
            tb_gan.on_batch_end(cnt, G_loss)
        print('epoch: %d, [Discriminator :: d_loss_real: %f, d_loss_fake: %f], [Generator :: loss: %f]' % (cnt, D_loss_real, D_loss_fake, G_loss))
        if cnt % save_interval == 0:
            plot_images(image_path, gen=G, save2file=True, step=cnt)
			
        if cnt % int(model_save_interval) == 0 and cnt != 0:
            grad_visual(G, 100, cnt, p)
            gen.save(image_path + "/generator.hd5")
            dis.save(image_path + "/discriminator.hd5")
            stacked_G_D.save(image_path + "/adversarial_model.hd5")
            plot_loss(des_loss, gen_loss, len(gen_loss), p)
    return image_path

print ("Number of Steps you want this file to run")
x = input()
epoch, run = int(x), run + 1

print("Enter the plotting interval")
x = input()

print("Enter the model save interval")
y = input()

im_path = train(train_data, gen, dis, stacked_G_D, epoch, run, plot_interval=int(x), model_save_interval=int(y))
print ("Execution complete! Check the images generated at this location - ")
print (str(im_path))

### needed in csae you need to reload a previous model
gen, dis, stacked_G_D = reload_model(p, run, epoch)

#needed in case you need to generate samples from your generator
os.mkdir(root + "/test_samples/")
plot_img(10, root, gen)


