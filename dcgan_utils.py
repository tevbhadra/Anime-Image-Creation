from keras.models import load_model
import numpy as np

#reloads a particular model into memory - note that the model needs to be compiled again after loading
#this model had earlier been saved into one of the image saving directories
def reload_model(p, run, epoch):
    image_path = p + "/" + str(run) + " run " + str(epoch) + " epoch"
    stacked_G_D = load_model(image_path + '/adversarial_model.hd5')
    gen = load_model(image_path + '/generator.hd5')
    dis = load_model(image_path + '/discriminator.hd5')
    return gen, dis, stacked_G_D
	
def flip_results(y_comb):
    for y in y_comb:
        if y == 1:
            y = 0
        else:
            y = 1
    return y_comb


def normalize(data, min, max):
    norm = ((max-min)*(data/255)) + min
    return norm


def get_x(batch_size, factor, train_data):
    x = int(batch_size*factor)
    random_index = np.random.randint(0, len(train_data) - x)
    temp = train_data[random_index:random_index + x].reshape(x, 32, 32, 3)
    legit_img = normalize(temp, -1, 1)
    gen_noise = np.random.uniform(-1, 1, (batch_size-x, 100))
    return legit_img, gen_noise

	
def get_y(batch_size, factor):
    y = int(batch_size*factor)
    y_legit = np.ones((y, 1)) - 0.1
    y_fake = np.zeros((batch_size-y, 1))
    return y_legit, y_fake




