import jittor as jt
from jittor import init, rand, randint
import os
import numpy as np
from jittor import nn
from mydataset import myDataset

if jt.has_cuda:
    jt.flags.use_cuda = 1

class option:
    def __init__(self,n=100,
                      batch_size=4,
                      lr=0.0005,
                      b1=0.5,
                      b2=0.9,
                      n_cpu=8,
                      latent_dim=100,
                      n_classes=17,
                      img_size=180,
                      channels=3,
                      sample_interval=100) :
        self.n_epochs=n
        self.batch_size=batch_size
        self.lr=lr
        self.b1=b1
        self.b2=b2
        self.cpu=n_cpu
        self.latent_dim=latent_dim
        self.n_classes=n_classes
        self.img_size=img_size
        self.channels=channels
        self.sample_interval=sample_interval

opt=option()
# here is to set command line options, use -h operation we can check these arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
# parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
# parser.add_argument('--lr', type=float, default=0.0005, help='adam: learning rate')
# parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
# parser.add_argument('--n_classes', type=int, default=4, help='number of classes for dataset')
# parser.add_argument('--img_size', type=int, default=80, help='size of each image dimension')
# parser.add_argument('--channels', type=int, default=3, help='number of image channels')
# parser.add_argument('--sample_interval', type=int, default=100, help='interval between image sampling')
# opt = parser.parse_args()  # assign all args to opt
print(opt) 

img_shape = (opt.channels, opt.img_size, opt.img_size) # 3*32*32 dimention of the pic, used when reshaping


# using cnn to generate:

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, 10)
        input_dim = ((opt.latent_dim + 10) )
        self.init_size = (opt.img_size // 4)
        self.l1 = nn.Sequential(nn.Linear(input_dim, (128 * (self.init_size ** 2))))
        self.conv_blocks = nn.Sequential(nn.BatchNorm(128),
                                         nn.Upsample(scale_factor=2), 
                                         nn.Conv(128, 128, 3, stride=1, padding=1), 
                                         nn.BatchNorm(128, eps=0.8), 
                                         nn.LeakyReLU(scale=0.2), 
                                         nn.Upsample(scale_factor=2), 
                                         nn.Conv(128, 64, 3, stride=1, padding=1), 
                                         nn.BatchNorm(64, eps=0.8), 
                                         nn.LeakyReLU(scale=0.2), 
                                         nn.Conv(64, opt.channels, 3, stride=1, padding=1), 
                                         nn.Tanh())
        def weights_init_normal(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') != (- 1)):
                init.gauss_(m.weight, mean=0.0, std=0.02)
            elif (classname.find('BatchNorm') != (- 1)):
                init.gauss_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, value=0.0)
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, noise, labels):
        gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)        
        out = self.l1(gen_input)
        out = out.view((out.shape[0], 128, self.init_size, self.init_size))
        img = self.conv_blocks(out)
        return img



# using all fc layer:

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__() # use nn.Module's __init__() to initialize generator. super means generator is a superclass of nn.Module
#         self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes) #generate data from 1~10, and each for 10, so it's 10*10

#         # use block to define layers, used in nn.Sequential to generate sequential layers
#         def block(in_feat, out_feat, normalize=True):
#             """
#             each block:
#                 1. linner layer (in_feat->out_feat)
#                 2. if normalize==true, add batchnorm1d layer
#                 3. leakyrelu layer
#             """
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:  # BatchNorm??????????????????????????????????????????????????????????????????????????????????????????????????????
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2))
#             return layers
#         # Sequential would add activated funtion automatically
#         self.model = nn.Sequential(*block((opt.latent_dim + opt.n_classes), 128, normalize=False),
#                                    *block(128, 256), 
#                                    *block( 256,512), 
#                                    *block(512, 1024), 
#                                    nn.Linear(1024, int(np.prod(img_shape))), # generate a picture
#                                    nn.Tanh())

#     def execute(self, noise, labels):
#         gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)
#         img = self.model(gen_input)
#         img = img.view((img.shape[0], *img_shape))
#         return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(opt.n_classes, 10)
        
        self.model = nn.Sequential(nn.Linear((10 + int(np.prod(img_shape))), 256),
                                   nn.LeakyReLU(0.2), 
                                   nn.Linear(256, 256), 
                                   nn.Dropout(0.4), 
                                   nn.LeakyReLU(0.2), 
                                   nn.Linear(256, 128), 
                                   nn.Dropout(0.4), 
                                   nn.LeakyReLU(0.2), 
                                   nn.Linear(128, 1))
                      

    def execute(self, img, labels):
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

# Configure data loader
# from jittor.dataset.mnist import MNIST
# import jittor.transform as transform
# transform = transform.Compose([
#     transform.Resize(opt.img_size),
#     transform.Gray(),
#     transform.ImageNormalize(mean=[0.5], std=[0.5]),
# ])
# dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

dogData=myDataset()
dataloader = dogData.set_attrs(batch_size=opt.batch_size, shuffle=True)

optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

from PIL import Image
def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    elif C==1:
        img = img[:,:,0]
    Image.fromarray(np.uint8(img)).save(path)

def sample_image(n_row, batches_done):
    # Sample noise
    z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
    labels = jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.numpy(), "images/%d.png" % batches_done, nrow=n_row)

def sample_generate(batches_done):
    label=randint(0,opt.n_classes)
    z = jt.array(np.random.normal(0, 1, (1, opt.latent_dim))).float32().stop_grad()
    gen_imgs = generator(z, label)
    save_image(gen_imgs.numpy(), "images/%dclass%d.png" % (batches_done,label), nrow=1)



# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = jt.ones([batch_size, 1]).float32().stop_grad()
        fake = jt.zeros([batch_size, 1]).float32().stop_grad()

        # Configure input
        real_imgs = jt.array(imgs)
        labels = jt.array(labels)

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise and labels as generator input
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
        gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.sync()
        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.sync()
        optimizer_D.step(d_loss)
        if i  % 20 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
            )
        # if g_loss<0.2:
        #     sample_generate( batches_done=batches_done)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_generate( batches_done=batches_done)

    sample_generate( batches_done=batches_done)
    if epoch % 5 == 0:
        generator.save("saved_models/generator_last.pkl")
        discriminator.save("saved_models/discriminator_last.pkl")
    if epoch %10 ==0:
        opt.lr*=0.9