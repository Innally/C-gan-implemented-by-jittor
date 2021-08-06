import jittor as jt
from jittor import nn
import numpy as np
import pylab as pl
import matplotlib
import matplotlib.pyplot as plt
from jittor import init, rand, randint
matplotlib.use('TkAgg')


# %matplotlib inline
batch_size=64
# 隐空间向量长度
latent_dim = 100
# 类别数量
n_classes = 10
# 图片大小
img_size = 32
# 图片通道数量
channels = 1
# 图片张量的形状
img_shape = (channels, img_size, img_size)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, 10)
        input_dim = ((latent_dim + 10) )
        self.init_size = (img_size // 4)
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
                                         nn.Conv(64, channels, 3, stride=1, padding=1), 
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear((n_classes + int(np.prod(img_shape))), 512), 
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 512), 
            nn.Dropout(0.4), 
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 512), 
            nn.Dropout(0.4), 
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 1))

    def execute(self, img, labels):
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        validity = self.model(d_in)
        return validity


if __name__=="__main__":
    # 定义模型
    generator = Generator()
    discriminator = Discriminator()
    generator.eval()
    discriminator.eval()

    # 加载参数
    generator.load('./saved_models/generator_last_final.pkl')
    discriminator.load('./saved_models/discriminator_last_final.pkl')

    # 定义一串数字
    # number =input("input which class of dog do you wanna generate(input 0-119)")
    # n_row = 1
    # z = jt.array(np.random.normal(0, 1, (n_row, latent_dim))).float32().stop_grad()
    # labels = jt.array(np.array(int(number))).float32().stop_grad()
    # gen_imgs = generator(z,labels)

    # pl.imshow(gen_imgs.data.transpose((1,2,0,3))[0].reshape((gen_imgs.shape[2], -1)))
    # Configure data loader


    from jittor.dataset.mnist import MNIST
    import jittor.transform as transform
    transform = transform.Compose([
        transform.Resize(img_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])
    adversarial_loss = nn.MSELoss()
    rloss=0
    floss=0
    
    dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)
    print(len(dataloader))
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = jt.ones([batch_size, 1]).float32().stop_grad()
        fake = jt.zeros([batch_size, 1]).float32().stop_grad()

            # Configure input
        real_imgs = jt.array(imgs)
        labels = jt.array(labels)
            # -----------------
            #  Test Generator
            # -----------------

            # Sample noise and labels as generator input
        z = jt.array(np.random.normal(0, 1, (batch_size, latent_dim))).float32()
        gen_labels = jt.array(np.random.randint(0, n_classes, batch_size)).float32()

            # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
          
        # ---------------------
        #  Test Discriminator
        # ---------------------
        # Loss for real images
        validity_real = discriminator(real_imgs, labels)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        d_fake_loss = adversarial_loss(validity_fake, fake)
        rloss+=d_real_loss
        floss+=d_fake_loss

        print(i,rloss)
        print(i,floss)
        # print(d_real_loss)
        # print(d_fake_loss)
    print(rloss/len(dataloader))
    print(floss/len(dataloader))
           
            
