import torchvision .datasets as datasets
import torchvision .models as models
import torchvision . transforms as transforms
import crypten

crypten.init()

# download and set up ImageNet dataset:
transform = transforms.ToTensor ()
dataset = datasets.ImageNet("/imagenet_folder" , transform =transform ,)
# secret share pre−trained ResNet−18 on GPU:
model = models.resnet18(pretrained =True)
model_enc = crypten.nn.from_pytorch(model , dataset [0],).encrypt ()
# perform inference on secret−shared images:
for image in dataset:
    image_enc = crypten.cryptensor(image)
    output_enc = model_enc(image_enc)
    output = output_enc.get_plain_text ()
