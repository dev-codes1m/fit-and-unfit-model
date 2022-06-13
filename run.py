import io
from PIL import Image

from tensorflow import keras
from scipy.spatial import distance

a = keras.models.load_model('transfer.h5')  # Loading the Model
IMG_SIZE = [224, 224]
from keras.preprocessing import image


# Finding the Euclidean Distance between the Parameter of cloth and Body

def Euclidean(chestsizecloth, chestsizebody, waistsizecloth, waistsizebody):
    # a = np.array((chestsizecloth,waistsizecloth))
    # b = np.array((chestsizebody,waistsizebody))
    a = (chestsizecloth, waistsizecloth)
    b = (chestsizebody, waistsizebody)
    # dist = np.sqrt(np.sum(np.square(a-b)))
    return distance.euclidean(a, b)


def fitUnfitModel(bodyImage, chestsizecloth, chestsizebody, waistsizecloth, waistsizebody):
    # parser = argparse.ArgumentParser(
    #     description="My Model"
    # )
    # parser.add_argument('--path', metavar='path', type=str, help='image path format: --path destination\img.jpg')
    # parser.add_argument('--body_dim', dest='bodydim', required=False, help="Body dimension format: --bodydim 40 50",
    #                     nargs='+', type=int, default=[100, 100, 100])
    # parser.add_argument('--cloth_dim', dest='clothdim', required=False,
    #                     help="cloth dimension part format: --clothdim 40 50", nargs='+', type=int,
    #                     default=[100, 100, 100])
    # args = parser.parse_args()
    # chestsizebody = args.bodydim[0]
    # waistsizebody = args.bodydim[1]
    # chestsizecloth = args.clothdim[0]
    # waistsizecloth = args.clothdim[1]

    print(type(chestsizecloth))
    Ed = Euclidean(chestsizecloth, chestsizebody, waistsizecloth, waistsizebody)

    # path = "args.path"
    # # print(path)
    # img = image.load_img(path, target_size=[224, 224])
    #
    # x = image.img_to_array(img)

    img = Image.open(io.BytesIO(bodyImage))
    img = img.resize((224, 224))
    x = image.img_to_array(img)

    x = x / 255.0  # scale Factor of Image

    import numpy as np

    x = np.expand_dims(x, axis=0)

    # import matplotlib.pyplot as plt

    # plt.imshow(img)

    k = a.predict([x]).round()

    # print(k)
    isFit = False
    if k[0][0] > k[0][1]:
        isFit = True

    print(k[0][0])
    print(Ed)
    fitness_score = (k[0][0] + 1 - (Ed / 80)) / 2  # Fitness Score calculation{(Score of Unfit and Fit Model)
    # + 1 - ((ED Between Various Parameter of Body and CLoths)/(80(General Calculations)}
    #
    # print(f"Fitness Score: {}%")  # Fitness Score*100 Scale to get Percentage Value

    fitness_score_percentage = round(fitness_score * 100, 3)

    return fitness_score, isFit
