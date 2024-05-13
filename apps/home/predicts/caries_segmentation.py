from image2base64.converters import rgb2base64
from ..unet import UNet
from ..helper import *
from ..preprocessing import *

def predict(image):
    imageSet = SimDataset(image)
    imageLoader = torch.utils.data.DataLoader(imageSet, batch_size=1, shuffle=False, num_workers=0)

    for input in imageLoader:
        # Setup Model
        device = torch.device("cpu")
        model = UNet(1)
        model.load_state_dict(torch.load('models/Unet_Model_Croped_Dataset_BCE.pth', map_location=device))
        model.eval()

        # Predict
        inputData = input.to(device)
        pred = model(inputData)
        pred = pred.cpu().detach().numpy()
        inputs_cpu = inputData.data.cpu().numpy()
        combinedImage = combine_image(inputs_cpu[0], pred[0])

        # Base64
        base64 = rgb2base64(combinedImage, 'JPEG')

    return base64

def predictSegmentationFullImage(image):
    imageSet = SimDatasetFullImage(image)
    imageLoader = torch.utils.data.DataLoader(imageSet, batch_size=1, shuffle=False, num_workers=0)
    base64 = ''

    for inputs in imageLoader:
        trims = []
        for split_input in inputs:
            # Setup Model
            device = torch.device("cpu")
            model = UNet(1)
            model.load_state_dict(torch.load('models/Unet_Model_Split_Dataset_BCE.pth', map_location=device))
            model.eval()

            # Predict
            inputData = split_input.to(device)
            pred = model(inputData)
            pred = pred.cpu().detach().numpy()
            inputs_cpu = inputData.data.cpu().numpy()
            combinedImage = combine_image(inputs_cpu[0], pred[0])
            trims.append(trim_from_np(combinedImage))

        # Gabungkan array secara horizontal dan vertical
        top_row = np.concatenate((trims[0], trims[1]), axis=1)
        bottom_row = np.concatenate((trims[2], trims[3]), axis=1)
        final_result = np.concatenate((top_row, bottom_row), axis=0)

        # Base64
        base64 = rgb2base64(final_result, 'JPEG')

    return base64