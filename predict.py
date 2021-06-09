import numpy as np
import cv2, random
from torch.jit.annotations import try_real_annotations
from torchvision import transforms
from utils import *
from args import get_args
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from datetime import datetime

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def detect(original_image, model, device, min_score, max_overlap, top_k, transform, suppress=None):
    
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    if transform:
        image = normalize(to_tensor(resize(original_image)))
        image = image.to(device) # Move to default device
    else:
        #image = to_tensor(resize(original_image))
        image = np.array(original_image)
        image = cv2.resize(image, (300, 300))
        image = to_tensor(image)
        #image = normalize(to_tensor(image))
        image = image.to(device) # Move to default device

    # print('\npass 1')

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # print('\npass 2')

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores, min_score=min_score,
        max_overlap=max_overlap, top_k=top_k
    )
    # print('\npass 3')

    det_labels_id = det_labels.copy()
    det_labels_id = det_labels_id[0].to('cpu').tolist()
    det_scores = det_scores[0].to('cpu').tolist()
    #print('det_boxes, det_labels, det_scores=',det_boxes, det_labels, det_scores)
    # print('\npass 4')

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    # print('\npass 5')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    # print('\npass 6')

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    # print('\npass 7')

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    '''if det_labels == ['background']:
        # Just return original image
        return original_image'''
    
    # print('\npass 8')

    # Annotate
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./Lato-Regular.ttf", 15)
    box_location_list = []

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
    
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness
        
        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
        box_location_list.append(box_location)
        #print(box_location)
        #print(det_labels,det_labels_id)
    del draw
    # print('\npass 9')
    return annotated_image, box_location_list, det_labels_id, det_scores


def detect_(original_image, model, device, min_score, max_overlap, top_k, transform, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    if transform:
        image = normalize(to_tensor(resize(original_image)))
        image = image.to(device) # Move to default device
    else:
        #image = to_tensor(resize(original_image))
        image = np.array(original_image)
        image = cv2.resize(image, (300, 300))
        image = to_tensor(image)
        #image = normalize(to_tensor(image))
        image = image.to(device) # Move to default device

    # print('\npass 1')

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # print('\npass 2')

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores, min_score=min_score,
        max_overlap=max_overlap, top_k=top_k
    )
    # print('\npass 3')

    det_labels_id = det_labels.copy()
    det_labels_id = det_labels_id[0].to('cpu').tolist()
    det_scores = det_scores[0].to('cpu').tolist()
    #print('det_boxes, det_labels, det_scores=',det_boxes, det_labels, det_scores)
    # print('\npass 4')

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    # print('\npass 5')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_boxes = det_boxes.detach().numpy()
    # print('\npass 6')

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    # print('\npass 7')

    #print(det_boxes)

    return det_boxes, det_labels_id, det_scores


def predict(args):
    device = args.device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(device)

    images_path = args.testPATH + '/' if args.testPATH[-1] != '/' else args.testPATH
    images = os.listdir(images_path)#[:50]

    # Load model checkpoint
    # checkpoint = 'checkpoint_ssd300.pth.tar'
    checkpoint = torch.load(args.savePATH + args.dt + args.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    #lst_img = []
    out_dict = {'image_filename': [], 'label_id': [], 'x': [], 'y': [], 'w': [], 'h': [], 'confidence': []}
    
    if args.detect:
        random.seed(4028)
        images = random.choices(images,  k=5)
        print('The output files have been saved as')
        for name in images:
            img = Image.open(images_path + name, mode='r')
            img = img.convert('RGB')
            annotated_image, box_location_list, id_det_labels, det_scores = detect(img, model=model, device=device, min_score=args.min_score, max_overlap=.5, top_k=200, transform=args.transform)
            fn = args.savePATH + args.dt + 'annotated_' + name
            annotated_image.save(fn)
            print('box_location_list', box_location_list)
            print('id_det_labels', id_det_labels)
            print('det_scores', det_scores)
            print('-->', fn)
            print()

    else:
        for name in tqdm(images):
            img = Image.open(images_path + name, mode='r')
            img = img.convert('RGB')
            box_location_list, id_det_labels, det_scores = detect_(
                img, model=model, device=device,
                min_score=args.min_score, max_overlap=.5,
                top_k=200, transform=args.transform
            )
            out_dict['image_filename'] += list(np.repeat(name, len(id_det_labels)))
            out_dict['label_id'] += id_det_labels
            out_dict['confidence'] += det_scores

            for box in box_location_list:
                out_dict['x'].append(box[0])
                out_dict['y'].append(box[1])
                out_dict['w'].append(box[2])
                out_dict['h'].append(box[3])

        return out_dict


if __name__ == '__main__':
    args = get_args().parse_args()
    dt = datetime.now().strftime('%d-%H-%M-%S')
    
    if args.detect:
        predict(args)
    else:
        out_dict = predict(args)
        df = pd.DataFrame(out_dict)
        fn = args.savePATH + args.dt + 'output_ms=' + str(args.min_score) + '_transform=' + str(args.transform) + '_' + dt + '.csv'
        df.loc[df.label_id != 0,:].to_csv(fn, index=None)
        print('The output file has been saved as')
        print('-->', fn)
