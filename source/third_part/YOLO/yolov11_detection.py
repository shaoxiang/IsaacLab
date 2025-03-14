from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import torch

# Load a model
model = YOLO("yolo11n.pt")
class_names = model.names
print("class_names:", class_names)

# Process the image
source1 = cv2.imread('cars.jpg')
source2 = cv2.imread('bus.jpg')
source3 = cv2.imread('one_person.jpg')
source4 = cv2.imread('two_person.jpg')
source5 = cv2.imread('two_person.png')
source6 = cv2.imread('three_person.png')
source7 = cv2.imread('one_person.png')
source8 = cv2.imread('biped_demo.png')

imgs = [source1, source2, source3, source4, source5, source6, source7, source8]
results = model(imgs, verbose = False)

for result in results:
    det_annotated = result.plot(show=True)

# boxes
# 'cls', 'conf', 'cpu', 'cuda', 'data', 'id', 'is_track', 'numpy', 'orig_shape', 'shape', 'to', 'xywh', 'xywhn', 'xyxy', 'xyxyn'

print("results:", dir(results))
print("results boxes:", dir(results[0].boxes), results[0].boxes.cls)
print("results boxes len:", len(results[0].boxes), len(results[0].boxes.cls))
print("results boxes len:", len(results[1].boxes), len(results[1].boxes.cls))

# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# print("results names:", results[0].names)
# print("results keypoints:", results[0].keypoints)
# print("results boxes:", dir(results[0].boxes))
# print("results xyxy:", dir(results[0].boxes.xyxyn.cpu()))

def yolo_results_filter(yolo_results, max_person = 3, choose_device = 'cpu'):
    new_results = torch.zeros(8, max_person, 5, device = choose_device)
    print(new_results)
    
    for i, yolo_result in enumerate(yolo_results):
        person_num = 0
        tmp_num = 0
        tmp_results = torch.zeros(max_person, 5, device = choose_device)
        for index, box in enumerate(yolo_result.boxes.xyxyn.cuda()):
            print("box:", box, yolo_result.boxes.cls[index])
            one_result = torch.cat((box, (yolo_result.boxes.cls[index] + 1).unsqueeze(0)), dim=0).to(choose_device)
            if yolo_result.boxes.cls[index] == 0.:
                new_results[i][person_num] = one_result
                person_num += 1
            else:
                if tmp_num < max_person:
                    tmp_results[tmp_num] = one_result
                    tmp_num += 1

            if person_num >= max_person:
                break

        if person_num < max_person:
            for tmp_index, tmp_result in enumerate(tmp_results):
                if person_num + tmp_index >= max_person:
                    break
                new_results[i][person_num + tmp_index] = tmp_result

    print(new_results)
    return new_results

# Extract results
annotator = Annotator(source1, example=model.names)

for index, box in enumerate(results[0].boxes.xyxyn.cuda()):
    print("box:", box, results[0].boxes.cls[index])
    width, height, area = annotator.get_bbox_dimension(box)
    print(f"width:{width}, height:{height}, area:{area}")
    print("Bounding Box Width {}, Height {}, Area {}".format(width.item(), height.item(), area.item()))


person_obs = []
for index, box in enumerate(results[1].boxes.xyxyn.cuda()):
    print("box:", box, results[1].boxes.cls[index])
    if results[1].boxes.cls[index] == 0.:
        result = torch.cat((box, results[1].boxes.cls[index].unsqueeze(0)), dim=0)
        person_obs.append(result)

    width, height, area = annotator.get_bbox_dimension(box)
    print(f"width:{width}, height:{height}, area:{area}")
    print("Bounding Box Width {}, Height {}, Area {}".format(width.item(), height.item(), area.item()))

print(person_obs)

yolo_results_filter(results)

# for index, box in enumerate(results[1].boxes.xyxyn.cpu()):
#     print("box:", box, results[1].boxes.cls[index])

# for i in len(results):
#     for index, box in enumerate(results[i].boxes.xyxyn.cpu()):
#         print("box:", box, results[i].boxes.cls[index])