import torch
from matplotlib import pyplot as plt
import os
import cv2

def display_image(img,my_title=None):
  '''
  Custom function to display single image with its title
  '''
  fig, axes = plt.subplots(1, figsize=(15,15))
  if len(img.shape) == 2 : # grayscale , only 1 channel
    plt.imshow(img,cmap='gray')
  else:
    plt.imshow(img)
  if my_title is not None:
    plt.title(my_title)
  plt.axis('off');


# tt = all_data[0]['image']
# xx = all_data[1]['image']
#
# torch.save(tt, './tensor1.pt')
# torch.save(xx, './tensor2.pt')


# print(os.getcwd())
pathy = '/home/chris/Desktop/'
# # img = cv2.imread(pathy)
# xx = torch.load(pathy + 'tensor2.pt')
# # zz = torch.load('./tensors2.pt')
# # display_image(img)
# plt.imshow(xx.permute(1,2,0))
# plt.show()

#######################################################################################################################
#
# import json
# pathy = '/home/chris/Desktop/'
# # Opening JSON file
# # val_data  -- 214354 qa_s
# # captions_val2014
# f = open(pathy + 'captions_val2014.json')
# # returns JSON object as
# # a dictionary
# data = json.load(f)
# #dict_keys(['info', 'images', 'licenses', 'annotations'])
# subdata = data['annotations'][:1000]
# #print(subdata)
# image_ids = []
# captions = []
# for s in subdata:
#   image_ids.append(s['image_id'])
#   captions.append(s['caption'])
# skilros= '/media/chris/4f8d85a4-7412-4e22-89be-f483a57450c0/home/morf/Desktop/tera_Downloads'
#
# # for i in image_ids:
# #   filename = f"/val2014/COCO_val2014_{int(i):012d}.jpg"
# #   break
#
# filename = f"/val2014/COCO_val2014_{int(image_ids[-1]):012d}.jpg"
# img = cv2.imread(skilros + filename)
# print(img)
# display_image(img,captions[-1])
# plt.show()
#
# # print(image_ids[-1])
# # print(captions[-1])


#######################################################################################################################


import json
pathy = '/home/chris/Desktop/'
f = open(pathy + 'val_data.json')
data = json.load(f)
subdata = data[:1000]
# image_id - answer - caption
print(subdata)

# print(image_ids[-1])
# print(captions[-1])
