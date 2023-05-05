# import os
# import numpy as np
# import skimage.io as io
# from pycocotools.coco import COCO
#
# # set the path to the COCO dataset
# dataDir = '/data1/datasets/coco2017'
#
#
# # load the annotations for the training set
# dataType = 'train2017'
# annFile = os.path.join(dataDir, 'annotations', 'instances_{}.json'.format(dataType))
# coco = COCO(annFile)
#
# # get the category ID for the "person" category
# catIds = coco.getCatIds(catNms=['person'])
#
# # generate a random image ID and load the image
# while True:
#     imgId = np.random.choice(coco.getImgIds(), 1)[0]
#     img = coco.loadImgs(imgId)
#     if img is not None and len(img) > 0:
#         break
#
# # load the image
# I = io.imread(os.path.join(dataDir, dataType, img[0]['file_name']))
#
# # create a mask for the "person" instances in the image
# annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# mask = np.zeros((img[0]['height'], img[0]['width']))
# for ann in anns:
#     mask += coco.annToMask(ann)
#
# # display the image and the mask
# io.imshow(I)
# io.imshow(mask, alpha=0.5)
# io.show()


import matplotlib.pyplot as plt
import numpy as np

# Generate some random image data
height = 224
width = 224
channels = 3
image = np.random.rand(height, width, channels)

# Display the image using imshow
plt.imshow(image)
plt.show()

# Save the image to a file
plt.savefig('my_image.png')