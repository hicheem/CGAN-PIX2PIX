import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

class DataLoader():
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.path_to_input_images_folder = 'PATH to input'
        self.path_to_output_images_folder = 'PATH to output'

    def load_data(self, batch_size=1, is_testing=False):
        
        path_input_images = os.listdir(self.path_to_input_images_folder)
        path_input_images.sort()
        path_output_images = os.listdir(self.path_to_output_images_folder)
        path_output_images.sort()
        batch_images = np.random.choice(len(path_input_images), size=batch_size)

        imgs_A = []
        imgs_B = []
        
        
        for i in batch_images:
            input_image = self.imread(self.path_to_input_images_folder + path_input_images[i])
            output_image = plt.imread(self.path_to_output_images_folder + path_output_images[i])
            if len(output_image.shape) == 3:
                r, g, b = output_image[:,:,0], output_image[:,:,1], output_image[:,:,2]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                output_image = gray
            
            img_A, img_B = output_image, input_image

            img_A = cv2.resize(img_A, self.img_res)
            img_B = cv2.resize(img_B, self.img_res)
            img_A = np.expand_dims(img_A, axis = -1 )
            

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B 
    # The type of imgs_A is np.array and its shape is (batch_size,img_res[0],img_res[1],channels=3)
    # imgs_A refers to groudtruth in training set; imgs_B refers to noise in training set 
    # imgs_A and imgs_B return the dataset at random.
    def load_batch(self, batch_size=1, is_testing=False):
        path_input_images = os.listdir(self.path_to_input_images_folder)
        path_input_images.sort()
        path_output_images = os.listdir(self.path_to_output_images_folder)
        path_output_images.sort()
        
        self.n_batches = int(len(path_input_images) / batch_size)
        
        k = 0
        for i in range(self.n_batches-1):
            imgs_A, imgs_B = [], []
            for batch in range(batch_size):
                img_B = self.imread(self.path_to_input_images_folder + path_input_images[k])
                img_A = plt.imread(self.path_to_output_images_folder + path_output_images[k])
                k += 1
                if len(img_A.shape) == 3:
                    r, g, b = img_A[:,:,0], img_A[:,:,1], img_A[:,:,2]
                    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    img_A = gray
                img_B = cv2.resize(img_B, self.img_res)
                img_A = cv2.resize(img_A, self.img_res)
                img_A = np.expand_dims(img_A, axis = -1)
                imgs_B.append(img_B)
                imgs_A.append(img_A)          
            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            
            yield imgs_A, imgs_B  ## This function looks like a generator
    ## imgs_A and imgs_B return the dataset in order.
    def imread(self, path):
        return cv2.cvtColor(cv2.imread(path) , cv2.COLOR_BGR2RGB).astype(np.float)
        
