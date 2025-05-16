import numpy as np
import os
import glob
import cv2 as cv
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from copy import deepcopy
import ntpath
import glob
import timeit
import albumentations as A

class CharacterFacialDetector:
    def __init__(self, params):
        self.params = params
        
        self.models = {
            'dad': None,
            'mom': None,
            'dexter': None,
            'deedee': None
        }

        self.window_sizes={
            'mom': (120,96),
            'dad':(96,80),
            'deedee':(80,112),
            'dexter':(96,96)
        }


        self.transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.5), p=0.3),
            A.RandomBrightnessContrast(p=0.3), 
            A.MedianBlur(blur_limit=3, p=0.2), 
            A.Downscale(scale_range=[0.6,0.9], p=0.3)  
        ], p=1.0)



    def get_positive_descriptors(self, character):
        images_path = os.path.join('data', f"{character}_examples",'positive_examples', '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        
        window_size = self.window_sizes[character]
        print(window_size)
        
        print(f'Procesam descriptorii pozitivi pentru {num_images} de imagini cu personajul {character} ...')
        for i in range(num_images):
            print(f'Exemplul pozitiv {i+1}/{num_images}...')
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            
            img = cv.resize(img, (window_size[1], window_size[0]))
            
            #imaginea originala
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(features)

            if self.params.use_flip_images:
                    features = hog(np.fliplr(img), 
                                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(2, 2), feature_vector=True)
                    positive_descriptors.append(features)

            # adaugam imagini transformate cu libraria albumentations
            for _ in range(2):
                transformed_image = self.transform(image=img)['image']
                features = hog(transformed_image, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)
                
                if self.params.use_flip_images:
                    features = hog(np.fliplr(transformed_image), 
                                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                cells_per_block=(2, 2), feature_vector=True)
                    positive_descriptors.append(features)

        return np.array(positive_descriptors)

    def get_negative_descriptors(self, character):  
        images_path = os.path.join('data', f"{character}_examples", "negative_examples", '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []
        
       
        window_size = self.window_sizes[character]
        
        print(f'Procesam descriptorii negativi pentru {num_images} de imagini pentru personajul {character} ...')
        
        for i in range(num_images):
            print(f'Procesam exemplul negativ {i+1}/{num_images}...')
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            
            img = cv.resize(img, (window_size[1], window_size[0]))
            
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(features)

            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(2, 2), feature_vector=True)
                negative_descriptors.append(features)

        return np.array(negative_descriptors)
    
    
    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]


    def train_classifier(self, character):

        svm_file_name = os.path.join(self.params.dir_save_files, f'best_model_{character}_{self.params.dim_hog_cell}')
        
        if os.path.exists(svm_file_name):
            print(f"Clasificatorul pentru personajul {character} a fost deja antrenat")
            self.models[character] = pickle.load(open(svm_file_name, 'rb'))
            return

        positive_features = self.get_positive_descriptors(character)
        negative_features = self.get_negative_descriptors(character)
            
     
        training_examples = np.concatenate((positive_features, negative_features), axis=0)
        train_labels = np.concatenate((np.ones(len(positive_features)), np.zeros(len(negative_features))))

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        
        for c in Cs:
            print(f'Antrenam clasificatorul pentru personajul {character} cu c={c}')
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print(f'Cel mai bun model pentru personajul {character} este c = {best_c}')
        pickle.dump(best_model, open(svm_file_name, 'wb'))
        self.models[character] = best_model

        
    def run(self):
        """
        Functia run() adaptata pentru a detecta personajele separat.
        """
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        results = {character: {"detections": [], "scores": [], "file_names": []} for character in self.models.keys()}
        num_test_images = len(test_files)
       
        #procentul de scalare
        scales = [0.05,0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8,0.9,1]

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print(f'Procesam imaginea de testare {i+1}/{num_test_images}...')

            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            initial_shape = img.shape

            detections_per_image = {character: [] for character in self.models.keys()}
            scores_per_image = {character: [] for character in self.models.keys()}

            for scale in scales:

                scaled_height = int(img.shape[0] * scale)
                scaled_width = int(img.shape[1] * scale)
                scaled_img = cv.resize(img, (scaled_width, scaled_height))

                # Parcurgem fiecare clasificator de personaj
                for character, model in self.models.items():
                    if model is None:
                        continue

                    # Dimensiunea ferestrei pentru personaj
                    window_height, window_width = self.window_sizes[character]
                    w = model.coef_.T
                    bias = model.intercept_[0]

                    hog_descriptors = hog(scaled_img, 
                                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2), 
                                        feature_vector=False)

                    num_cell_in_template_height = window_height // self.params.dim_hog_cell - 1
                    num_cell_in_template_width = window_width // self.params.dim_hog_cell - 1
                    num_cols = scaled_img.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = scaled_img.shape[0] // self.params.dim_hog_cell - 1

                    # Fereastra glisanta
                    for y in range(0, num_rows - num_cell_in_template_height + 1, 1):
                        for x in range(0, num_cols - num_cell_in_template_width + 1, 1):
                            descr = hog_descriptors[y:y + num_cell_in_template_height,
                                                x:x + num_cell_in_template_width].flatten()

                            score = np.dot(descr, w)[0] + bias

                            if score > self.params.threshold:
                                
                                x_min = int(x * self.params.dim_hog_cell / scale)
                                y_min = int(y * self.params.dim_hog_cell / scale)
                                x_max = int((x * self.params.dim_hog_cell + window_width) / scale)
                                y_max = int((y * self.params.dim_hog_cell + window_height) / scale)

                                detections_per_image[character].append([x_min, y_min, x_max, y_max])
                                scores_per_image[character].append(score)

            # NMS
            for character in self.models.keys():
                if len(scores_per_image[character]) > 0:
                    detections = np.array(detections_per_image[character])
                    scores = np.array(scores_per_image[character])

                    nms_detections, nms_scores = self.non_maximal_suppression(
                        detections,
                        scores,
                        initial_shape)

                    if len(nms_scores) > 0:
                        results[character]["detections"].extend(nms_detections.tolist())
                        results[character]["scores"].extend(nms_scores.tolist())
                        short_name = ntpath.basename(test_files[i])
                        image_names = [short_name for _ in range(len(nms_scores))]
                        results[character]["file_names"].extend(image_names)

            end_time = timeit.default_timer()
            print(f'Timpul de procesare al imaginii de testare {i+1}/{num_test_images} este {end_time - start_time:.2f} sec.')

        return results
