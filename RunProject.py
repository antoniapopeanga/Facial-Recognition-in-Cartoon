import os
import numpy as np
import glob
from Parameters import *
from DatasetGenerator import *
from FacialDetector import *
from CharacterFacialDetector import *



params = Parameters()
params.dim_hog_cell = 6
params.number_negative_examples = 12002
params.threshold = 0
params.use_hard_mining = False
params.use_flip_images = True

print("=========================================================================")
print("Task 1. Recunoasterea tuturor fetelor din desenul Laboratorul lui Dexter")
print("=========================================================================\n")


#formele pentru clasificatori+dimensiuni
shapes = ['rectangle_vertical', 'square', 'rectangle_horizontal']
shape_dimensions = {
            'rectangle_vertical': (54, 66),    
            'square': (54, 54),               
            'rectangle_horizontal': (66, 54)   
        }


#1. Generam exemplele pozitive si negative din folderul 'antrenare/'
print("Generam exemplele pozitive si negative...")

dataset_generator = DatasetGenerator()
dataset_generator.generate_positive_examples()
dataset_generator.generate_negative_examples()


detector = FacialDetector(params)

#2. Generam/incarcam descriptorii pozitivi si negativi pentru fiecare forma
for shape in shapes:
   
    params.number_positive_examples = len(glob.glob(os.path.join(params.dir_pos_examples, shape, '*.jpg')))
    if params.use_flip_images:
        params.number_positive_examples *= 2
    
    #descriptorii pozitivi
    positive_features_path = os.path.join(params.dir_save_files, 
                                        f'descriptors_positive_{shape}_{params.dim_hog_cell}_{params.number_positive_examples}.npy')
    if os.path.exists(positive_features_path):
        positive_features = np.load(positive_features_path)
        print(f'Au fost deja incarcati descriptorii pozitivi pentru forma {shape}')
    else:
        print(f'Procesam descriptorii pozitivi pentru forma {shape}:')
        positive_features = detector.get_positive_descriptors(shape)
        np.save(positive_features_path, positive_features)
        print(f'Descriptorii pozitivi au fost salvati in {positive_features_path}')

    # descriptorii negativi
    negative_features_path = os.path.join(params.dir_save_files, 
                                        f'descriptors_negative_{shape}_{params.dim_hog_cell}_{params.number_negative_examples}.npy')
    if os.path.exists(negative_features_path):
        negative_features = np.load(negative_features_path)
        print(f'Au fost deja incarcati descriptorii negativi pentru forma {shape}')
    else:
        print(f'Procesam descriptorii negativi pentru forma {shape}:')
        negative_features = detector.get_negative_descriptors(shape)
        np.save(negative_features_path, negative_features)
        print(f'Descriptorii negativi au fost salvati in {negative_features_path}')
    
    #3. Antrenam clasificatorul pentru fiecare forma
    print(f"\nAntrenam clasificatorul pentru forma {shape}...")
    detector.train_classifier(shape)

solution_dir="341_Popeanga_Antonia/task1/"
os.makedirs(solution_dir, exist_ok=True)

#folderul cu imagini de test
images_path = os.path.join(params.dir_test_examples, '*.jpg')
test_files = glob.glob(images_path)
num_images = len(test_files)

print(f"\nTestam detectorul facial pe {num_images} de imagini...")

detections, scores, filenames, shapes_detected = detector.run()

#4. Salvam detectiile
np.save(os.path.join(solution_dir, 'detections_all_faces.npy'), detections)
np.save(os.path.join(solution_dir, 'scores_all_faces.npy'), scores)
np.save(os.path.join(solution_dir, 'file_names_all_faces.npy'), filenames)

print(f"Rezultatele detectorului au fost salvate in folderul: {solution_dir}")

print("============================================================================================")
print("Task 2. Recunoasterea fetelor fiecarui personaj in parte din desenul Laboratorul lui Dexter")
print("============================================================================================\n")

characters=["dad","deedee","dexter","mom"]

#modificam dimensiunea celulei hog
params.dim_hog_cell = 8
params.use_flip_images = True

#1. Generam exemplele pozitive si negative pentru fiecare personaj
print("Generam exemplele pozitive si negative in functie de personaj")
dataset_generator.generate_positive_examples_by_character()
dataset_generator.generate_negative_examples_by_character()

character_detector = CharacterFacialDetector(params)

#2. Generam/incarcam descriptorii negativi si pozitivi
for character in characters:
    params.number_positive_examples = len(glob.glob(os.path.join('data', character, '_examples','positive_examples','*.jpg')))
    if params.use_flip_images:
        params.number_positive_examples *= 2
    
    #descriptorii pozitivi
    positive_features_path = os.path.join(params.dir_save_files, 
                                        f'descriptors_positive_{character}_{params.dim_hog_cell}_{params.number_positive_examples}.npy')
    if os.path.exists(positive_features_path):
        positive_features = np.load(positive_features_path)
        print(f'Au fost deja incarcati descriptorii pozitivi pentru personajul {character}')
    else:
        print(f'Procesam descriptorii pozitivi pentru personajul {character}:')
        positive_features =character_detector.get_positive_descriptors(character)
        np.save(positive_features_path, positive_features)
        print(f'Descriptorii pozitivi au fost salvati in {positive_features_path}')

    # descriptorii negativi
    params.number_negative_examples = len(glob.glob(os.path.join('data', character, '_examples','negative_examples','*.jpg')))
    if params.use_flip_images:
        params.number_positive_examples *= 2
    negative_features_path = os.path.join(params.dir_save_files, 
                                        f'descriptors_negative_{character}_{params.dim_hog_cell}_{params.number_negative_examples}.npy')
    if os.path.exists(negative_features_path):
        negative_features = np.load(negative_features_path)
        print(f'Au fost deja incarcati descriptorii negativi pentru personajul {character}')
    else:
        print(f'Procesam descriptorii negativi pentru personajul {character}')
        negative_features = character_detector.get_negative_descriptors(character)
        np.save(negative_features_path, negative_features)
        print(f'Descriptorii negativi au fost salvati in {negative_features_path}')
    
    #3. Antrenam modelul specializat pentru fiecare personaj
    print(f"\nAntrenam clasificatorul pentru personajul {character}...")
    character_detector.train_classifier(character)


solution_dir_characters="341_Popeanga_Antonia/task2/"
os.makedirs(solution_dir_characters, exist_ok=True)


print(f"\nTestam detectorul facial specializat pentru fiecare personaj pe {num_images} de imagini...")


results = character_detector.run()

for character, data in results.items():
    detections = np.array(data["detections"])
    scores = np.array(data["scores"])
    filenames = np.array(data["file_names"])
    
    #5. Salvam detectiile, scorul si numele fisierelor pentru fiecare personaj
    np.save(os.path.join(solution_dir_characters, f'detections_{character}.npy'), detections)
    np.save(os.path.join(solution_dir_characters, f'scores_{character}.npy'), scores)
    np.save(os.path.join(solution_dir_characters, f'file_names_{character}.npy'), filenames)

print(f"Rezultatele detectorului au fost salvate in folderul: {solution_dir_characters}")