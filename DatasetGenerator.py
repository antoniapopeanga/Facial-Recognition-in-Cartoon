import os
from PIL import Image
import random

class DatasetGenerator:
    def __init__(self, base_path="antrenare", positive_base_folder="data/exemplePozitive", negative_folder="data/exempleNegative"):
        self.base_path = base_path
        self.data_folder='data'
        self.positive_base_folder = positive_base_folder
        self.negative_folder = negative_folder
        self.characters = ['dad', 'mom', 'dexter', 'deedee', 'unknown']
        
        self.crop_sizes= {
            'rectangle_vertical': (54, 66),    
            'square': (54, 54),               
            'rectangle_horizontal': (66, 54)   
        }
        self.character_sizes={
            'mom': (120,96),
            'dad':(96,80),
            'deedee':(80,112),
            'dexter':(96,96)
        }
    def _create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return True
        return False

    def _is_folder_empty(self, folder_path):
        if not os.path.exists(folder_path):
            return True
        return len(os.listdir(folder_path)) == 0

    def get_crop_type(self, width, height):
        #determinam forma in functie de raport latime/inaltime
        aspect_ratio = width / float(height)
        
        if aspect_ratio > 1.2:  
            return 'rectangle_horizontal'
        elif aspect_ratio < 0.8:  
            return 'rectangle_vertical'
        else:  
            return 'square'

    def generate_positive_examples(self):
        if not self._is_folder_empty(self.positive_base_folder):
            print("Exemplele pozitive au fost deja generate.")
            return

        for shape in self.crop_sizes.keys():
            shape_folder = os.path.join(self.positive_base_folder, shape)
            self._create_folder(shape_folder)
            
        for character in self.characters:
            char_image_dir = os.path.join(self.base_path, character)
            annotation_file = os.path.join(self.base_path, f"{character}_annotations.txt")
            
            if not os.path.exists(annotation_file):
                continue
            
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue
                    
                    image_name, x1, y1, x2, y2, label = parts
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    image_path = os.path.join(char_image_dir, image_name)
            
                    
                    try:
                        with Image.open(image_path) as img:
                         
                            img_width, img_height = img.size
                            
                            x1_buffered = max(0, x1 - 10)
                            y1_buffered = max(0, y1 - 10)
                            x2_buffered = min(img_width, x2 + 10)
                            y2_buffered = min(img_height, y2 + 10)
                            
                            cropped = img.crop((x1_buffered, y1_buffered, x2_buffered, y2_buffered))
                            
                            
                            width, height = x2_buffered - x1_buffered, y2_buffered - y1_buffered
                            crop_type = self.get_crop_type(width, height)
                            
                          
                            crop_size = self.crop_sizes[crop_type]
                            cropped_resized = cropped.resize(crop_size)
                            
                            #salvam in folderul corespunzator formei
                            shape_folder = os.path.join(self.positive_base_folder, crop_type)
                            self._create_folder(shape_folder)
                            output_filename = f"{label}_{image_name.split('.')[0]}_{x1_buffered}_{y1_buffered}_{x2_buffered}_{y2_buffered}.jpg"
                            output_path = os.path.join(shape_folder, output_filename)
                            cropped_resized.save(output_path)
                            print(f"Poza: {output_filename} afost salavata in folderul {shape_folder}")
                    except Exception as e:
                        print(f"Eroare la procesarea imaginii {image_path}: {str(e)}")

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

    def load_annotations(self):
        annotations = {}
        
        for character in self.characters:
            annotation_file = os.path.join(self.base_path, f"{character}_annotations.txt")
            if not os.path.exists(annotation_file):
                continue
                
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue
                        
                    image_name, x1, y1, x2, y2, label = parts
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    if image_name not in annotations:
                        annotations[image_name] = []
                    annotations[image_name].append((x1, y1, x2, y2, label))
                    
        return annotations

    def generate_negative_examples(self):
        if not self._is_folder_empty(self.negative_folder):
            print("Exemplele negative au fost deja generate.")
            return

        self._create_folder(self.negative_folder)
        annotations = self.load_annotations()
        
        #definim limitele pentru crop-ul random
        patch_configs = [
            {'min_width': 64, 'max_width': 128, 'min_height': 96, 'max_height': 192},  
            {'min_width': 96, 'max_width': 192, 'min_height': 64, 'max_height': 128}, 
            {'min_width': 64, 'max_width': 128, 'min_height': 64, 'max_height': 128}   
        ]
        
        for character in self.characters:
            char_image_dir = os.path.join(self.base_path, character)
            if not os.path.exists(char_image_dir):
                continue
                
            for image_name in os.listdir(char_image_dir):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(char_image_dir, image_name)
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        face_boxes = annotations.get(image_name, [])
                        
                        
                        for patch_idx, patch_config in enumerate(patch_configs):
                            attempts = 0
                            sample_generated = False
                            
                            while not sample_generated and attempts < 100:
                                attempts += 1
                                
                                
                                patch_width = random.randint(patch_config['min_width'], patch_config['max_width'])
                                patch_height = random.randint(patch_config['min_height'], patch_config['max_height'])
                                
                               
                                x1 = random.randint(0, max(0, width - patch_width))
                                y1 = random.randint(0, max(0, height - patch_height))
                                x2 = min(x1 + patch_width, width)
                                y2 = min(y1 + patch_height, height)
                                
                                patch_box = (x1, y1, x2, y2)
                                is_valid = True
                                
                                #verificam sa nu fie overlap-ul prea mare intre patch si bounding box-ul fetei
                                for face_box in face_boxes:
                                    iou = self.intersection_over_union(patch_box, face_box[:4])
                                    if iou > 0.1:
                                        is_valid = False
                                        break
                                
                                if is_valid:
                                    cropped = img.crop(patch_box)
                                    aspect_type = ['vertical', 'horizontal', 'square'][patch_idx]
                                    output_filename = f"negative_{character}_{image_name.split('.')[0]}_{aspect_type}_{x1}_{y1}_{x2}_{y2}.jpg"
                                    output_path = os.path.join(self.negative_folder, output_filename)
                                    cropped.save(output_path)
                                    sample_generated = True
                                    print(f"Exemplul negativ salvat: {output_filename}")
                                 
                except Exception as e:
                    print(f"Eroare la procesarea imaginii {image_path}: {str(e)}")

    def generate_positive_examples_by_character(self):
        characters = ['dad', 'mom', 'dexter', 'deedee']
        all_generated = True
        for character in characters:
            character_folder = os.path.join(self.data_folder, f"{character}_examples")
            positive_folder = os.path.join(character_folder, "positive_examples")
            if not os.path.exists(positive_folder):
                all_generated = False
                break

        if all_generated:
            print("Exemplele pozitive pentru fiecare personaj au fost deja generate.")
            return

        for character in characters:

            character_folder = os.path.join(self.data_folder, f"{character}_examples")
            positive_folder = os.path.join(character_folder, "positive_examples")
            os.makedirs(positive_folder, exist_ok=True)

            #preluam exemplele pozitive din folder-ul fiecarei forme
            for shape in self.crop_sizes.keys():
                shape_folder = os.path.join(self.positive_base_folder, shape)
                if not os.path.exists(shape_folder):
                    continue

                for image_name in os.listdir(shape_folder):
                    if image_name.lower().startswith(character.lower()):
                        src_path = os.path.join(shape_folder, image_name)

                        
                        with Image.open(src_path) as img:
                           
                            target_height, target_width = self.character_sizes[character]
                            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                            dst_path = os.path.join(positive_folder, image_name)
                            img.save(dst_path)
                            print(f"Imaginea {image_name} cu personajul {character} a fost salvata")


        

    def generate_negative_examples_by_character(self):
        characters = ['dad', 'mom', 'dexter', 'deedee']
        NUM_NEGATIVE_EXAMPLES = 7000  
        
    
        all_generated = True
        for character in characters:
            character_folder = os.path.join(self.data_folder, f"{character}_examples")
            negative_folder = os.path.join(character_folder, "negative_examples")
            if not os.path.exists(negative_folder):
                all_generated = False
                break
        
        if all_generated:
            print("Exemplele negative pentru fiecare personaj au fost deja generate.")
            return
            
        for character in characters:
    
            character_folder = os.path.join(self.data_folder, f"{character}_examples")
            negative_folder = os.path.join(character_folder, "negative_examples")
            
           
            if os.path.exists(negative_folder) and len(os.listdir(negative_folder)) > 0:
                print(f"Exemplele negative pentru {character} exista deja.")
                continue
                
            os.makedirs(negative_folder, exist_ok=True)
            
            #selectam poze random din folderul de exemple negative
            print(f"Procesam exemplele negative generale pentru {character}...")
            all_negatives = [img for img in os.listdir(self.negative_folder) 
                            if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            selected_negatives = random.sample(all_negatives, NUM_NEGATIVE_EXAMPLES)
            
            print(f"Selectam {NUM_NEGATIVE_EXAMPLES} exemple negative aleatorii pentru {character}...")
            for neg_image in selected_negatives:
                src_path = os.path.join(self.negative_folder, neg_image)
                dst_path = os.path.join(negative_folder, f"neg_{neg_image}")
                try:
                    with Image.open(src_path) as img:
                        img.save(dst_path)
                        print(f"Imaginea {neg_image} a fost salvata in folderul {dst_path}")
                except Exception as e:
                    print(f"Eroare la copierea imaginii {neg_image}: {str(e)}")

            if os.path.exists(negative_folder):
                neg_count = len([f for f in os.listdir(negative_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"In total sunt {neg_count} exemple negative pentru {character}")