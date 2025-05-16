import os

class Parameters:
    def __init__(self):
        self.base_dir = 'data'
        self.dir_pos_examples = os.path.join(self.base_dir, 'exemplePozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_test_examples ='test'  #directorul pentru exemplele de test

        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        self.path_annotations ='validare/task1_gt_validare.txt'

    
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # Setăm parametrii
        self.dim_window = 36  # Exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # Dimensiunea celulei
        self.dim_descriptor_cell = 36  # Dimensiunea descriptorului unei celule
        self.overlap = 0.3  # Suprapunerea minimă pentru ferestrele glisante
        self.number_positive_examples = 5813  # Numărul exemplelor pozitive
        self.number_negative_examples = 8000  # Numărul exemplelor negative
        self.has_annotations = False  # Indică dacă există adnotări pentru datele de antrenare
        self.threshold = 2.5 # Pragul de clasificare
