import os
import math
import numpy as np
import pandas as pd
import ast

from utils.words import *
#from utils.coco.coco import *
#from utils.coco.coco import *
from pycocotools.coco import *

class DataSet():
    def __init__(self, img_ids, img_files, caps=None, masks=None, batch_size=1, is_train=False, shuffle=False):
        self.img_ids = np.array(img_ids)
        self.img_files = np.array(img_files)
        self.caps = np.array(caps)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.img_ids)
        self.num_batches = int(self.count * 1.0 / self.batch_size)
        self.current_index = 0
        self.indices = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_idx = self.indices[start:end]
        img_files = self.img_files[current_idx]
        img_ids = self.img_ids[current_idx]
        if self.is_train:
            caps = self.caps[current_idx]
            masks = self.masks[current_idx]
            self.current_index += self.batch_size
            return img_files, caps, masks, img_ids
        else:
            self.current_index += self.batch_size
            return img_files

    def has_next_batch(self):
        """ Determine whether there is any batch left. """
        return self.current_index + self.batch_size <= self.count


def prepare_train_data(args):
    """ Prepare relevant data for training the model. """
    image_dir, caption_file, segmentation_file, annotation_file, train_segmentation_annotation_dir = args.train_image_dir, args.train_caption_file, args.train_instance_file, args.train_annotation_file, args.train_segmentation_annotation_dir
    init_embed_with_glove, vocab_size, word_table_file, glove_dir = args.init_embed_with_glove, args.vocab_size, args.word_table_file, args.glove_dir
    dim_embed, batch_size, max_sent_len = args.dim_embed, args.batch_size, args.max_sent_len

    coco_captions = COCO(caption_file)
    coco_captions.filter_by_cap_len(max_sent_len)
    
    coco_segmentations = COCO(segmentation_file)

    print("Building the word table...")
    word_table = WordTable(vocab_size, dim_embed, max_sent_len, word_table_file)
    if not os.path.exists(word_table_file):
        if init_embed_with_glove:
            word_table.load_glove(glove_dir)
        word_table.build(coco_captions.all_captions())
        word_table.save()
    else:
        word_table.load()
    print("Word table built. Number of words = %d" %(word_table.num_words))

    coco_captions.filter_by_words(word_table.all_words())
    overlap_image_ids = get_image_ids_with_cap_and_seg(coco_captions, coco_segmentations)
    
    if not os.path.exists(annotation_file):
        annotations = process_captions(coco_captions, overlap_image_ids, image_dir, annotation_file)
    else:
        annotations = pd.read_csv(annotation_file)

    img_ids = annotations['image_id'].values
    img_files = annotations['image_file'].values
    captions = annotations['caption'].values
    print("Number of training captions = %d" %(len(captions)))

    caps, masks = symbolize_captions(captions, word_table)
    #
    save_segmentations(coco_segmentations, overlap_image_ids, train_segmentation_annotation_dir)

    print("Building the training dataset...")
    dataset = DataSet(img_ids, img_files, caps, masks, batch_size, True, True)
    print("Dataset built.")
    return coco_captions, dataset

def prepare_val_data(args):
    """ Prepare relevant data for validating the model. """
    image_dir, caption_file = args.val_image_dir, args.val_caption_file

    coco = COCO(caption_file)

    img_ids = list(coco.imgs.keys())
    img_files = [os.path.join(image_dir, coco.imgs[img_id]['file_name']) for img_id in img_ids]
  
    print("Building the validation dataset...")
    dataset = DataSet(img_ids, img_files)
    print("Dataset built.")
    return coco, dataset

def prepare_test_data(args):
    """ Prepare relevant data for testing the model. """
    image_dir = args.test_image_dir

    files = os.listdir(image_dir)
    img_files = [os.path.join(image_dir, f) for f in files if f.lower().endswith('.jpg')]
    img_ids = list(range(len(img_files)))

    print("Building the testing dataset...")    
    dataset = DataSet(img_ids, img_files)
    print("Dataset built.")
    return dataset

def process_captions(coco_captions, overlap_image_ids, image_dir, annotation_file):
    """ Build an annotation file containing the training information. """
    ann_ids = coco_captions.getAnnIds(imgIds=overlap_image_ids)
    anns = coco_captions.loadAnns(ids=ann_ids)
    captions = [ann['caption'] for ann in anns]
    image_ids = [ann['image_id'] for ann in anns]
#     captions = [coco_captions.anns[ann_id]['caption'] for ann_id in coco_captions.anns if coco_captions.anns[ann_id]['image_id'] in overlap_image_ids]
# #     segmentations = [' '.join(map(str, coco_segmentations.anns[ann_id]['segmentation'][0])) for ann_id in which_ids]
# #     segmentations = [coco_segmentations.anns[ann_id]['segmentation'][0] for ann_id in coco_segmentations.anns if coco_segmentations.anns[ann_id]['image_id'] in overlap]
#     image_ids = [coco_captions.anns[ann_id]['image_id'] for ann_id in coco_captions.anns if coco_captions.anns[ann_id]['image_id'] in overlap_image_ids]
    image_files = [os.path.join(image_dir, coco_captions.imgs[img_id]['file_name']) for img_id in image_ids]
    annotations = pd.DataFrame({'image_id': image_ids, 'image_file': image_files, 'caption': captions})
    annotations.to_csv(annotation_file)
    return annotations


def symbolize_captions(captions, word_table):
    """ Translate the captions into the indicies of their words in the vocabulary, and get their masks. """
    caps = []
    masks = []
    for cap in captions:
        idx, mask = word_table.symbolize_sent(cap)
        caps.append(idx)
        masks.append(mask)
    return np.array(caps), np.array(masks)


'''
--------------------------------

Added by Logan

--------------------------------
'''
def save_segmentations(coco_segmentations, overlap_image_ids, train_segmentation_annotation_dir):
    ''' Convert segmentations to binary masks and save to disk '''
    catids = coco_segmentations.getCatIds()
    cats = coco_segmentations.loadCats(catids)
    id_to_word = {}
    for cat in cats:
        id_to_word[cat['id']] = cat['name']
        
    # For each image, get all the segmentation LABELS
    for image_id in overlap_image_ids:
        img = coco_segmentations.loadImgs(ids=image_id)[0]
        ann_ids = coco_segmentations.getAnnIds(imgIds=image_id)
        anns = coco_segmentations.loadAnns(ids=ann_ids)
        category_ids = set([ann['category_id'] for ann in anns])
        
        # For each segmentation label, get all of the segmentations under that label
        for category_id in category_ids:
            cat_name = id_to_word[category_id]
            cat_ann_ids = coco_segmentations.getAnnIds(catIds=category_id, imgIds=image_id)
            compiled_mask = np.zeros((img['height'], img['width']), dtype=bool)
            
            # For each segmentation, convert to a binary mask and add to the compiled mask
            for ann_id in cat_ann_ids:
                ann = coco_segmentations.loadAnns(ids=ann_id)[0]
                mask = coco_segmentations.annToMask(ann)
                mask = mask.astype(bool)
                compiled_mask = np.logical_or(compiled_mask, mask)
                
            # Save to disk
            file_dir = train_segmentation_annotation_dir + str(image_id)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_name = os.path.join(file_dir, cat_name + '.npz')
            np.savez(file_name, compiled_mask)
    
def get_image_ids_with_cap_and_seg(coco_captions, coco_segmentations):
    ''' Get the images that have both caption and segmentation annotations '''
    caption_image_ids = coco_captions.getImgIds()
    segmentation_image_ids = coco_segmentations.getImgIds()
    overlap = list(set(caption_image_ids) & set(segmentation_image_ids))
    return overlap

if __name__=="__main__":
    import main
    main.main(sys.argv)

