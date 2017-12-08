import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from skimage import transform
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from dataset import *
from utils.words import *
#from utils.coco.coco import *
from pycocotools.coco import *
#from utils.coco.pycocoevalcap.eval import *
from utils.pycocoevalcap.eval import *

class ImageLoader(object):
    def __init__(self, mean_file, segmentation_dir):
        self.bgr = True 
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)
        self.segmentation_dir = segmentation_dir
        self.segmentation_scale_shape = np.array([14, 14], np.int32)

    def load_img(self, img_file):    
        """ Load and preprocess an image. """  
        img = cv2.imread(img_file)

        if self.bgr:
            temp = img.swapaxes(0, 2)
            temp = temp[::-1]
            img = temp.swapaxes(0, 2)

        img = cv2.resize(img, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        img = img[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1], :]
        img = img - self.mean
        return img

    def load_imgs(self, img_files):
        """ Load and preprocess a list of images. """
        imgs = []
        for img_file in img_files:
            imgs.append(self.load_img(img_file))
        imgs = np.array(imgs, np.float32)
        return imgs
    
    def load_segmentation(self, img_id):
        img_dir = self.segmentation_dir + str(img_id) + '/'
        label_to_seg = {}
        for file in os.listdir(img_dir):
            if file.endswith(".npz"):
                seg = np.load(os.path.join(img_dir + file))
                seg = transform.resize(seg, self.scale_shape, anti_aliasing=True)
                label = file.split('.npz')[0]
                if label == '':
                    raise Exception('no caption found for file ' + file)
                label_to_seg[label] = seg
        return label_to_seg
    
    def load_segmentations(self, img_ids):
        img_segs = []
        for img_id in img_ids:
            label_to_seg = self.load_segmentation(img_id)
            img_segs.append(self.load_img(label_to_seg))
        img_segs = np.array(img_segs, np.float32)
        return imgs

class BaseModel(object):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.batch_size = params.batch_size if mode=='train' else 1

        self.cnn_model = params.cnn_model
        self.train_cnn = params.train_cnn
        self.init_lstm_with_fc_feats = params.init_lstm_with_fc_feats if self.cnn_model=='vgg16' else False

        self.class_balancing_factor = params.class_balancing_factor

        self.save_dir = os.path.join(params.save_dir, self.cnn_model+'/')

        self.word_table = WordTable(params.vocab_size, params.dim_embed, params.max_sent_len, params.word_table_file)
        self.word_table.load()
        
        self.lemmatizer = WordNetLemmatizer()
        self.special_wn_cases = {
            'stop sign': self.syn('street_sign'),
            'sports ball': self.syn('ball'),
            'kite': self.syn('kite', 2),
            'wine glass': self.syn('wineglass'),
            'hot dog': self.syn('hot_dog', 2),
            'cake': self.syn('cake', 2),
            'potted plant': self.syn('pot plant'),
            'toilet': self.syn('toilet', 1),
            'tv': self.syn('tv', 1),
            'mouse': self.syn('mouse', 3),
            'cell phone': self.syn('cellular_telephone'),
            'microwave': self.syn('microwave', 1),
            'toaster': self.syn('toaster', 1)
            }

        self.img_loader = ImageLoader(params.mean_file, params.train_segmentation_annotation_dir)
        self.img_shape = [224, 224, 3]

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.build()
        self.saver = tf.train.Saver(max_to_keep = 100)
  


    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train, contexts=None, feats=None):
        raise NotImplementedError()

    def train(self, sess, train_coco, train_data):
        """ Train the model. """
        print("Training the model...")
        params = self.params
        num_epochs = params.num_epochs

        for epoch_no in tqdm(list(range(num_epochs)), desc='epoch'):
            for idx in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                _, sentences, _, img_ids = batch
                
                segmentations = self.img_loader.load_segmentations(img_ids)
                attention_gold_standards, attention_masks = self.create_attention_gold_standards(sentences, segmentations)

                if self.train_cnn:
                    # Train CNN and RNN 
                    feed_dict = self.get_feed_dict(batch, is_train=True)
                    _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict)

                else:              
                    # Train RNN only 
                    img_files, _, _, img_ids = batch
                    imgs = self.img_loader.load_imgs(img_files)
                    
                    feed_dict = self.get_feed_dict(batch, is_train=True) #------------ADDED
                    
                    if self.init_lstm_with_fc_feats:
                        contexts, feats = sess.run([self.conv_feats, self.fc_feats], feed_dict={self.imgs:imgs, self.is_train:False})
                        feed_dict = self.get_feed_dict(batch, is_train=True, contexts=contexts, feats=feats)
                    else:
                        contexts = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})
                        feed_dict = self.get_feed_dict(batch, is_train=True, contexts=contexts)

                    _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict)

                print(" Loss0=%f Loss1=%f" %(loss0, loss1))

                if (global_step + 1) % params.save_period == 0:
                    self.save(sess)

            train_data.reset()

        self.save(sess)

        print("Training complete.")

    def val(self, sess, val_coco, val_data):
        """ Validate the model. """
        print("Validating the model ...")
        results = []
        result_dir = self.params.val_result_dir
        tf.get_default_graph().finalize()   # Added by Logan to ensure no more nodes are added to graph
        # Generate the captions for the images
        for k in tqdm(list(range(val_data.count))):
            batch = val_data.next_batch()
            img_files = batch
            img_file = img_files[0]
            img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]

            if self.train_cnn:
                feed_dict = self.get_feed_dict(batch, is_train=False)
            else:
                img_files = batch
                imgs = self.img_loader.load_imgs(img_files)

                if self.init_lstm_with_fc_feats:
                    contexts, feats = sess.run([self.conv_feats, self.fc_feats], feed_dict={self.imgs:imgs, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts, feats=feats)
                else:
                    contexts = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts)

            result = sess.run(self.results, feed_dict=feed_dict)
            sentence = self.word_table.indices_to_sent(result.squeeze())
            results.append({'image_id': val_data.img_ids[k], 'caption': sentence})

            # Save the result in an image file
            img = mpimg.imread(img_file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(sentence)
            plt.savefig(os.path.join(result_dir, img_name+'_result.jpg'))
            plt.cla()

        val_data.reset() 

        # Evaluate these captions
        val_res_coco = val_coco.loadRes2(results)
        scorer = COCOEvalCap(val_coco, val_res_coco)
        scorer.evaluate()
        print("Validation complete.")

    def test(self, sess, test_data, show_result=False):
        """ Test the model. """
        print("Testing the model ...")
        result_file = self.params.test_result_file
        result_dir = self.params.test_result_dir
        captions = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.count))):
            batch = test_data.next_batch()
            img_files = batch
            img_file = img_files[0]
            img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]

            if self.train_cnn:
                feed_dict = self.get_feed_dict(batch, is_train=False)
            else:
                imgs = self.img_loader.load_imgs(img_files)

                if self.init_lstm_with_fc_feats:
                    contexts, feats = sess.run([self.conv_feats, self.fc_feats], feed_dict={self.imgs:imgs, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts, feats=feats)
                else:
                    contexts = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})
                    feed_dict = self.get_feed_dict(batch, is_train=False, contexts=contexts)

            result = sess.run(self.results, feed_dict=feed_dict)
            sentence = self.word_table.indices_to_sent(result.squeeze())
            captions.append(sentence)
        
            # Save the result in an image file
            img = mpimg.imread(img_file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(sentence)
            plt.savefig(os.path.join(result_dir, img_name+'_result.jpg'))

        # Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.img_files, 'caption':captions})
        results.to_csv(result_file)
        print("Testing complete.")

    def save(self, sess):
        """ Save the model. """
        print(("Saving model to %s" % self.save_dir))
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        """ Load the model. """
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

    def load2(self, data_path, session, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading CNN model from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        miss_count = 0
        print("Trainable_variables")
        for var in tf.trainable_variables():
            print var.name    
        for op_name in data_dict:
            datas = data_dict[op_name]
            param_names = [op_name + '/' + s + ':0' for s in ['weights', 'biases']]
            for i in range(2):
                param_name = param_names[i]
                data = datas[i]
                if 'fc8' in param_name: continue
                try:
                    var = self.get_variable_by_name(param_name)
                    session.run(var.assign(data))
                    count += 1
                   #print("Variable %s:%s loaded" %(op_name, param_name))
                except ValueError:
                    miss_count += 1
                   #print("Variable %s:%s missed" %(op_name, param_name))
                    if not ignore_missing:
                        raise
        print("%d variables loaded. %d variables missed." %(count, miss_count))
        
'''
--------------------------------

Added by Logan

--------------------------------
'''
    def get_variable_by_name(self, name):
        ''' Gets a variable from the current TensorFlow graph '''
        list = [v for v in tf.global_variables() if v.name == name]
        if len(list) <= 0:
            raise Exception('No variable found by name: ' + name)
        if len(list) > 1:
            raise Exception('Multiple variables found by name: ' + name)
        return list[0] 


if __name__=="__main__":
    import main
    main.main(sys.argv)

