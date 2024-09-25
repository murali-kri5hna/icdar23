import os.path
from .regex import ImageFolder

class WriterZoo:

    @staticmethod
    def new(desc, **kwargs):
        return ImageFolder(desc['path'], regex=desc['regex'], **kwargs)

    @staticmethod
    def get(dataset, set, **kwargs):
        _all = WriterZoo.datasets
        d = _all[dataset]
        s = d['set'][set]

        s['path'] = os.path.join(d['basepath'], s['path'])
        return WriterZoo.new(s, **kwargs)

    datasets = {

        'icdar2017': {
            'basepath': '/cluster/qy41tewa/rl-map/dataset',
            #'basepath': '/scratch/qy41tewa/rl-map/dataset',
            'set': {
                'test' :  {'path': 'test/icdar2017_test_sift_patches_binarized_2kpp',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+-IMG_MAX_(\d+)'}},

                'train' :  {'path': 'train/icdar2017_train_sift_patches_binarized_5000',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+-IMG_MAX_(\d+)_\d+'}},

                'test_debug' :  {'path': 'test_debug/icdar2017_test_sift_patches_binarized_2kpp',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+-IMG_MAX_(\d+)'}}, 

                'train_classify' :  {'path': 'classify/train',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+-IMG_MAX_(\d+)_\d+'}},

                'val_classify' :  {'path': 'classify/val',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+-IMG_MAX_(\d+)_\d+'}},
                
            }
        },

        'icdar2013': {
            'basepath': '/cluster/qy41tewa/rl-map/dataset/icdar19/binarized',
            'set': {
                'test' :  {'path': 'test',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'train_icdar19',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}}                             
            }
        },
        
        'icdar2019': {
            'basepath': '/cluster/qy41tewa/rl-map/dataset/icdar19/binarized',
            'set': {
                'test' :  {'path': 'test_icdar19',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'train_icdar19',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}},
            }
        },

        'icdar2019_color': {
            'basepath': '/cluster/qy41tewa/rl-map/dataset/icdar19/color',
            'set': {
                'test' :  {'path': 'icdar19_test_patches',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'icdar19_train_2000patches',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}},
            }
        }
    }
