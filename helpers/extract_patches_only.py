import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
import logging
import sklearn.preprocessing
from sklearn.decomposition import PCA
from tqdm import tqdm


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def black_pixels(img):
    return np.sum(img == 0)
def white_pixels(img):
    return np.sum(img==255)


def calc_features(input_img, arguments):
    img = cv2.imread(input_img)

    if arguments.scale != -1:
        scale = arguments.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray.copy()

    sift = cv2.SIFT_create(sigma=args.sigma)
    kp = sift.detect(img_gray, None)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    def filter_keypoint(point):
        # if 3 < point.size < 10:
        #     return point
        return point


    new_kp = []
    uniq_kp = []

    num_pixel = arguments.win_size ** 2 

    for p in kp:
        point = (int(p.pt[0]), int(p.pt[1]))

        roi = get_image_patch(img, point[0], point[1], arguments)

        if roi is None:
            continue

        if arguments.centered:
            if img_bin[point[1], point[0]] != 0:
                continue

        if arguments.black_pixel_thresh != -1:
            if black_pixels(roi) > (arguments.black_pixel_thresh * num_pixel):
                continue 

        if arguments.white_pixel_thresh != -1:
            if white_pixels(roi) > (arguments.white_pixel_thresh * num_pixel):
                continue


        if not point in uniq_kp and filter_keypoint(p):
            new_kp.append(p)
            uniq_kp.append(point)

    _, desc = sift.compute(img_gray, new_kp)

    desc = sklearn.preprocessing.normalize(desc, norm='l1')
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    desc = sklearn.preprocessing.normalize(desc, norm='l2')

    print(f'{len(new_kp)} out of {len(kp)} for {os.path.splitext(os.path.basename(input_img))[0]}')
    kp_tupple = [(p, desc[i]) for i, p in enumerate(uniq_kp)]


    return kp_tupple


def get_image_patch(img, px, py, arguments):
    half_win_size = int(arguments.win_size / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        # if px - half_win_size < 0 or px + half_win_size > img.shape[1] or py - half_win_size < 0 or py + half_win_size > img.shape[0]:
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                (half_win_size * 2, half_win_size * 2,
                                                                 roi.shape[0], roi.shape[1])
    return roi


def extract_patches(filename, tup, args):
    if len(tup) > args.patches_per_page and args.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, args.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]

    points, _ = zip(*tup)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.scale != -1:
        scale = args.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)


    count = 0
    for p in points:
        roi = get_image_patch(img, p[0], p[1], args)
        if roi is None:
            continue
        
        out_path = os.path.join(args.out_dir[0], os.path.splitext(os.path.basename(filename))[0],
                                os.path.splitext(os.path.basename(filename))[0] + '_' + str(count) + '.png')

        count = count + 1
        cv2.imwrite(out_path, roi)


if __name__ == "__main__":
    import argparse
    import pickle

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    parser = argparse.ArgumentParser(description="extract patches from images")
    parser.add_argument('--in_dir', metavar='in_dir', dest='in_dir', type=str, nargs=1,
                        help='input directory', required=True)
    parser.add_argument('--out_dir', metavar='out_dir', dest='out_dir', type=str, nargs=1,
                        help='output directory', required=True)
    parser.add_argument('--win_size', metavar='win_size', dest='win_size', type=int, nargs='?',
                        help='size of the patch',
                        default=32)
    parser.add_argument('--patches_per_page', metavar='patches_per_page', dest='patches_per_page', type=int, nargs='?',
                        help='maximal number of patches per page (-1 for no limit)',
                        default=-1)
    parser.add_argument('--scale', type=float, help='scale images up or down',
                        default=-1)
    parser.add_argument('--sigma', type=float, help='blur factor for SIFT',
                        default=1.6)
    parser.add_argument('--black_pixel_thresh', type=float, help='if more black_pixel_thresh percent of the pixels are black -> discard',
                        default=0.5)
    parser.add_argument('--white_pixel_thresh', type=float, help='if more than white_pixel_thresh percent of the pixels are white -> discard',
                        default=0.5)
    parser.add_argument('--centered', type=bool, help='filter patches whose keypoints are not located on handwriting, only for binarized datasets',
                    default=True)

    
    args = parser.parse_args()


    assert os.path.exists(args.in_dir[0]), 'in_dir {} does not exist'.format(args.in_dir[0])


    if not os.path.exists(args.out_dir[0]):
        logging.info('creating directory %s' % args.out_dir[0])
        os.mkdir(args.out_dir[0])

    assert len(os.listdir(args.out_dir[0])) == 0, 'out_dir is not empty'

    assert args.win_size % 2 == 0, 'win_size must be even'

    num_cores = int(multiprocessing.cpu_count() / 2 + 2)
   # num_cores = multiprocessing.cpu_count()
    path_to_centers = ''

    files = [f for f in glob.glob(args.in_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]

    assert len(files) > 0, 'no images found'
    logging.info('Found {} images'.format(len(files)))

    logging.info('calculating features for images in %s (number of cores:%d)' % (args.in_dir, num_cores))
    results = []
    results = Parallel(n_jobs=num_cores, verbose=9)(delayed(calc_features)(f, args) for f in files)

    logging.info('collecting descriptors')
    desc_list = []
    kp_list = []
    fn_list = []

    for r, f in tqdm(zip(results, files)):
        if len(r) == 0:
            logging.warning('no keypoints found in file {} '.format(f))
        for kp, desc in r:
            kp_list.append(kp)
            desc_list.append(desc)
            fn_list.append(f)

    # filtered = []
    # if args.cluster:
    #     descs = np.array(desc_list)
    #     samples = min(descs.shape[0], 500000)

    #     pca = PCA(32, whiten=True)
    #     pca.fit(descs[np.linspace(0, descs.shape[0]-1, samples).astype('int')])

    #     center_path = '/data/mpeer/resources/icdar2017_train_sift_patches_binarized/centers.pkl' #args.cluster
    #     kmeans = pickle.load(open(center_path,'rb'))
    #     patches_files = {}
    #     feature_count = 0
    #     batch_size = 50000
    #     for r, f in zip(results, files):
    #         kp, desc = zip(*r)
                
    #         desc = np.array(desc)
    #         desc = pca.transform(desc)
    #         dist = kmeans.transform(desc)
    #         prediction = kmeans.predict(desc)

    #         dist = np.sort(dist)
    #         ratio = dist[:, 0] / dist[:, 1]
    #         idx = (ratio > 0.9).nonzero()[0]
    #         print(f'remove {idx.shape} keypoints')
    #         kp = np.delete(kp, idx).tolist()
    #         desc =  np.delete(desc, idx).tolist()

    #         filtered.append([(k,d) for k,d in zip(kp, desc)])
    #             # for p, f, c, r in zip(batch(kp_list), batch(fn_list), prediction, ratio):
    #             #     if r <= 0.9:
    #             #         feature_count += 1
    #             #         if f in patches_files:
    #             #             patches_files[f].append((p, c))
    #             #         else:
    #             #             patches_files[f] = [(p, c)]

    # descs = None
    # results = filtered
    logging.info('creating labels directories in output directory')
    for fn in set(fn_list):
        f_name = os.path.splitext(os.path.basename(fn))[0]
        if not os.path.exists(os.path.join(args.out_dir[0], str(f_name))):
            os.mkdir(os.path.join(args.out_dir[0], str(f_name)))

    # logging.info('copying %i (all patches per page) image patches to %s ' % (kp_list, args.out_dir[0]))
    results = Parallel(n_jobs=num_cores, verbose=9)(
        delayed(extract_patches)(filename, tup, args) for filename, tup in zip(files, results))

    config_out_path = os.path.join(args.out_dir[0], 'db-creation-parameters.json')
    logging.info('writing config parameters to {}')
    with open(config_out_path, 'w') as f:
        import json

        json.dump(vars(args), f)

    out_files = [f for f in glob.glob(args.out_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]
    logging.info(f'done - extracted {len(out_files)} patches')