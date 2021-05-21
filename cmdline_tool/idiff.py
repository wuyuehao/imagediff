import logging
from category_match import img_classification, img_similarity
import os
from pixel_diff import simple_pixel_diff
from feature_match import feature_matching, feature_diff

def similarity(f1, f2):
    f1_name = f1[f1.rindex('/')+1:f1.rindex('.')]
    f2_name = f2[f2.rindex('/')+1:f2.rindex('.')]
    id = f1_name + "_vs_" + f2_name
    pixel_match_file = f"{id}_PM_.png"
    feature_match_file = f"{id}_FM_.png"
    feature_diff_file1 =  f"{id}_FD1_.png"
    feature_diff_file2 = f"{id}_FD2_.png"
    catagory_match_file = f"{id}_CM_.png"
    logging.info('comparing %s with %s', f1, f2)
    pd = simple_pixel_diff(f1, f2, pixel_match_file)
    logging.info('pixel matching score: %f', pd)
    feature_match_score = feature_matching(f1, f2, feature_match_file)
    logging.info('feature matching score: %d', feature_match_score)

    feature_diff_score = feature_diff(f1, f2, feature_diff_file1, feature_diff_file2)

    category1 = img_classification(f1)
    category2 = img_classification(f2)

    euclidean_distance, cosine_distance = img_similarity(f1, f2)

    return {'f1': f1_name, 'f2': f2_name, 'success': True, 'pixel_match_score': pd,
            'pixel_match_img': "PM_" + id, 'feature_match_score': feature_match_score,
            'feature_match_img': "FM_" + id, 'euclidean_distance': euclidean_distance,
            'cosine_distance': cosine_distance,
            'category_match_img': "CM_" + id, 'feature_diff_score': feature_diff_score,
            'feature_diff_img1': "FD1_" + id, 'feature_diff_img2': "FD2_" + id, "category1": category1,
            "category2": category2}


import sys
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python idiff.py [filename1] [filename2]")
    else:
        result = similarity(sys.argv[1], sys.argv[2])
        print(result)