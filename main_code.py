# Try to seperate program into clear verion and useful functions

import matplotlib.pyplot as plt
import os,sys
import pydicom
import numpy as np
import cv2
import copy
import seaborn as sns
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
import csv, codecs
import traceback


def gen_ct_dicom_dict(ct_filelist):
    CtCache = {}
    CtCache["SOPInstanceUID"] = {}  # Query ct data by the key SOPInstanceUID
    CtCache["SliceLocation"] = {}  # Query ct data by the key SliceLocation
    CtCache["filepath"] = {}  # Query ct data by the key filepath

    for filepath in ct_filelist:
        ct_fp = pydicom.read_file(filepath)
        ct_SOPInstanceUID = ct_fp.SOPInstanceUID
        ct_SliceLocation = ct_fp.SliceLocation
        ct_filepath = filepath

        ct_dict = {}
        ct_dict["SOPInstanceUID"] = ct_SOPInstanceUID
        ct_dict["SliceLocation"] = ct_SliceLocation
        ct_dict["filepath"] = ct_filepath
        # Additional appending data but not key for query
        ct_dict["ImagePositionPatient_x"] = ct_fp.ImagePositionPatient[0]  # CT_origin_x
        ct_dict["ImagePositionPatient_y"] = ct_fp.ImagePositionPatient[1]  # CT_origin_y
        ct_dict["ImagePositionPatient_z"] = ct_fp.ImagePositionPatient[2]  # CT_origin_z, Same as SliceLocation
        ct_dict["PixelSpacing_x"] = ct_fp.PixelSpacing[0]  # CT_ps_x
        ct_dict["PixelSpacing_y"] = ct_fp.PixelSpacing[1]  # CT_ps_y
        ct_dict["TableHeight"] = ct_fp.TableHeight  # Table_H
        ct_dict["Columns"] = ct_fp.Columns  # CT_columns
        ct_dict["Rows"] = ct_fp.Rows  # CT_rows
        ct_dict["ROIName"] = {}
        ct_dict["pixel_array"] = copy.deepcopy(ct_fp.pixel_array)
        ct_dict["RescaleSlope"] = ct_fp.RescaleSlope
        ct_dict["RescaleIntercept"] = ct_fp.RescaleIntercept
        ct_dict["rescale_pixel_array"] = ct_fp.pixel_array * ct_fp.RescaleSlope + ct_fp.RescaleIntercept

        # Wish can get contourdata[x,y,z...] by ct_dict["ROIName"][roiName]["ContourData"]
        CtCache["SOPInstanceUID"][ct_SOPInstanceUID] = ct_dict
        CtCache["SliceLocation"][ct_SliceLocation] = ct_dict
        CtCache["filepath"][ct_filepath] = ct_dict
    return CtCache
    pass


def get_ct_filelist_by_folder(folder):
    ct_filelist = []
    for file in os.listdir(folder):
        # print(file)
        filepath = "{}\\{}".format(folder, file)
        file_exists = os.path.isfile(filepath)
        if not file_exists:
            continue
        ct_fp = None
        try:
            ct_fp = pydicom.read_file(filepath)
        except:
            # Not dicom file
            continue
        if ct_fp.Modality != 'CT':
            continue
        # print(filepath)
        ct_filelist.append(filepath)

    return ct_filelist


def get_max_contours_by_filter_img(A, filter_img, ContourRetrievalMode=cv2.RETR_EXTERNAL):
    # gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
    gray_image = filter_img
    # findContours
    # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)
    return contours


def get_max_contours(A, constant_value=None, ContourRetrievalMode=cv2.RETR_EXTERNAL):
    constant = None
    if constant_value == None:
        # Algoruthm to find constant value
        data = A.ravel()
        sorted_data = np.copy(data)
        sorted_data.sort()
        constant = sorted_data[-20] - 100
    else:
        constant = constant_value
    # The RGB format for cv2 is
    filter_img = np.zeros((A.shape[0], A.shape[1], 3), np.uint8)
    # Make filter_img be mask array
    filter_img[A <= constant] = (0, 0, 0)
    filter_img[A > constant] = (255, 255, 255)
    # convert mask array to gray image format
    gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
    # findContours
    # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)
    # return contours (list of np.array) and constant (you assume they are almsot highest)
    return (contours, constant)


# Useful
def get_rect_infos_and_center_pts(contours, h_min=13, w_min=13, h_max=19, w_max=19):
    app_center_pts = []
    rect_infos = []
    for contour in contours:
        # Step 2. make useful information
        i = contour
        con = i.reshape(i.shape[0], i.shape[2])
        x_min = con[:, 0].min()
        x_max = con[:, 0].max()
        x_mean = con[:, 0].mean()
        y_min = con[:, 1].min()
        y_max = con[:, 1].max()
        y_mean = con[:, 1].mean()
        h = y_max - y_min
        w = x_max - x_min
        x_mean = int(x_mean)
        y_mean = int(y_mean)
        rect_info = [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
        # if h >= 13 and h < 19 and w >= 13 and h < 19:
        if h >= h_min and h < h_max and w >= w_min and w < w_max:
            cen_pt = [x_mean, y_mean]
            app_center_pts.append(cen_pt)
        else:
            # print('(h={},{} , w={},{})'.format(h_max, h_min, w_max, w_min))
            # print('Not matching ! rect_info = ', rect_info)
            pass
        # print(rect_info)
        rect_infos.append(rect_info)
    sorted_app_center_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
    return (sorted_app_center_pts, rect_infos, app_center_pts)


def convert_to_gray_image(pixel_array):
    img = np.copy(pixel_array)
    # Convert to float to avoid overflow or underflow losses.
    img_2d = img.astype(float)
    # Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    # Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)
    return img_2d_scaled


# DISCUSS_LATTER
def get_2level_max_contours(img, gray_img):
    def get_max_contours_by_filter_img(A, filter_img, ContourRetrievalMode=cv2.RETR_TREE):
        # gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
        gray_image = filter_img
        # findContours
        # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)
        return contours

    def get_max_contours(A, constant_value=None, ContourRetrievalMode=cv2.RETR_TREE):
        constant = None
        if constant_value == None:
            # Algoruthm to find constant value
            data = A.ravel()
            sorted_data = np.copy(data)
            sorted_data.sort()
            constant = sorted_data[-20] - 100

        else:
            constant = constant_value
        # The RGB format for cv2 is
        filter_img = np.zeros((A.shape[0], A.shape[1], 3), np.uint8)
        # Make filter_img be mask array
        filter_img[A <= constant] = (0, 0, 0)
        filter_img[A > constant] = (255, 255, 255)
        # convert mask array to gray image format
        gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
        # findContours
        # RETR_TREE will show the contour included in contour
        # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)

        # RETR_EXTERNAL will NOT show the contour inclued in contour
        # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # return contours (list of np.array) and constant (you assume they are almsot highest)
        return (contours, constant)

    def get_minimum_rect_from_contours(contours, padding=2):
        rect = (x_min, x_max, y_min, y_max) = (0, 0, 0, 0)
        is_first = True
        for contour in contours:
            reshaped_contour = contour.reshape(contour.shape[0], contour.shape[2])
            for pt in reshaped_contour:
                x = pt[0]
                y = pt[1]
                if is_first == True:
                    x_min = x
                    x_max = x
                    y_min = y
                    y_max = y
                    is_first = False
                else:
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        rect = (x_min, x_max, y_min, y_max)
        return rect

    def is_point_in_rect(pt, rect=(0, 0, 0, 0)):
        (x_min, x_max, y_min, y_max) = rect
        x = pt[0]
        y = pt[1]
        if x >= x_min and x < x_max and y >= y_min and y < y_max:
            return True
        else:
            return False

    def is_contour_in_rect(contour, rect=(0, 0, 0, 0)):
        (x_min, x_max, y_min, y_max) = rect
        isContourInRect = True
        reshaped_contour = contour.reshape(contour.shape[0], contour.shape[2])
        for pt in reshaped_contour:
            if False == is_point_in_rect(pt, rect):
                isContourInRect = False
                break
        return isContourInRect

    def is_just_found_3_contained_contour(contours):
        # contained_countour is mean a contour which continaed by some large contour
        contained_contour_num = 0
        contained_contours = []
        for idx, contour in enumerate(contours):
            other_contours = copy.deepcopy(contours)
            other_contours.pop(idx)  # remove the contour in other_contours
            is_contour_to_be_contained = False
            for large_contour in other_contours:
                rect = (min_x, max_x, min_y, max_y) = get_minimum_rect_from_contours([large_contour], padding=0)
                if is_contour_in_rect(contour, rect):
                    is_contour_to_be_contained = True
                    contained_contours.append(contour)
                    break
            if is_contour_to_be_contained == True:
                contained_contour_num = contained_contour_num + 1
        if contained_contour_num == 3:
            return (True, contained_contours)
        else:
            return (False, None)

    # When two applicator touch together,
    # inner contour case of level1_contour is better one case of then level2_contour

    # When middle applicator cutting,
    # inner tontour case of level2_contour is better one case of then level1_contour

    # So, how to decide which case is for use between these two cases both touching and middle cutting?
    # and then we can process it easily after decision.

    (level1_contours, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_TREE)
    print('shape of level1_contours[0] = ', level1_contours[0].shape)
    is_just_found_3_contained_contour(level1_contours)
    (is_found, contained_contours) = is_just_found_3_contained_contour(level1_contours)
    if is_found:
        print('This is case of two applicator touched together (maybe)')
        return contained_contours

    # the remain case is process the middle-cut problem

    (x_min, x_max, y_min, y_max) = get_minimum_rect_from_contours(level1_contours, padding=2)

    threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    filter_img = threshed_im
    level2_contours = get_max_contours_by_filter_img(img, filter_img)

    # level1_contours make a rectangle scope. Then select leve2 contour which is in this rectangle scope.
    filtered_level2_contours = []
    for contour in level2_contours:
        if True == is_contour_in_rect(contour, rect=(x_min, x_max, y_min, y_max)):
            filtered_level2_contours.append(contour)

    (is_found, contained_contours) = is_just_found_3_contained_contour(filtered_level2_contours)
    if is_found == True:
        return contained_contours
    else:
        print('we cannot found our expected case, please see data to process different case')
        return None

    return filtered_level2_contours


# WHAT_IS_SPECIAL_CASE ?
def get_contours_of_first_slice_in_special_case(first_slice_dict):
    def convert_to_gray_image(pixel_array):
        img = np.copy(pixel_array)
        # Convert to float to avoid overflow or underflow losses.
        img_2d = img.astype(float)
        # Rescaling grey scale between 0-255
        img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
        # Convert to uint
        img_2d_scaled = np.uint8(img_2d_scaled)
        return img_2d_scaled

    def get_gray_img_of_slice_dict(slice_dict):
        img = slice_dict['rescale_pixel_array']
        gray_img = convert_to_gray_image(img)
        return gray_img

    img = first_slice_dict['rescale_pixel_array']
    gray_img = get_gray_img_of_slice_dict(first_slice_dict)
    contours = get_2level_max_contours(img, gray_img)
    return contours

    pass


def get_app_center_pts_of_first_slice(first_slice_dict):
    ps_x = first_slice_dict['PixelSpacing_x']
    ps_y = first_slice_dict['PixelSpacing_y']
    h_max = int((19.0 * 4.19921e-1) / ps_y)
    h_min = int((13.0 * 4.19921e-1) / ps_y)
    w_max = int((19.0 * 4.19921e-1) / ps_x)
    w_min = int((13.0 * 4.19921e-1) / ps_x)
    # print('(h={},{} , w={},{})'.format(h_max, h_min, w_max, w_min))

    (contours, constant) = get_max_contours(first_slice_dict['rescale_pixel_array'])

    # (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
    (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours, h_max=h_max,
                                                                                        h_min=h_min, w_max=w_max,
                                                                                        w_min=w_min)
    print('\n\n')
    print(sorted_app_center_pts)
    # TODO After researching done, write the code to finish this task
    if sorted_app_center_pts == None or len(sorted_app_center_pts) != 3:
        contours = get_contours_of_first_slice_in_special_case(first_slice_dict)
        if len(contours) != 3:
            print('Error process for special case of first slice')
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours, h_max=h_max,
                                                                                            h_min=0, w_max=w_max,
                                                                                            w_min=0)
        x_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
        return x_sorted_pts
        pass
    print('\n\n')

    x_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
    print('get_app_center_pts_of_first_slice() -> x_sorted_pts = ', x_sorted_pts)
    return x_sorted_pts
    pass



# Useful in code
def get_view_scope_by_slice(first_slice_dict, padding=30):
    (contours, constant) = get_max_contours(first_slice_dict['rescale_pixel_array'])
    print('PixelSpacing_(x,y)=({}, {})'.format(first_slice_dict['PixelSpacing_x'], first_slice_dict['PixelSpacing_y']))
    ps_x = first_slice_dict['PixelSpacing_x']
    ps_y = first_slice_dict['PixelSpacing_y']
    h_max = int((19.0 * 4.19921e-1) / ps_y)
    h_min = int((13.0 * 4.19921e-1) / ps_y)
    w_max = int((19.0 * 4.19921e-1) / ps_x)
    w_min = int((13.0 * 4.19921e-1) / ps_x)
    print('(h={},{} , w={},{})'.format(h_max, h_min, w_max, w_min))

    # (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
    (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours, h_max=h_max,
                                                                                        w_max=w_max, h_min=h_min,
                                                                                        w_min=w_min)
    print('sorted_app_center_pts = ', sorted_app_center_pts)
    if sorted_app_center_pts == None or len(sorted_app_center_pts) != 3:
        contours = get_contours_of_first_slice_in_special_case(first_slice_dict)
        if len(contours) != 3:
            print('Error process for special case of first slice')
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours, h_max=h_max,
                                                                                            h_min=0, w_max=w_max,
                                                                                            w_min=0)
        padding = padding + 30

    x_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
    y_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[1], reverse=False)

    (min_x, max_x) = (x_sorted_pts[0][0], x_sorted_pts[-1][0])
    (min_y, max_y) = (y_sorted_pts[0][1], y_sorted_pts[-1][1])
    print('X:({}, {})'.format(min_x, max_x))
    print('Y:({}, {})'.format(min_y, max_y))
    # Print information
    # [[228, 324], [264, 334], [299, 347]]
    # X:(228, 299)
    # Y:(324, 347)
    w = max_x - min_x
    h = max_y - min_y
    dist = np.max([w, h])
    cen_x = int(min_x + (w) / 2)
    cen_y = int(min_y + (h) / 2)
    loc_min_x = int(cen_x - dist / 2)
    loc_max_x = int(cen_x + dist / 2)
    loc_min_y = int(cen_y - dist / 2)
    loc_max_y = int(cen_y + dist / 2)
    print('loc X:({}, {})'.format(loc_min_x, loc_max_x))
    print('loc Y:({}, {})'.format(loc_min_y, loc_max_y))
    # padding = 30 #Suitable value for no center point?

    # loc X:(227, 298)
    # loc Y:(299, 370)
    # And We have gray_img = gray_img[270:390, 200:320]
    # So it should be
    # gray_img = gray_img[ loc_min_y-padding: loc_max_y+padding, loc_min_x-padding:loc_max_x+padding]
    view_min_y = loc_min_y - padding
    view_max_y = loc_max_y + padding
    view_min_x = loc_min_x - padding
    view_max_x = loc_max_x + padding
    return (view_min_y, view_max_y, view_min_x, view_max_x)
    # The way to use return value is
    # gray_img = gray_img[ view_min_y: view_max_y, view_min_x:view_max_x]




# Useful
def distance(pt1, pt2):
    import math
    axis_num = len(pt1)
    sum = 0.0
    for idx in range(axis_num):
        sum = sum + (pt1[idx] - pt2[idx]) ** 2
    ans = math.sqrt(sum)
    return ans


# Useful
def get_most_closed_pt(src_pt, pts, allowed_distance=100):
    if pts == None:
        return None
    if pts == []:
        return None
    dst_pt = None
    for pt in pts:
        if distance(src_pt, pt) > allowed_distance:
            # the point , whoes distance with src_pt < allowed_distance, cannot join this loop
            continue

        if dst_pt == None:
            dst_pt = pt
        else:
            if distance(src_pt, pt) < distance(src_pt, dst_pt):
                dst_pt = pt
        pass
    return dst_pt

def make_lines_process(app_pts):
    lines = [[], [], []]
    sorted_app_pts_keys = sorted(app_pts.keys())
    print(sorted_app_pts_keys)

    for key_idx in range(len(sorted_app_pts_keys)):
        key = sorted_app_pts_keys[key_idx]
        pts = app_pts[key]

        if key_idx == 0:
            lines[0].append(pts[0])
            lines[1].append(pts[1])
            lines[2].append(pts[2])
        else:
            for line in lines:
                last_line_pt = line[-1]
                if last_line_pt == None:
                    continue
                last_line_pt_x = last_line_pt[0]

                candidate_pt = None
                # looking forward for candidate_pt
                for pt_idx in range(len(pts)):
                    pt = pts[pt_idx]
                    pt_x = pt[0]
                    ##if abs(last_line_pt_x - pt_x) < 5
                    # if abs(last_line_pt_x - pt_x) < 5 or (lines[0][-1] == None and lines[2][-1] == None):
                    if abs(last_line_pt_x - pt_x) < 10:
                        if candidate_pt == None:
                            candidate_pt = pt
                        else:
                            candidate_pt_x = candidate_pt[0]
                            if abs(candidate_pt_x - last_line_pt_x) > abs(pt_x - last_line_pt_x):
                                candidate_pt = last_line_pt
                line.append(candidate_pt)
                # the data structure fo each line will be like
                # [(x0,y0,z0), (x1,y1,z1), ... ,(xn,yn,zn),None]

    # clean dummy None in last element in each line
    for idx in range(len(lines)):
        line = lines[idx]
        line = line[:-1]
        lines[idx] = line
    return lines

def algo_run_by_folder(folder):
    # app_pts_dict[z] = [[x,y,z], [x,y,z], [x,y,z] ]
    app_pts_dict = {}
    ct_filelist = get_ct_filelist_by_folder(folder)
    ct_dicom_dict = gen_ct_dicom_dict(ct_filelist)
    sorted_ct_dicom_dict_keys = sorted(ct_dicom_dict['SliceLocation'].keys())
    first_slice_dict = ct_dicom_dict['SliceLocation'][sorted_ct_dicom_dict_keys[0]]
    based_center_pts = get_app_center_pts_of_first_slice(first_slice_dict)
    if len(based_center_pts) != 3:
        print('len(based_center_pts) is wrong, folder = ', folder)
        return
    else:
        print(based_center_pts)

    first_slice_dict['data'] = {}
    first_slice_dict['data']['center_pts'] = based_center_pts
    (view_min_y, view_max_y, view_min_x, view_max_x) = get_view_scope_by_slice(first_slice_dict, padding=100)

    prev_slice_dict = None
    for z in sorted_ct_dicom_dict_keys:
        app_pts_dict[z] = []
        slice_dict = ct_dicom_dict['SliceLocation'][z]
        if 'data' not in slice_dict.keys():
            slice_dict['data'] = {}
        slice_dict['data']['prev_slice_dict'] = prev_slice_dict
        print('z = ', z, 'filepath = ', slice_dict['filepath'])

        if 'data' in slice_dict and 'center_pts' in slice_dict['data']:
            prev_slice_dict = slice_dict
            # pure_show_slice_dict(slice_dict, (view_min_y, view_max_y, view_min_x, view_max_x))
            # First slice
            print('center_pts = ', slice_dict['data']['center_pts'])
            for pt in slice_dict['data']['center_pts']:
                x = pt[0]
                y = pt[1]
                app_pts_dict[z].append([x, y, z])
            continue

        img = slice_dict['rescale_pixel_array']
        gray_img = convert_to_gray_image(img)
        # fig = plt.figure(figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')
        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
        gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
        img = img[view_min_y: view_max_y, view_min_x:view_max_x]

        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -22)
        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
        threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
        # I'm not sure why it is the perfect value in our case.

        # plt.subplot(1, 4, 1)
        # plt.imshow(threshed_im, cmap=plt.cm.bone)
        # plt.subplot(1, 4, 2)
        # plt.imshow(img, cmap=plt.cm.bone)
        # plt.subplot(1, 4, 3)
        # plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

        filter_img = threshed_im
        # contours = get_max_contours_by_filter_img(img, filter_img)

        # contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_TREE)
        # (contours_without_filter,constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_TREE)
        # proc_img = np.copy(img)
        # contours.extend(contours_without_filter)

        contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_TREE)

        (contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_TREE)
        contours.extend(contours_without_filter)

        (contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_EXTERNAL)
        contours.extend(contours_without_filter)

        the_contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_EXTERNAL)
        contours.extend(the_contours)

        proc_img = np.copy(img)
        contours.extend(contours_without_filter)

        ellipse_center_pts = []
        draw_ellipse_center_pts = []
        for contour in contours:
            if len(contour) < 5:
                # You need at least 5 points in contour, so that you can use fitEllipse

                reshape_contour = contour.reshape(contour.shape[0], contour.shape[2])
                xs = [pt[0] for pt in reshape_contour]
                ys = [pt[1] for pt in reshape_contour]
                x = int((min(xs) + max(xs)) / 2)
                y = int((min(ys) + max(ys)) / 2)
                # enablePrint()
                # print("special fitEllipse(x,y) = ({},{})".format(x,y))
                # blockPrint()
                ellipse_center_pts.append([x, y])
                continue
            ellipse = cv2.fitEllipse(contour)  # auto-figure the ellipse to fit contour
            # print(ellipse)
            ellipse_poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                            (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360,
                                            5)
            draw_x = int(ellipse[0][0])
            draw_y = int(ellipse[0][1])
            draw_ellipse_center_pts.append([draw_x, draw_y])
            x = int(ellipse[0][0]) + view_min_x
            y = int(ellipse[0][1]) + view_min_y
            ellipse_center_pts.append([x, y])
            reshape_poly = ellipse_poly.reshape(ellipse_poly.shape[0], 1, ellipse_poly.shape[1])
            cv2.drawContours(proc_img, reshape_poly, -1, (255, 0, 0), 1)
            # cv2.line(proc_img,(draw_x,draw_y),(draw_x,draw_y),(255,0,0),3)

        figure_center_pts = []
        for pt in prev_slice_dict['data']['center_pts']:
            if len(prev_slice_dict['data']['center_pts']) == 1:
                eval_pt = pt
                pp_slice_dict = prev_slice_dict['data']['prev_slice_dict']
                # pp_slice_dict is prev_prev_slice_dict
                # if prev_prev_slice_dict != None and len(prev_prev_slice_dict['data']['center_pts'])==1:
                if pp_slice_dict != None and len(pp_slice_dict['data']['center_pts']) == 1:
                    # prev_prev_pt = prev_prev_slice_dict['data']['center_pts'][0]
                    pp_pt = pp_slice_dict['data']['center_pts'][0]
                    prev_pt = prev_slice_dict['data']['center_pts'][0]
                    eval_pt[0] = eval_pt[0] + (prev_pt[0] - pp_pt[0])
                    eval_pt[1] = eval_pt[1] + (prev_pt[1] - pp_pt[1])
                    print('update eval_pt = {}', eval_pt)
                    ps_x = slice_dict['PixelSpacing_x']
                    ps_y = slice_dict['PixelSpacing_y']
                    # print("(ps_x, ps_y) = ({},{})".format(ps_x,ps_y))
                    a_distance_mm = 15.0
                    a_distance = int(a_distance_mm / ps_x)
                dst_pt = get_most_closed_pt(eval_pt, ellipse_center_pts, allowed_distance=a_distance)
            else:
                ps_x = slice_dict['PixelSpacing_x']
                ps_y = slice_dict['PixelSpacing_y']
                # print("(ps_x, ps_y) = ({},{})".format(ps_x,ps_y))
                a_distance_mm = 10.0
                a_distance = int(a_distance_mm / ps_x)
                dst_pt = get_most_closed_pt(pt, ellipse_center_pts, allowed_distance=a_distance)
            if dst_pt != None:
                print('dst_pt != None with dst_pt = ({},{})'.format(dst_pt[0], dst_pt[1]))
                figure_center_pts.append(dst_pt)
            else:
                print('dst_pt == None with pt = ({},{})'.format(pt[0], pt[1]))

        if 'data' not in slice_dict.keys():
            slice_dict['data'] = {}
        slice_dict['data']['center_pts'] = figure_center_pts

        print('ellipse_center_pts = ', ellipse_center_pts)
        print('center_pts = ', slice_dict['data']['center_pts'])

        # plt.subplot(1, 4, 4)
        for [x, y] in figure_center_pts:
            app_pts_dict[z].append([x, y, z])
            draw_x = x - view_min_x
            draw_y = y - view_min_y
            cv2.line(proc_img, (draw_x, draw_y), (draw_x, draw_y), (255, 0, 0), 3)

        # plt.subplot(1, 4, 4)
        # plt.imshow(proc_img, cmap=plt.cm.bone)
        # plt.show()
        prev_slice_dict = slice_dict
    print(app_pts_dict)
    return app_pts_dict



# Implementation of get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)

# REWRITE get_metric_pt_info_by_travel_distance, so the get_metric_pt, reduct_distance_step and get_metric_pt_info_travel_distance will not be USED
def get_metric_pt(metric_line, pt_idx, pt_idx_remainder):
    # print('get_metric_pt(metric_line={}, pt_idx={}, pt_idx_remainder={})'.format(metric_line, pt_idx, pt_idx_remainder))
    pt = metric_line[pt_idx].copy()
    try:
        if (pt_idx + 1 >= len(metric_line)):
            end_pt = metric_line[pt_idx]
        else:
            end_pt = metric_line[pt_idx + 1]


    except Exception as e:
        print('EEEEEE')
        print('pt_idx = {}'.format(pt_idx))
        print('pt_idx_remainder = {}'.format(pt_idx_remainder))
        print('metric_line[{}] = {}'.format(pt_idx, metric_line[pt_idx]))
        raise

    for axis_idx in range(3):
        # diff = end_pt[axis_idx] - pt[axis_idx]
        # diff_with_ratio = diff * pt_idx_remainder
        # print('axis_idx = {} ->  diff_with_ratio = {}'.format(axis_idx, diff_with_ratio) )
        pt[axis_idx] += ((end_pt[axis_idx] - pt[axis_idx]) * pt_idx_remainder)
        # pt[axis_idx] = pt[axis_idx] + diff_with_ratio
    return pt

def reduce_distance_step(metric_line, pt_idx, pt_idx_remainder, dist):
    # reduce dist and move further more step for (pt_idx, pt_idx_remainder)
    # ret_dist = ??  reduce dist into ret_dist
    # Just implement code here , so that the data move a little distance. (mean reduce dist and move more)
    def distance(pt1, pt2):
        import math
        # print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2)
        return ret_dist

    start_pt_idx = pt_idx
    start_pt_idx_remainder = pt_idx_remainder
    start_pt = get_metric_pt(metric_line, start_pt_idx, start_pt_idx_remainder)

    # To figure out what distance we perfer to reduce in this step
    # And the idea is seperate int to two case
    if start_pt_idx < len(metric_line) - 1:
        # CASE: there is a next pt_idx for start_pt_idx
        # In this case, we let end_pt_idx be the next pt_idx of start_pt_idx
        # So it start_pt_idx +1. and don't forget to reset remainder in to zero
        end_pt_idx = start_pt_idx + 1
        end_pt_idx_remainder = 0

    else:
        # CASE there is no any next _pt_idx for start_pt_idx
        # In this case, we let end_pt_idx point to end idx of line and let remainder be in maximum value (1.0)
        # So the end_pt with idx and remainder can represent the most far point in the metric line.
        end_pt_idx = start_pt_idx
        end_pt_idx_remainder = 1

    end_pt = get_metric_pt(metric_line, end_pt_idx, end_pt_idx_remainder)
    max_reducable_dist = distance(start_pt, end_pt)  # max_reducable_dist in this iteration

    # We have start_pt and end_pt , and we have the dist value
    # So we can try to walk from start_pt to some point which belong to [start_pt, end_pt)
    # There are two cases for this walking
    # CASE 1: the end_pt is not enough to walking dist , so just walking to the end_pt
    # CASE 2: the end_pt is enough and we just need to figure where to stop between [start_pt, end_pt)
    # PS: 'is enough' is mean distance will be reduced to zero, so the end of travel is in  [start_pt, end_pt)
    if dist > max_reducable_dist:  # CASE 1 the end_pt is not enough to walking dist
        dist_after_walking = dist - max_reducable_dist
        walking_stop_pt_idx = end_pt_idx
        walking_stop_pt_idx_remainder = end_pt_idx_remainder
        # return (dist, end_pt_idx, end_pt_idx_remainder)
        return (dist_after_walking, walking_stop_pt_idx, walking_stop_pt_idx_remainder)
    else:  # CASE 2 the end_pt is enough, so walking_stop_pt is between [start_pt, end_pt)
        walking_stop_pt_idx = start_pt_idx

        # Figure out walking_stop_pt_idx_remainder
        segment_dist = distance(start_pt, end_pt)
        ratio = dist / segment_dist
        walking_stop_pt_idx_remainder = start_pt_idx_remainder + (1 - start_pt_idx_remainder) * ratio

        dist_after_walking = 0
        return (dist_after_walking, walking_stop_pt_idx, walking_stop_pt_idx_remainder)

    pass
    # return (ret_dist, ret_pt_idx, ret_pt_idx_remainder)

def get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist):
    dist = travel_dist
    count_max = len(metric_line)
    count = 0

    while (True):
        (t_dist, t_pt_idx, t_pt_idx_remainder) = reduce_distance_step(metric_line, pt_idx, pt_idx_remainder, dist)

        if pt_idx == len(metric_line) - 1 and pt_idx_remainder == 1:
            # CASE 0: This is mean the distanced point will out of the line
            print('out of line and remaind unproces distance = ', t_dist)
            t_pt = metric_line[-1].copy()
            return (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist)
            break
        if t_dist == 0:
            # CASE 1: All distance have been reduced
            t_pt = get_metric_pt(metric_line, t_pt_idx, t_pt_idx_remainder)
            return (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist)

        count += 1
        if count > count_max:
            # CASE 2: over looping of what we expect. This is case of bug in my source code
            print('The out of counting in loop is happended. this is a bug')
            t_pt = get_metric_pt(metric_line, t_pt_idx, t_pt_idx_remainder)
            return (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist)
        pt_idx = t_pt_idx
        pt_idx_remainder = t_pt_idx_remainder
        dist = t_dist

# Useful
def get_maps_with_folder(folder):
    import pydicom
    import os
    z_map = {}
    ct_filepath_map = {}
    ct_filelist = []
    for file in os.listdir(folder):
        ct_filepath = r"{}/{}".format(folder, file)
        ct_fp = None
        try:
            ct_fp = pydicom.read_file(ct_filepath)
            if ct_fp.Modality != 'CT':
                continue
        except:
            continue

        # print(ct_filepath)
        ct_filelist.append(ct_filepath)

        x_spacing, y_spacing = float(ct_fp.PixelSpacing[0]), float(ct_fp.PixelSpacing[1])
        origin_x, origin_y, origin_z = ct_fp.ImagePositionPatient
        # print(x_spacing, y_spacing)
        # print(origin_x, origin_y, origin_z)

        z_dict = {}
        z_dict['ct_filepath'] = ct_filepath
        z_dict['origin_x'] = origin_x
        z_dict['origin_y'] = origin_y
        z_dict['origin_z'] = origin_z
        z_dict['x_spacing'] = x_spacing
        z_dict['y_spacing'] = y_spacing

        z_map[origin_z] = z_dict
        ct_filepath_map[ct_filepath] = z_dict
    return z_map, ct_filepath_map


# The CT data is the format with 512 x 512, but we want to transfer it into real metric space
def convert_lines_in_metrics(lines, ct_folder):
    from decimal import Decimal
    z_map, ct_filepath_map = get_maps_with_folder(ct_folder)
    new_lines = []
    for i in range(len(lines)):
        new_lines.append([])

    for line_idx in range(len(lines)):
        line = lines[line_idx]
        new_line = new_lines[line_idx]
        for pt in line:
            pt_z = pt[2]
            z_dict = z_map[pt_z]
            x_spacing = z_dict['x_spacing']
            y_spacing = z_dict['y_spacing']
            origin_x = z_dict['origin_x']
            origin_y = z_dict['origin_y']
            pt_x = pt[0]
            pt_y = pt[1]
            tmp_x = pt_x * x_spacing + origin_x
            tmp_y = pt_y * y_spacing + origin_y
            new_pt_x = float(Decimal(str(tmp_x)).quantize(Decimal('0.00')))  # Some format transfer stuff
            new_pt_y = float(Decimal(str(tmp_y)).quantize(Decimal('0.00')))  # Some format transfer stuff
            new_pt = [new_pt_x, new_pt_y, pt_z]
            new_line.append(new_pt)
    return new_lines

def get_and_show_tandem(metric_line, first_purpose_distance_mm, each_purpose_distance_mm):
    tandem_rp_line = []

    def distance(pt1, pt2):
        import math
        # print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2)
        return ret_dist

    pt_idx = 0
    pt_idx_remainder = 0
    # first_purpose_distance_mm = 7 # get first RD point by 7mm
    # each_purpose_distance_mm = 5
    travel_dist = first_purpose_distance_mm
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,
                                                                                         pt_idx_remainder, travel_dist)
    print(t_pt)
    tandem_rp_line.append(t_pt)

    for i in range(100):
        eeeStr = 'eee 0'
        try:
            eeeStr = 'eee a'
            travel_dist = each_purpose_distance_mm
            eeeStr = 'eee b'
            (pt_idx, pt_idx_remainder) = (t_pt_idx, t_pt_idx_remainder)
            eeeStr = 'eee c'
            (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,
                                                                                                 pt_idx_remainder,
                                                                                                 travel_dist)
            eeeStr = 'eee d'
            print(t_pt, t_pt_idx, t_pt_idx_remainder)
            eeeStr = 'eee f'
            if (t_pt == tandem_rp_line[-1]):
                break
            tandem_rp_line.append(t_pt)
            eeeStr = 'eee g'
        except Exception as e:
            print('Exception happen START : {} and eeeStr = {}'.format(e, eeeStr))
            traceback.print_exc(file=sys.stdout)
            print('Exception happen END: {} and eeeStr = {}'.format(e, eeeStr))
            break
    return tandem_rp_line


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__




# Un-useful
def drwang_output_result_dump(f_list, dump_filepath):
    idx = 0
    drwang_output_result = {}
    for folder in f_list:
        if idx <= 1:
            idx = idx + 1
            continue
        print(folder)
        blockPrint()
        line = get_CT_tandem_metric_line_by_folder(folder)
        enablePrint()
        print('len(line) = {}, line = {}'.format(len(line), line))
        interpolated_line = line_interpolate(line, 20)
        print('len() = {}, interpolated_line = {}'.format(len(interpolated_line), interpolated_line))
        print(interpolated_line)

        out3_list = []  # interpolated_line, [man_pt, distance]
        out3_dict = {}
        for pt in interpolated_line:
            float_ai_pt = [float(i) for i in pt]
            tuple_pt = tuple(float_ai_pt)
            item = [None, '', '']
            item[0] = tuple_pt
            out3_dict[tuple_pt] = item
            out3_list.append(item)
        man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
        print('number points of man_line = {}'.format(len(man_line)))
        for pt in man_line:
            # print(pt)
            ai_pt, dist = get_closed_ai_pt(interpolated_line, pt)
            print('man_pt = {},  most closed ai_pt = {} with dist={}'.format(pt, ai_pt, dist))
            tuple_ai_pt = tuple(ai_pt)
            item = out3_dict[tuple_ai_pt]
            float_man_pt = [float(i) for i in pt]  # convert man_pt in list type into float man_pt
            item[1] = tuple(float_man_pt)
            item[2] = dist
        # show data
        drwang_output_result[folder] = out3_list
    # python_object_dump(drwang_output_result, 'drwang_output_result.bytes')
    python_object_dump(drwang_output_result, dump_filepath)


# Un-useful
def drwang_output_result_to_csv(dump_filepath, csv_filepath):
    drwang_output_result = python_object_load(dump_filepath)
    print('aaa')
    sorted_folder = sorted(drwang_output_result.keys())
    # Step 1. insert header behind body
    for folder in sorted_folder:
        print(folder)
        header = [folder, '', '']
        body = drwang_output_result[folder]  # body is list for data, which are 3 element list
        body.insert(0, header)
    # Step 2. figure which is the maxminum value ofr len of each body
    maximum_len = 0
    for folder in sorted_folder:
        body = drwang_output_result[folder]
        len_body = len(body)
        if len_body > maximum_len:
            maximum_len = len_body

    # Step 3. Start to generate csv
    # Step 3.1 generate total maxiumum list
    maximum_list = []
    for idx in range(maximum_len):
        row_item = []
        for folder in sorted_folder:
            body = drwang_output_result[folder]
            if idx < len(body):
                row_item.extend(body[idx])
            else:
                row_item.extend(['', '', ''])
        maximum_list.append(row_item)
    # Step 3.1.2 Eliminate the float, whose decimal point are too long,  value by round() (optional)
    for row_idx, rowlist in enumerate(maximum_list):
        if row_idx == 0:  # if header, ignore it
            continue
        for col_idx, tuple_item in enumerate(rowlist):
            # cell_item = [] #maximum_list[row_idx][col_idx]
            if type(tuple_item) == tuple:
                cell_item = []
                for float_item in tuple_item:
                    cell_item.append(round(float_item, 3))
                tuple_cell_item = tuple(cell_item)
                maximum_list[row_idx][col_idx] = tuple_cell_item
    # Step 3.2 generate csv file
    output_csv_filepath = csv_filepath
    with open(output_csv_filepath, mode='w', newline='') as csv_file:
        csv_writter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writter.writerow(out_dict['header'])
        for rowlist in maximum_list:
            csv_writter.writerow(rowlist)
    pass


# Un-useful
wang_f_list = [
    'RAL_plan_new_20190905/29059811-3',  # (max = 1.38, avg = 1.02)
    'RAL_plan_new_20190905/34698361-1',  # (max = 4.103, avg = 0.906 )
    'RAL_plan_new_20190905/34698361-5',  # (max = 2.47, avg = 0.545 )
    'RAL_plan_new_20190905/35413048-3'  # (max = 2.936, avg = 0.598 )
]


# Un-Useful
def drwang_output_show_3D(dump_filepath, show_folder='RAL_plan_new_20190905/29059811-3'):
    from mpl_toolkits import mplot3d
    import numpy as np
    from matplotlib import cm
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x_list = []
    y_list = []
    z_list = []
    c_ai = 255
    c_ai_diff = 128
    c_man = 0
    c_list = []

    # Write code to fill x_list,y_list, z_list and c_list
    drwang_output_result = python_object_load(dump_filepath)
    sorted_folder = sorted(drwang_output_result.keys())
    summary_result = {}
    for folder in sorted_folder:
        data_list = drwang_output_result[folder]
        folder_summary = {}
        summary_result[folder] = folder_summary
        summary_list = []
        folder_summary['list'] = summary_list
        for data_item in data_list:
            d = data_item
            # data_item is in format of
            # [(ai_x,ai_y,ai_z), '', ''] or
            # [(ai_x,ai_y,ai_z), (man_x,man_y,man_z), dist]
            # if dist != '', it mean manual pt(man_x,man_y,man_z) is matching to closed ai differential point in (ai_x,ai_y,ai_z)
            if (type(d[2]) != str):
                ai_pt = d[0]
                man_pt = d[1]
                dist = d[2]
                # print(dist)
                summary_list.append([ai_pt, man_pt, dist])

    d = summary_result[show_folder]
    d_list = d['list']
    for item in d_list:
        ai_pt = item[0]
        man_pt = item[1]
        x_list.append(ai_pt[0])
        y_list.append(ai_pt[1])
        z_list.append(ai_pt[2])
        c_list.append(c_ai)

        x_list.append(man_pt[0])
        y_list.append(man_pt[1])
        z_list.append(man_pt[2])
        c_list.append(c_man)

    ax.scatter3D(x_list, y_list, z_list, c=c_list, cmap=cm.coolwarm)
    plt.show()


# drwang_output_show_3D(dump_filepath = 'drwang_output_result.bytes')

# Un-useful
for f in wang_f_list:
    if (True == False):
        show_folder = f
        print('3D data for show_folder = ', show_folder)
        drwang_output_show_3D(dump_filepath='drwang_output_result.bytes', show_folder=show_folder)


# Un-useful
def drawang_output_show_avg_max_min(dump_filepath):
    drwang_output_result = python_object_load(dump_filepath)
    sorted_folder = sorted(drwang_output_result.keys())
    summary_result = {}
    for folder in sorted_folder:
        data_list = drwang_output_result[folder]
        folder_summary = {}
        summary_result[folder] = folder_summary
        summary_list = []
        folder_summary['list'] = summary_list
        for data_item in data_list:
            d = data_item
            # data_item is in format of
            # [(ai_x,ai_y,ai_z), '', ''] or
            # [(ai_x,ai_y,ai_z), (man_x,man_y,man_z), dist]
            # if dist != '', it mean manual pt(man_x,man_y,man_z) is matching to closed ai differential point in (ai_x,ai_y,ai_z)
            if (type(d[2]) != str):
                ai_pt = d[0]
                man_pt = d[1]
                dist = d[2]
                # print(dist)
                summary_list.append([ai_pt, man_pt, dist])
    for folder in sorted_folder:
        d = summary_result[folder]
        dist_max = 0.0
        dist_sum = 0.0
        for item in d['list']:
            dist = item[2]
            if dist > dist_max:
                dist_max = dist
            dist_sum = dist_sum + dist
        dist_avg = dist_sum / len(d['list'])
        d['max_dist'] = dist_max
        d['avg_dist'] = dist_avg
        print('folder={}\n dist_max={}\n dist_avg={}\n\n'.format(folder, d['max_dist'], d['avg_dist']))


# Un-useful
def process_drwang_output_csv_compare_output():
    bytes_filepath = 'drwang_output_result.bytes'
    # drwang_output_result_dump(f_list, dump_filepath=bytes_filepath)
    # drawang_output_show_avg_max_min(dump_filepath=bytes_filepath)
    drawang_output_show_avg_max_min(dump_filepath=bytes_filepath)
    # drwang_output_result_to_csv(dump_filepath=bytes_filepath, csv_filepath='drwang_output_result.csv')


# process_drwang_output_csv_compare_output()

# Un-useful
def test_draw_3d():
    from mpl_toolkits import mplot3d
    import numpy as np
    from matplotlib import cm
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    maxlen = 256
    x_list = [i for i in range(maxlen)]
    y_list = [i for i in range(maxlen)]
    z_list = [i for i in range(maxlen)]
    c_ai = 255
    c_ai_diff = 128
    c_man = 0
    c_list = [c_ai] * maxlen
    ax.scatter3D(x_list, y_list, z_list, c=c_list, cmap=cm.coolwarm)
    plt.show()


# test_draw_3d()


# Un-useful
def process_manual_point_5mm_check(f_list, csv_filepath):
    # Step 1. Set out_dict
    from statistics import mean
    out_dict = {}
    for folder in f_list:
        print(folder)
        folder_data = {}

        man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
        # print(man_line)
        dists = []
        for idx, pt in enumerate(man_line):
            if idx == 0:
                continue
            pt1 = man_line[idx - 1]
            pt2 = pt
            record_data = [pt1, pt2, distance(pt1, pt2)]
            dists.append(record_data)

        for idx, d in enumerate(dists):
            print('dists[{}]={}'.format(idx, dists[idx]))
        folder_data['dists'] = dists
        pure_dists = [item[2] for item in dists]
        folder_data['pure_dists'] = pure_dists
        folder_data['dists_max'] = max(pure_dists)
        folder_data['dists_avg'] = mean(pure_dists)
        folder_data['dists_min'] = min(pure_dists)
        print('max = {}'.format(max(pure_dists)))
        print('avg(mean) = {}'.format(mean(pure_dists)))
        print('\n')
        out_dict[folder] = folder_data

    sorted_folders = sorted(out_dict.keys())
    for folder in sorted_folders:
        folder_data = out_dict[folder]
        print(folder_data)
        break
    # Step 2. save out_dict into csv file
    output_csv_filepath = csv_filepath
    with open(output_csv_filepath, mode='w', newline='') as csv_file:
        csv_writter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        sorted_folders = sorted(out_dict.keys())
        # Step 2.1 header row prepare
        row_folder_name = []
        for folder in sorted_folders:
            row_folder_name.extend([folder, '', ''])
        row_max_dist = []
        for folder in sorted_folders:
            row_max_dist.extend(['', 'max distance = ', out_dict[folder]['dists_max']])
        row_avg_dist = []
        for folder in sorted_folders:
            row_avg_dist.extend(['', 'avg distance = ', out_dict[folder]['dists_avg']])
        row_min_dist = []
        for folder in sorted_folders:
            row_min_dist.extend(['', 'min distance = ', out_dict[folder]['dists_min']])
        row_empty = []
        for folder in sorted_folders:
            row_empty.extend(['', '', ''])
        row_header_of_body = []
        for folder in sorted_folders:
            row_header_of_body.extend(['start point', 'end point', 'distance'])
        csv_writter.writerow(row_folder_name)
        csv_writter.writerow(row_max_dist)
        csv_writter.writerow(row_avg_dist)
        csv_writter.writerow(row_min_dist)
        csv_writter.writerow(row_empty)
        csv_writter.writerow(row_header_of_body)

        # Step 2.2 body row prepare
        print(out_dict[folder].keys())
        maximum_len = 0
        for folder in sorted_folders:
            if maximum_len < len(out_dict[folder]['dists']):
                maximum_len = len(out_dict[folder]['dists'])
        for idx in range(maximum_len):
            row = []
            for folder in sorted_folders:
                dists = out_dict[folder]['dists']
                if idx >= len(dists):
                    cell_datas = ['', '', '']
                else:
                    record_data = dists[idx]
                    from_pt = record_data[0]
                    to_pt = record_data[1]
                    dist = record_data[2]
                    from_pt = [round(float(v), 3) for v in from_pt]
                    to_pt = [round(float(v), 3) for v in to_pt]
                    tuple_from_pt = tuple(from_pt)
                    tuple_to_pt = tuple(to_pt)
                    the_dist = round(dist, 6)
                    cell_datas = [tuple_from_pt, tuple_to_pt, the_dist]
                row.extend(cell_datas)
            csv_writter.writerow(row)
        # for row in sheet.rows:
        #    csv_writter.writerow([cell.value for cell in row])
    print('Done to write csv_filepath = {}'.format(output_csv_filepath))
    pass


# process_manual_point_5mm_check(f_list, 'manual_point_5mm_check.csv')

# Un-useful
def get_man_points(folder):
    print('folder = ', folder)
    man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
    float_man_metric_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in man_line]
    return float_man_metric_line


# Un-useful
def get_ai_points(folder):
    print('folder = ', folder)
    # STEP 1. get endpoint from AI predict points
    # the function will get all 3D pt of applicator
    app_pts = algo_run_by_folder(folder)
    # transform all 3D pt of applicator into each line for each applicator and the line have been sorted by z
    lines = make_lines_process(app_pts)
    # The CT data is the format with 512 x 512, but we want to tranfer it into real metric space
    metric_lines = convert_lines_in_metrics(lines, folder)
    # Show the lines information in metrics
    show_lines(metric_lines)
    metric_line = metric_lines[1].copy()
    float_ai_metric_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in metric_line]
    return float_ai_metric_line


# Un-useful
def get_ai_man_endpoints(folder):
    print('folder = ', folder)
    # STEP 1. get endpoint from AI predict points
    # the function will get all 3D pt of applicator
    app_pts = algo_run_by_folder(folder)
    # transform all 3D pt of applicator into each line for each applicator and the line have been sorted by z
    lines = make_lines_process(app_pts)
    # The CT data is the format with 512 x 512, but we want to tranfer it into real metric space
    metric_lines = convert_lines_in_metrics(lines, folder)
    # Show the lines information in metrics
    show_lines(metric_lines)
    metric_line = metric_lines[1].copy()
    print('metric_line = ', metric_line)
    float_ai_metric_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in metric_line]
    max_z = max([pt[2] for pt in float_ai_metric_line])
    pt_with_max_z = [pt for pt in float_ai_metric_line if pt[2] == max_z][0]
    print('max_z = {}'.format(max_z))
    print('pt_with_max_z = {}'.format(pt_with_max_z))
    ai_endpoint_pt = pt_with_max_z

    man_endpoint_pt = (0.0, 0.0, 0.0)

    man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
    float_man_metric_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in man_line]
    max_z = max([pt[2] for pt in float_man_metric_line])
    pt_with_max_z = [pt for pt in float_man_metric_line if pt[2] == max_z][0]
    man_endpoint_pt = pt_with_max_z
    # print('float_man_metric_line printing')
    # for pt in float_man_metric_line:
    #    print(pt)

    print('ai_endpoint_pt = {}'.format(ai_endpoint_pt))
    print('man_endpoint_pt = {}'.format(man_endpoint_pt))
    return (ai_endpoint_pt, man_endpoint_pt)


# Un-useful
def pickle_dump_ai_man_endpoints_dict(f_list, dump_filepath='ai_man_endpoints.bytes'):
    out_dict = {}
    for folder in f_list:
        blockPrint()
        (ai_endpoint_pt, man_endpoint_pt) = get_ai_man_endpoints(folder)
        enablePrint()
        print(
            'folder = {}\nai_endpoint_pt = {}\nman_endpoint_pt={}\n\n'.format(folder, ai_endpoint_pt, man_endpoint_pt))
        out_dict[folder] = (ai_endpoint_pt, man_endpoint_pt)
    print('The dumped out_dict = {}'.format(out_dict))
    python_object_dump(out_dict, dump_filepath)


# pickle_dump_ai_man_endpoints_dict(f_list, dump_filepath='ai_man_endpoints.bytes')

# Un-useful
def pickle_dump_ai_man_points_dict(f_list, dump_filepath='ai_man_points.bytes'):
    out_dict = {}
    for folder in f_list:
        blockPrint()
        ai_points = get_ai_points(folder)
        man_points = get_man_points(folder)
        enablePrint()
        out_dict[folder] = {}
        out_dict[folder]['ai_points'] = ai_points
        out_dict[folder]['man_points'] = man_points
        print('out_dict[{}] = {}'.format(folder, out_dict[folder]))
    python_object_dump(out_dict, dump_filepath)


# pickle_dump_ai_man_points_dict(f_list, dump_filepath='ai_man_points.bytes')


# Un-useful
def travel_5mm_check_with_man_first_point(f_list, dump_filepath):
    import math
    drwang_output_result = {}
    for f_idx, folder in enumerate(f_list):
        # if f_idx > 1:
        #    break

        print(folder)
        blockPrint()
        line = get_CT_tandem_metric_line_by_folder(folder)
        enablePrint()
        print('len(line) = {}, line = {}'.format(len(line), line))
        interpolated_line = line_interpolate(line, 20)
        print('len() = {}, interpolated_line = {}'.format(len(interpolated_line), interpolated_line))
        print(interpolated_line)

        float_tuple_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in line]
        print('float_tuple_line = {}'.format(float_tuple_line))
        # Find the closed point in interpolated_line s.t. it most clost to first man point
        tmp_man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
        float_tuple_man_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in tmp_man_line]
        float_tuple_interpolated_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in interpolated_line]
        print('man tips with idx=0->', float_tuple_man_line[0])  # it is with max z -> tips
        print(float_tuple_man_line[-1])  # it is with min z
        print(float_tuple_interpolated_line[0])  # it is with min z
        print(float_tuple_interpolated_line[-1])  # it is with max_z -> tips
        print(float_tuple_line[0])  # it is with min z
        print('ai tips with idx=-1->', float_tuple_line[-1])  # it is with max_z -> tips

        man_tips_pt = float_tuple_man_line[0]  # man_tips_pt is tips point in man line
        ai_tips_pt = float_tuple_line[-1]
        closed_pt = float_tuple_interpolated_line[-1]

        def figure_dist(pt1, pt2):
            return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2)

        closed_dist = figure_dist(man_tips_pt, closed_pt)
        for pt in float_tuple_interpolated_line:
            dist = figure_dist(man_tips_pt, pt)
            if dist < closed_dist:
                closed_dist = dist
                closed_pt = pt
        # Now closed_pt is the pt in interpolated_line that most closed to man_tips_pt
        # Process the case if closed_pt is never in man_line (maybe ai predict too much error)
        if closed_pt == ai_tips_pt:
            print('man_tips_pt = {}, closed_pt {} is equal to ai_tips_pt {}'.format(man_tips_pt, closed_pt, ai_tips_pt))
        else:
            print('man_tips_pt ={}, closed_pt {} is NOT equal to ai_tips_pt {}'.format(man_tips_pt, closed_pt,
                                                                                       ai_tips_pt))
        # we remove all float_tuple_line's pt that z > closed_pt'z
        # And then add closed_pt into float_tuple_line if there is no closed_pt in float_tuple_line
        new_float_tuple_line = float_tuple_line.copy()
        print('new len{}, old len{} '.format(len(new_float_tuple_line), len(float_tuple_line)))
        for pt in float_tuple_line:
            if pt[2] >= closed_pt[2]:
                if pt in new_float_tuple_line:
                    new_float_tuple_line.remove(pt)
        print('new len{}, old len{} '.format(len(new_float_tuple_line), len(float_tuple_line)))
        new_float_tuple_line.append(closed_pt)
        print('new len{}, old len{} '.format(len(new_float_tuple_line), len(float_tuple_line)))
        print('new ai_tips_pt = {}'.format(new_float_tuple_line[-1]))

        # Now we have new_Float_tuple_line, and then we can create interpolated_line again
        # Because the following code is start to travel each node in 5mm
        # line = get_CT_tandem_metric_line_by_folder(folder)
        # enablePrint()

        # interpolated_line = line_interpolate(line, 20)
        interpolated_line = line_interpolate(new_float_tuple_line, 20)
        print('new_interpolated_line = {}'.format(interpolated_line))

        out3_list = []  # interpolated_line, [man_pt, distance]
        out3_dict = {}
        for pt in interpolated_line:
            float_ai_pt = [float(i) for i in pt]
            tuple_pt = tuple(float_ai_pt)
            item = [None, '', '']
            item[0] = tuple_pt
            out3_dict[tuple_pt] = item
            out3_list.append(item)

        man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
        print('number points of man_line = {}'.format(len(man_line)))
        for pt in man_line:
            # print(pt)
            ai_pt, dist = get_closed_ai_pt(interpolated_line, pt)
            print('man_pt = {},  most closed ai_pt = {} with dist={}'.format(pt, ai_pt, dist))

            tuple_ai_pt = tuple(ai_pt)
            item = out3_dict[tuple_ai_pt]
            float_man_pt = [float(i) for i in pt]  # convert man_pt in list type into float man_pt
            item[1] = tuple(float_man_pt)
            item[2] = dist
        # show data
        drwang_output_result[folder] = out3_list
    # python_object_dump(drwang_output_result, 'drwang_output_result.bytes')
    python_object_dump(drwang_output_result, dump_filepath)


# travel_5mm_check_with_man_first_point(f_list = f_list, dump_filepath='travel_5mm_with_manual_tip.bytes')

# result = python_object_load('travel_5mm_with_manual_tip.bytes')
# Un-useful
def process_new_drwang_output_csv_compare_output():
    bytes_filepath = 'travel_5mm_with_manual_tip.bytes'
    # drwang_output_result_dump(f_list, dump_filepath=bytes_filepath)
    # drawang_output_show_avg_max_min(dump_filepath=bytes_filepath)
    drawang_output_show_avg_max_min(dump_filepath=bytes_filepath)
    drwang_output_result_to_csv(dump_filepath=bytes_filepath, csv_filepath='travel_5mm_with_manual_tip.csv')


# process_new_drwang_output_csv_compare_output()


# Un-useful
def travel_every_5mm_in_ai(f_list, dump_filepath):
    import math
    drwang_output_result = {}
    for f_idx, folder in enumerate(f_list):
        print(folder)
        blockPrint()
        line = get_CT_tandem_metric_line_by_folder(folder)
        enablePrint()
        print('len(line) = {}, line = {}'.format(len(line), line))
        interpolated_line = line_interpolate(line, 20)
        print('len() = {}, interpolated_line = {}'.format(len(interpolated_line), interpolated_line))
        print(interpolated_line)

        """
        float_tuple_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in line]
        print('float_tuple_line = {}'.format(float_tuple_line))
        #Find the closed point in interpolated_line s.t. it most clost to first man point
        tmp_man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
        float_tuple_man_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in tmp_man_line]
        float_tuple_interpolated_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in interpolated_line]
        print('man tips with idx=0->',float_tuple_man_line[0]) # it is with max z -> tips
        print(float_tuple_man_line[-1]) # it is with min z
        print(float_tuple_interpolated_line[0]) # it is with min z
        print(float_tuple_interpolated_line[-1]) # it is with max_z -> tips
        print(float_tuple_line[0]) # it is with min z
        print('ai tips with idx=-1->',float_tuple_line[-1]) # it is with max_z -> tips

        man_tips_pt = float_tuple_man_line[0]  # man_tips_pt is tips point in man line
        ai_tips_pt = float_tuple_line[-1]
        closed_pt = float_tuple_interpolated_line[-1]

        def figure_dist(pt1, pt2):
            return math.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 )

        closed_dist = figure_dist(man_tips_pt, closed_pt)
        for pt in float_tuple_interpolated_line:
            dist = figure_dist(man_tips_pt, pt)
            if dist < closed_dist:
                closed_dist = dist
                closed_pt = pt
        # Now closed_pt is the pt in interpolated_line that most closed to man_tips_pt
        # Process the case if closed_pt is never in man_line (maybe ai predict too much error)
        if closed_pt == ai_tips_pt:
            print('man_tips_pt = {}, closed_pt {} is equal to ai_tips_pt {}'.format(man_tips_pt, closed_pt, ai_tips_pt))
        else:
            print('man_tips_pt ={}, closed_pt {} is NOT equal to ai_tips_pt {}'.format(man_tips_pt, closed_pt, ai_tips_pt))
        # we remove all float_tuple_line's pt that z > closed_pt'z
        # And then add closed_pt into float_tuple_line if there is no closed_pt in float_tuple_line
        new_float_tuple_line = float_tuple_line.copy()
        print('new len{}, old len{} '.format(len(new_float_tuple_line), len(float_tuple_line)))




        for pt in float_tuple_line:
            if pt[2] >= closed_pt[2]:
                if pt in new_float_tuple_line:
                    new_float_tuple_line.remove(pt)
        print('new len{}, old len{} '.format(len(new_float_tuple_line), len(float_tuple_line)))
        new_float_tuple_line.append(closed_pt)
        print('new len{}, old len{} '.format(len(new_float_tuple_line), len(float_tuple_line)))
        print('new ai_tips_pt = {}'.format(new_float_tuple_line[-1]))

        # Now we have new_Float_tuple_line, and then we can create interpolated_line again
        # Because the following code is start to travel each node in 5mm
        #line = get_CT_tandem_metric_line_by_folder(folder)
        #enablePrint()
        #interpolated_line = line_interpolate(line, 20)
        """

        new_float_tuple_line = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in line]
        interpolated_line = line_interpolate(new_float_tuple_line, 20)
        print('new_interpolated_line = {}'.format(interpolated_line))

        out3_list = []  # interpolated_line, [man_pt, distance]
        out3_dict = {}
        for pt in interpolated_line:
            float_ai_pt = [float(i) for i in pt]
            tuple_pt = tuple(float_ai_pt)
            item = [None, '', '']
            item[0] = tuple_pt
            out3_dict[tuple_pt] = item
            out3_list.append(item)

        man_line = get_CT_tandem_metric_rp_line_by_folder(folder)
        print('number points of man_line = {}'.format(len(man_line)))
        for pt in man_line:
            # print(pt)
            ai_pt, dist = get_closed_ai_pt(interpolated_line, pt)
            print('man_pt = {},  most closed ai_pt = {} with dist={}'.format(pt, ai_pt, dist))

            tuple_ai_pt = tuple(ai_pt)
            item = out3_dict[tuple_ai_pt]
            float_man_pt = [float(i) for i in pt]  # convert man_pt in list type into float man_pt
            item[1] = tuple(float_man_pt)
            item[2] = dist
        # show data
        drwang_output_result[folder] = out3_list
    # python_object_dump(drwang_output_result, 'drwang_output_result.bytes')
    python_object_dump(drwang_output_result, dump_filepath)


# travel_every_5mm_in_ai(f_list = f_list, dump_filepath='travel_every_5mm_in_ai.bytes')

# Un-useful
def run_and_make_rp(folder, out_rp_filepath):
    rp_template_filepath = r'RP_Template/Brachy_RP.1.2.246.352.71.5.417454940236.2063186.20191015164204.dcm'
    rs_filepath = ''
    ct_filelist = []
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        fp = pydicom.read_file(filepath)
        if (fp.Modality == 'CT'):
            ct_filelist.append(filepath)
        elif (fp.Modality == 'RTSTRUCT'):
            rs_filepath = filepath
    # Read RS file as input
    rs_fp = pydicom.read_file(rs_filepath)
    # read RP tempalte into rp_fp
    rp_fp = pydicom.read_file(rp_template_filepath)

    rp_fp.OperatorsName = 'cylin'
    rp_fp.PhysiciansOfRecord = rs_fp.PhysiciansOfRecord
    rp_fp.FrameOfReferenceUID = rs_fp.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
    rp_fp.ReferencedStructureSetSequence[0].ReferencedSOPClassUID = rs_fp.SOPClassUID
    rp_fp.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = rs_fp.SOPInstanceUID

    directAttrSet = [
        'PhysiciansOfRecord', 'PatientName', 'PatientID',
        'PatientBirthDate', 'PatientBirthTime', 'PatientSex',
        'DeviceSerialNumber', 'SoftwareVersions', 'StudyID',
        'StudyDate', 'StudyTime', 'StudyInstanceUID']
    for attr in directAttrSet:
        val = getattr(rs_fp, attr)
        setattr(rp_fp, attr, val)
    rp_fp.InstanceCreationDate = rp_fp.RTPlanDate = rp_fp.StudyDate = rs_fp.StudyDate
    rp_fp.RTPlanTime = str(float(rs_fp.StudyTime) + 0.001)
    rp_fp.InstanceCreationTime = str(float(rs_fp.InstanceCreationTime) + 0.001)

    # Start to prepare 5mm points and write data into rp_fp as points
    # TODO

    # In the finally, just write file back
    pydicom.write_file(out_rp_filepath, rp_fp)


# run_and_make_rp(folder='RAL_plan_new_20190905/29059811-1', out_rp_filepath=r'out.brachy.rp.withpoints.dcm')


def run_and_make_rp_v02(folder, out_rp_filepath):
    print('folder = ', folder)
    # the function will get all 3D pt of applicator
    app_pts = algo_run_by_folder(folder)
    # transform all 3D pt of applicator into each line for each applicator and the line have been sorted by z
    lines = make_lines_process(app_pts)
    # The CT data is the format with 512 x 512, but we want to tranfer it into real metric space
    metric_lines = convert_lines_in_metrics(lines, folder)
    metric_line = metric_lines[1].copy()
    print('metric_line = ', metric_line)

    def distance(pt1, pt2):
        import math
        # print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2)
        return ret_dist

    pt_idx = 0
    pt_idx_remainder = 0
    purpose_distance_mm = 5
    max_mm = purpose_distance_mm
    orig_pt = metric_line[0]
    print('metric_line = ', metric_line)

    pt_idx = 0
    pt_idx_remainder = 0
    orig_pt = metric_line[0]
    # purpose_distance_mm = 7
    purpose_distance_mm = 5
    travel_dist = purpose_distance_mm
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,
                                                                                         pt_idx_remainder, travel_dist)
    print('{} -> {}'.format((t_pt, t_pt_idx, t_pt_idx_remainder), distance(orig_pt, t_pt)))

    # tandem_rp_line = get_and_show_tandem(metric_line, 4.5, 5) # 29059811-1 => z=58 tips to z=-48
    metric_line.reverse()
    tandem_rp_line = get_and_show_tandem(metric_line, 4, 5)
    print('metric_line[-1] = ', metric_line[-1])
    print('metric_line[0] = ', metric_line[0])

    # show_tandem(metric_line, 4.5, 5)
    print('tandem_rp_line[-1] = ', tandem_rp_line[-1])
    print('tandem_rp_line[0] = ', tandem_rp_line[0])
    print('len(tandem_rp_line) = ', len(tandem_rp_line))
    print('tandem_rp_line = {}', tandem_rp_line)

    # max_mm = purpose_distance_mm
    # orig_pt = metric_line[0]
    # for mm in range(max_mm+1):
    #    travel_dist = mm
    #    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)
    #    print( '{} -> {}'.format((t_pt, t_pt_idx, t_pt_idx_remainder), distance(orig_pt,t_pt) )  )


# run_and_make_rp_v02(folder='RAL_plan_new_20190905/29059811-1', out_rp_filepath=r'out.brachy.rp.withpoints.v04.dcm')
# run_and_make_rp_v02(folder='RP_Template_TestData', out_rp_filepath=r'out.brachy.rp.withpoints.v04.dcm')
run_and_make_rp_v02(folder='RALmilo', out_rp_filepath=r'out.brachy.rp.withpoints.v04.dcm')

# folder = 'RAL_plan_new_20190905/34698361-1'
# print('folder = {}'.format(folder))
# app_pts = algo_run_by_folder(folder)
# print('app_pts = \n {}', app_pts)
# lines = make_lines_process(app_pts)
# The CT data is the format with 512 x 512, but we want to tranfer it into real metric space
# metric_lines = convert_lines_in_metrics(lines, folder)
# print('lines[0] = {}\n'.format(lines[0]))
# print('lines[1] = {}\n'.format(lines[1]))
# print('lines[2] = {}\n'.format(lines[2]))

# for folder in wang_f_list:
#    get_ai_man_endpoints(folder)
#    break


exit(0)

# Un-useful
if True:
    broken_f_list = []
    for folder in f_list:
        break
        # if folder != 'RAL_plan_new_20190905/35252020-2':
        #    continue
        try:
            ai_tandem_rp_line = predict_tandem_rp_line_by_folder(folder, start_mm=4.5, gap_mm=5, is_debug=False)
            man_tandem_rp_line = get_tandem_from_man(man_dict, folder)
            print('folder = {}, \nai_tandem_rp_line= {}, \nman_tandem_rp_line={}\n'.format(folder, ai_tandem_rp_line,
                                                                                           man_tandem_rp_line))
        except:
            enablePrint()  # Because predict_tandem_rp_line_by_folder() use blockPrint(), so enablePrint when catch exception
            print('folder  = {} is break'.format(folder))
            broken_f_list.append(folder)
            continue
    print('len = {}, f_list = {}'.format(len(f_list), f_list))
    print('len = {}, broken_f_list = {}'.format(len(broken_f_list), broken_f_list))

    broken_f_list = ['RAL_plan_new_20190905/29059811-2', 'RAL_plan_new_20190905/35252020-2']

    folder_idx = 0
    for folder in broken_f_list:
        break
        if folder_idx != 0:
            ai_tandem_rp_line = predict_tandem_rp_line_by_folder(folder, start_mm=4.5, gap_mm=5, is_debug=False)
            print('processed error folder name = ', folder)
            break
        folder_idx = folder_idx + 1

    # exit(0)

    broken_f_list = []
    f_list = []
    correct_dir_f_list = []
    incorrect_dir_f_list = []
    loop_idx = 0

    test_f_list = ['RAL_plan_new_20190905/34698361-5', 'RAL_plan_new_20190905/35413048-3']
    for folder in sorted(man_dict.keys()):
        if folder not in test_f_list:
            continue
        # if folder != 'RAL_plan_new_20190905/35252020-2': #Case
        #    continue

        # if folder != 'RAL_plan_new_20190905/35413048-3': #Case tandem cannot get to over middle button
        #    continue

        # if folder != 'RAL_plan_new_20190905/34698361-1': # Case of diff_dist > 7mm
        #    continue
        print('<START> loop_idx = {}'.format(loop_idx))
        print('folder = {}, with folder_idx = {}'.format(folder, folder_idx))
        # figure out the distance between ai tandem line and manual tandem line

        man_line = get_tandem_from_man(man_dict, folder)
        ai_line = []
        try:
            ai_line = predict_tandem_rp_line_by_folder(folder, start_mm=0, gap_mm=5)
        except:
            enablePrint()
            print('Why dead on case of folder = {}? finding it '.format(folder))
            broken_f_list.append(folder)
            continue
        f_list.append(folder)
        ai_line = list(reversed(ai_line))
        print('folder = {}\nman_line={}\nai_line={}\n\n'.format(folder, man_line, ai_line))
        print('p2p compare')
        max_len = max([len(ai_line), len(man_line)])
        for idx in range(max_len):
            ai_pt = []
            man_pt = []
            if idx < len(ai_line):
                ai_pt = ai_line[idx]
            if idx < len(man_line):
                man_pt = man_line[idx]
            print('idx = {} '.format(idx))
            print('ai_pt = {}'.format(ai_pt))
            print('man_pt = {}'.format(man_pt))
            if len(man_pt) > 0 and len(ai_pt) > 0:
                d = math.sqrt((ai_pt[0] - man_pt[0]) ** 2 + (ai_pt[1] - man_pt[1]) ** 2 + (ai_pt[2] - man_pt[2]) ** 2)
                print('distance = {}'.format(d))
            print('\n')

        # ai_line = reversed(ai_line)
        # ai_line is generated from outside into deeper-side
        # But brachy is tag light  from most deepr-side first.
        # So we reverse the order of tandem ai_line.
        # Them we can compare

        # print(ai_line)
        man_line_len = len(man_line)
        if man_line_len > len(ai_line):
            print('In case folder = {}, len of man line = {} > len of ai line = {}'.format(folder, man_line_len,
                                                                                           len(ai_line)))
            incorrect_dir_f_list.append(folder)
            continue
        else:
            correct_dir_f_list.append(folder)
        man_1st_pt = man_line[0]
        ai_1st_pt = ai_line[0]
        man_last_pt = man_line[man_line_len - 1]
        ai_last_pt = ai_line[man_line_len - 1]
        dist_1st = distance(ai_1st_pt, man_1st_pt)

        dist_last = distance(ai_last_pt, man_last_pt)
        # diff_distance = distance(ai_last_pt, man_last_pt) - distance(ai_1st_pt, man_1st_pt)
        diff_distance = dist_last - dist_1st
        print('ai_1st_pt = ', ai_1st_pt)
        print('man_1st_pt = ', man_1st_pt)
        print('dist_1st = ', dist_1st)
        print('')
        print('ai_last_pt = ', ai_last_pt)
        print('man_last_pt = ', man_last_pt)
        print('dist_last = ', dist_last)
        print('')
        print('folder = {}, diff_distance = {}'.format(folder, diff_distance))

        blockPrint()
        # app_pts = algo_run_by_folder(folder)
        # app_pts_show3D(app_pts)
        enablePrint()
        folder_idx = folder_idx + 1
        loop_idx = loop_idx + 1

    print('broken_f_list = {}'.format(broken_f_list))
    print('f_list = {}'.format(f_list))
    print('correct_dir_f_list = {}'.format(correct_dir_f_list))
    print('incorrect_dir_f_list = {}'.format(incorrect_dir_f_list))

    print('folder_idx = ', folder_idx)
    exit(0)






