# Try to seperate program into clear verion and useful functions
import os
import pydicom
import numpy as np
import cv2
import copy
import math
from sys import exit
import sys
import datetime


from IPython.display import display, HTML

import openpyxl
import csv, codecs
from decimal import Decimal
import random
import pickle


# FUNCTIONS - Utility
def python_object_dump(obj, filename):
    file_w = open(filename, "wb")
    pickle.dump(obj, file_w)
    file_w.close()
def python_object_load(filename):
    try:
        file_r = open(filename, "rb")
        obj2 = pickle.load(file_r)
        file_r.close()
    except:
        try:
            file_r.close()
            return None
        except:
            return None
    return obj2
# enable / disable for print

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# FUNCTIONS - Algorithm processing Fucntions
def distance(pt1, pt2):
    axis_num = len(pt1)
    # Assume maximum of axis number of pt is 3
    # Because we may save pt[4] as another appended information for algorithm
    if(axis_num > 3):
        axis_num = 3

    sum = 0.0
    for idx in range(axis_num):
        sum = sum + (pt1[idx] - pt2[idx]) ** 2
    ans = math.sqrt(sum)
    return ans
def convert_to_gray_image(pixel_array):
    img = np.copy(pixel_array)
    # Convert to float to avoid overflow or underflow losses.
    img_2d = img.astype(float)
    # Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    # Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)
    return img_2d_scaled
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
    filter_img = np.zeros((A.shape[0], A.shape[1], 3), np.uint8)
    filter_img[A <= constant] = (0, 0, 0)
    filter_img[A > constant] = (255, 255, 255)
    gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
    _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)
    return (contours, constant)
def get_max_contours_by_filter_img(A, filter_img, ContourRetrievalMode=cv2.RETR_EXTERNAL):
    # gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
    gray_image = filter_img
    # findContours
    # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(gray_image, ContourRetrievalMode, cv2.CHAIN_APPROX_NONE)
    return contours
def get_view_scope_by_dicom_dict(dicom_dict):
    def get_rect_info_from_cv_contour(cv_contour):
        i = cv_contour
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
        return rect_info
    def get_rect_infos_and_center_pts(contours, h_min=13, w_min=13, h_max=19, w_max=19):
        app_center_pts = []
        app_center_pts_extend_data = []
        rect_infos = []
        for contour in contours:
            # Step 2. make useful information
            rect_info = get_rect_info_from_cv_contour(cv_contour=contour)
            # rect_info is [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
            """
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
            """
            (x_min, x_max, y_min, y_max) = rect_info[0]
            (w, h) = rect_info[1]
            (x_mean, y_mean) = rect_info[2]

            # if h >= 13 and h < 19 and w >= 13 and h < 19:
            if h >= h_min and h < h_max and w >= w_min and w < w_max:
                cen_pt = [x_mean, y_mean]
                app_center_pts.append(cen_pt)
                app_center_pts_extend_data.append({'cen_pt': cen_pt, 'contour': contour, 'rect_info': rect_info})
            else:
                # print('(h={},{} , w={},{})'.format(h_max, h_min, w_max, w_min))
                # print('Not matching ! rect_info = ', rect_info)
                pass
            # print(rect_info)
            rect_infos.append(rect_info)
        sorted_app_center_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
        return (sorted_app_center_pts, rect_infos, app_center_pts, app_center_pts_extend_data)
    # Figure out (x_min, x_max, y_min, y_max) view scope that without bone
    # Get the first slice (minimum z value)
    #dicom_dict = get_dicom_dict(folder)
    z_map = dicom_dict['z']
    first_slice_z = sorted(z_map.keys())[0] # the z is minimum value. The most light OVoid and tandem
    ct_obj = z_map[first_slice_z]
    ps_x = ct_obj['ps_x']
    ps_y = ct_obj['ps_y']
    h_max = int((19.0 * 4.19921e-1) / ps_y)
    h_min = int((13.0 * 4.19921e-1) / ps_y)
    w_max = int((19.0 * 4.19921e-1) / ps_x)
    w_min = int((13.0 * 4.19921e-1) / ps_x)
    (contours, constant) = get_max_contours(ct_obj['rescale_pixel_array'])
    (sorted_app_center_pts, rect_infos, app_center_pts, app_center_pts_extend_data) = get_rect_infos_and_center_pts(contours, h_max=h_max, w_max=w_max, h_min=h_min, w_min=w_min)
    x_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
    y_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[1], reverse=False)
    (min_x, max_x) = (x_sorted_pts[0][0], x_sorted_pts[-1][0])
    (min_y, max_y) = (y_sorted_pts[0][1], y_sorted_pts[-1][1])
    #print('X:({}, {})'.format(min_x, max_x))
    #print('Y:({}, {})'.format(min_y, max_y))
    w = max_x - min_x
    h = max_y - min_y
    dist = np.max([w, h])
    cen_x = int(min_x + (w) / 2)
    cen_y = int(min_y + (h) / 2)
    loc_min_x = int(cen_x - dist / 2)
    loc_max_x = int(cen_x + dist / 2)
    loc_min_y = int(cen_y - dist / 2)
    loc_max_y = int(cen_y + dist / 2)
    #print('loc X:({}, {})'.format(loc_min_x, loc_max_x))
    #print('loc Y:({}, {})'.format(loc_min_y, loc_max_y))
    padding = 50
    view_min_y = loc_min_y - padding
    view_max_y = loc_max_y + padding
    view_min_x = loc_min_x - padding
    view_max_x = loc_max_x + padding
    #(view_min_y, view_max_y, view_min_x, view_max_x) = (0,0,0,0)
    # fake answer
    return (156, 356, 156, 356)
    #return (view_min_y=156, view_max_y=356, view_min_x=156, view_max_x=356)
    return (view_min_y, view_max_y, view_min_x, view_max_x)
def get_contours_from_edge_detection_algo_01(img, filter_img):
    contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_TREE)
    return contours
def get_contours_from_edge_detection_algo_02(img, filter_img):
    contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_EXTERNAL)
    return contours
def get_contours_from_edge_detection_algo_03(img):
    (contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_TREE)
    contours = contours_without_filter
    return contours
def get_contours_from_edge_detection_algo_04(img):
    (contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_EXTERNAL)
    contours = contours_without_filter
    return contours
def get_rect_info_from_cv_contour(cv_contour):
    i = cv_contour
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
    return rect_info
def get_contour_xy_mean(cv_contour):
    rect_info = get_rect_info_from_cv_contour(cv_contour)
    (x_mean, y_mean) = rect_info[2]
    return (x_mean, y_mean)
def get_contour_area_mm2(contour,ps_x, ps_y) :
    area_mm2  = cv2.contourArea(contour) * ps_x * ps_y
    return area_mm2
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
def get_rect_info_from_cv_contour(cv_contour):
    i = cv_contour
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
    return rect_info
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
def algo_to_get_pixel_lines(dicom_dict):
    # type: (dicom_dict) -> (lt_ovoid, tandem, rt_ovoid)
    # Step 1. Use algo01 to get center point of inner contour
    last_z_in_step1 = sorted(dicom_dict['z'].keys())[0]
    center_pts_dict = {} # The following loop will use algo03 to figure L't Ovoid, R't Ovoid and half tandem
    for z in sorted(dicom_dict['z'].keys()):
        contours = dicom_dict['z'][z]['output']['contours512']['algo03']
        #plot_with_contours(dicom_dict, z=z, algo_key='algo03')
        # Step 1.1 The process to collect the contour which is inner of some contour into inner_contours[]
        inner_contours = []
        for inner_idx, inner_contour in enumerate(contours):
            is_inner = False
            for outer_idx, outer_contour in enumerate(contours):
                if inner_idx == outer_idx: # ignore the same to compare itself
                    continue
                outer_rect = get_minimum_rect_from_contours([outer_contour])
                if is_contour_in_rect(inner_contour, outer_rect):
                    is_inner = True
                    break
            if (is_inner == True) :
                inner_contours.append(inner_contour)
        # Step 1.2 if there is no any inner contour, then last_z_in_step1 = z and exit the z loop
        if (len(inner_contours)) == 0:
            last_z_in_step1 = z
            break

        # Step 1.3 figure out center point of contour in inner_contour and sorting it by the order x
        print('z = {}, len(inner_contours) = {}'.format(z, len(inner_contours)))
        inner_cen_pts = []
        for contour in inner_contours:
            #rect_info = [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
            rect_info = get_rect_info_from_cv_contour(contour)
            cen_pt = ( rect_info[2][0], rect_info[2][1] )
            inner_cen_pts.append(cen_pt)
        inner_cen_pts.sort(key=lambda pt:pt[0])
        print('z = {}, inner_cen_pts = {}'.format(z, inner_cen_pts) )
        center_pts_dict[z] = inner_cen_pts


    # Step 2. Figure L't Ovoid
    print('STEP 2.')
    lt_ovoid = []
    allowed_distance_mm = 2.5 # allowed distance when trace from bottom to tips of L't Ovoid
    prev_info = {}
    prev_info['pt'] = None
    prev_info['ps_x'] = None
    prev_info['ps_y'] = None
    print('sorted(center_pts_dict.keys()) = {}'.format(sorted(center_pts_dict.keys())))
    for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']
        if idx_z == 0:
            prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            lt_ovoid.append(prev_pt)
            continue
        prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
        prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
        x_mm = center_pts_dict[z][0][0] * ps_x
        y_mm = center_pts_dict[z][0][1] * ps_y
        if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
            prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            lt_ovoid.append(prev_pt)
            print('lt_ovoid = {}'.format(lt_ovoid))

        else:
            break

    # Step 3. Figure R't Ovoid
    rt_ovoid = []
    allowed_distance_mm = 2.5 # allowed distance when trace from bottom to tips of R't Ovoid
    prev_info = {}
    prev_info['pt'] = None
    prev_info['ps_x'] = None
    prev_info['ps_y'] = None
    print('sorted(center_pts_dict.keys()) = {}'.format(sorted(center_pts_dict.keys())))
    for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']
        if idx_z == 0:
            prev_pt = ( center_pts_dict[z][-1][0], center_pts_dict[z][-1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            rt_ovoid.append(prev_pt)
            continue
        prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
        prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
        x_mm = center_pts_dict[z][-1][0] * ps_x
        y_mm = center_pts_dict[z][-1][1] * ps_y
        if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
            prev_pt = ( center_pts_dict[z][-1][0], center_pts_dict[z][-1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            rt_ovoid.append(prev_pt)
            print('rt_ovoid = {}'.format(rt_ovoid))
        else:
            #distance_mm = math.sqrt((x_mm - prev_x_mm) ** 2 + (y_mm - prev_y_mm) ** 2)
            #print('distnace_mm = {}, '.format(distance_mm))

            break

    # Step 4. Figure Tandem bottom-half (thicker pipe part of tandem)
    tandem = []
    allowed_distance_mm = 4.5 # allowed distance when trace from bottom to tips
    prev_info = {}
    prev_info['pt'] = None
    prev_info['ps_x'] = None
    prev_info['ps_y'] = None
    for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']
        if idx_z == 0:
            # It is possible that thicker pipe part of tandem is not scanned in CT file, so that only can detect two pipe in this case.
            # So that when center_pts_dict < 3 in following case after using algo03
            if (len(center_pts_dict[z]) < 3)  :
                break
            prev_pt = ( center_pts_dict[z][1][0], center_pts_dict[z][1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            tandem.append(prev_pt)
            continue
        prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
        prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
        x_mm = center_pts_dict[z][1][0] * ps_x
        y_mm = center_pts_dict[z][1][1] * ps_y
        #print('x_mm = {}, y_mm ={}'.format(x_mm, y_mm))
        if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
            prev_pt = ( center_pts_dict[z][1][0], center_pts_dict[z][1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            tandem.append(prev_pt)
            print('tandem = {}'.format(tandem))
        else:
            break
    #
    # Step 5. The case to process the tandem without thicker pipe in scanned CT. when tandem = [] (empty list)
    if len(tandem) == 0:
        # Step 5.1 find out inner_cotnour of tandem by algo01
        contours = dicom_dict['z'][z]['output']['contours512']['algo01']
        #plot_with_contours(dicom_dict, z=z, algo_key='algo03')
        # Step 5.1.1 The process to collect the contour which is inner of some contour into inner_contours[]
        inner_contours = []
        for inner_idx, inner_contour in enumerate(contours):
            is_inner = False
            for outer_idx, outer_contour in enumerate(contours):
                if inner_idx == outer_idx: # ignore the same to compare itself
                    continue
                outer_rect = get_minimum_rect_from_contours([outer_contour])
                if is_contour_in_rect(inner_contour, outer_rect):
                    is_inner = True
                    break
            if (is_inner == True) :
                inner_contours.append(inner_contour)
        # Step 5.1.2 figure out center point of contour in inner_contour and sorting it by the order x
        print('z = {}, len(inner_contours) = {}'.format(z, len(inner_contours)))
        inner_cen_pts = []
        for contour in inner_contours:
            #rect_info = [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
            rect_info = get_rect_info_from_cv_contour(contour)
            cen_pt = ( rect_info[2][0], rect_info[2][1] )
            inner_cen_pts.append(cen_pt)
        inner_cen_pts.sort(key=lambda pt:pt[0])
        print('tandem first slice evaluation inner_cen_pts = {}'.format(inner_cen_pts))
        if(len(inner_cen_pts) != 3 ) :
            print('inner_cen_pts is not == 3')
            raise Exception
        tandem.append( (inner_cen_pts[1][0], inner_cen_pts[1][1], float(z)) )
        # Step 5.1. Find Algo01 and detect the inner_contour

        print('TODO tandem for the case that without thicker pipe in scanned CT')

    # Step 6. Trace tandem
    print('Step 6. Trace tandem')
    # Step 6.1 Figure out [upper_half_z_idx_start, upper_half_z_idx_end) for upper-part of tandem
    z = sorted(dicom_dict['z'].keys())[0]
    last_z = tandem[-1][2]
    print('last_z = {}'.format(last_z))
    z_idx = sorted(dicom_dict['z'].keys()).index(last_z)
    upper_half_z_idx_start = z_idx + 1 # upper_half_z_idx_start is the next z of last_z in current tandem data.
    upper_half_z_idx_end = len(dicom_dict['z'].keys())
    print('upper_half_z_idx [start,end) = [{},{})'.format(upper_half_z_idx_start, upper_half_z_idx_end))

    # Step 6.2 Setup first prev_info for loop to run and also set allowed_distnace to indicate the largest moving distance between two slice.
    # allowed_distance_mm = 8.5 # allowed distance when trace from bottom to tips of Tandem [ 8.5 mm is not ok for 35252020-2 ]
    #allowed_distance_mm = 10.95  # allowed distance when trace from bottom to tips of Tandem
    allowed_distance_mm = 10.95  # allowed distance when trace from bottom to tips of Tandem

    prev_info = {}
    prev_info['pt'] = tandem[-1]
    prev_info['ps_x'] = dicom_dict['z'][last_z]['ps_x']
    prev_info['ps_y'] = dicom_dict['z'][last_z]['ps_y']
    # The case for 29059811-2 folder , will have the following value
    # last_z == -92.0
    # prev_info == {'pt': (240, 226, -92.0), 'ps_x': "3.90625e-1", 'ps_y': "3.90625e-1"}

    # Step 6.3. Start to trace tandem
    for z_idx in range(upper_half_z_idx_start, upper_half_z_idx_end):
        z = sorted(dicom_dict['z'].keys())[z_idx]
        ps_x = dicom_dict['z'][z]['ps_x']
        ps_y = dicom_dict['z'][z]['ps_y']

        #print('z = {}'.format(z))
        # Step 6.3.1. Make contours variable as collecting of all contour in z-slice
        contours = []
        for algo_key in dicom_dict['z'][z]['output']['contours512'].keys():
            contours = contours + dicom_dict['z'][z]['output']['contours512'][algo_key]
        # Step 6.3.2. Convert to center pt that the idx is the same as to contours
        cen_pts = []
        for c_idx, c in enumerate(contours):
            rect_info = get_rect_info_from_cv_contour(c)
            # [(x_min, x_max, y_min, y_max), (w, h), (x_mean, y_mean)]
            cen_pt = (rect_info[2][0], rect_info[2][0])
            cen_pts.append(cen_pt)
        # Step 6.3.3
        # prev_info is like {'pt': (240, 226, -92.0), 'ps_x': "3.90625e-1", 'ps_y': "3.90625e-1"}
        # pt in cen_pts is like (240, 226)
        minimum_distance_mm = allowed_distance_mm + 1  # If minimum_distance_mm is finally large than allowed_distance_mm, it's mean there is no pt closed to prev_pt
        minimum_pt = (0, 0)
        for pt in cen_pts:
            prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
            prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
            x_mm = pt[0] * ps_x
            y_mm = pt[1] * ps_y
            distance_mm = math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2 )
            if (distance_mm > allowed_distance_mm ) : # distance_mm cannot large than allowed_distance_mm
                continue
            if (distance_mm < minimum_distance_mm):
                minimum_distance_mm = distance_mm
                minimum_pt = pt

        if (minimum_distance_mm > allowed_distance_mm):
            # This is case to say ending for the upper looper
            break
        else:
            tandem.append( (minimum_pt[0], minimum_pt[1],float(z)) )
            prev_info['pt'] = minimum_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            print('tandem = {}'.format(tandem))
    return (lt_ovoid, tandem, rt_ovoid)

def get_applicator_rp_line(metric_line, first_purpose_distance_mm, each_purpose_distance_mm):
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
            if (segment_dist == 0):
                # To solve bug of divide zero, Sometimes the segment_dist will be zero
                segment_dist = 0.000000001
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

    tandem_rp_line = []
    pt_idx = 0
    pt_idx_remainder = 0
    travel_dist = first_purpose_distance_mm
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,pt_idx_remainder, travel_dist)
    tandem_rp_line.append(t_pt)
    for i in range(100):
        travel_dist = each_purpose_distance_mm
        (pt_idx, pt_idx_remainder) = (t_pt_idx, t_pt_idx_remainder)
        (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,pt_idx_remainder,travel_dist)
        if (t_pt == tandem_rp_line[-1]):
            break
        tandem_rp_line.append(t_pt)

    return tandem_rp_line

def wrap_to_rp_file(RP_OperatorsName, rs_filepath, tandem_rp_line, out_rp_filepath, lt_ovoid_rp_line, rt_ovoid_rp_line):
    rp_template_filepath = r'RP_Template/Brachy_RP.1.2.246.352.71.5.417454940236.2063186.20191015164204.dcm'
    def get_new_uid(old_uid='1.2.246.352.71.5.417454940236.2063186.20191015164204', study_date='20190923'):
        uid = old_uid
        def gen_6_random_digits():
            ret_str = ""
            for i in range(6):
                ch = chr(random.randrange(ord('0'), ord('9') + 1))
                ret_str += ch
            return ret_str
        theStudyDate = study_date
        uid_list = uid.split('.')
        uid_list[-1] = theStudyDate + gen_6_random_digits()
        new_uid = '.'.join(uid_list)
        return new_uid

    # Read RS file as input
    rs_fp = pydicom.read_file(rs_filepath)
    # read RP tempalte into rp_fp
    rp_fp = pydicom.read_file(rp_template_filepath)

    rp_fp.OperatorsName = RP_OperatorsName
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
        #rs_val = getattr(rs_fp, attr)
        #rp_val = getattr(rp_fp, attr)
        #print('attr={}, \n In RS->{} \n In RP->{}'.format(attr, rs_val, rp_val))
        val = getattr(rs_fp, attr)
        setattr(rp_fp, attr, val)
        #new_rp_val = getattr(rp_fp, attr)
        #print('after update, RP->{}\n'.format(new_rp_val))

    newSeriesInstanceUID = get_new_uid(old_uid=rp_fp.SeriesInstanceUID, study_date=rp_fp.StudyDate)
    newSOPInstanceUID = get_new_uid(old_uid=rp_fp.SOPInstanceUID, study_date=rp_fp.StudyDate)
    rp_fp.SeriesInstanceUID = newSeriesInstanceUID
    rp_fp.SOPInstanceUID = newSOPInstanceUID
    rp_fp.InstanceCreationDate = rp_fp.RTPlanDate = rp_fp.StudyDate = rs_fp.StudyDate
    rp_fp.RTPlanTime = str(float(rs_fp.StudyTime) + 0.001)
    rp_fp.InstanceCreationTime = str(float(rs_fp.InstanceCreationTime) + 0.001)

    # Clean Dose Reference
    rp_fp.DoseReferenceSequence.clear()


    # The template structure for applicator
    # Tandem -> rp_fp.ApplicationSetupSequence[0].ChannelSequence[0]
    # Rt Ovoid -> rp_fp.ApplicationSetupSequence[0].ChannelSequence[1]
    # Lt OVoid -> rp_fp.ApplicationSetupSequence[0].ChannelSequence[2]
    # For each applicator .NumberOfControlPoints is mean number of point
    # For each applicator .BrachyControlPointSequence is mean the array of points


    BCPItemTemplate = copy.deepcopy(rp_fp.ApplicationSetupSequence[0].ChannelSequence[0].BrachyControlPointSequence[0])
    rp_lines = [tandem_rp_line, rt_ovoid_rp_line, lt_ovoid_rp_line]

    #TODO rp_Ref_ROI_Numbers need to match to current RS's ROI number of three applicators
    rp_Ref_ROI_Numbers = [16, 17, 18]
    rp_ControlPointRelativePositions = [3.5, 3.5, 3.5]
    for idx,rp_line in enumerate(rp_lines):
        # Change ROINumber of RP_Template_TestData RS into output RP output file
        # Do  I need to fit ROINumber in RS or not? I still have no answer
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].ReferencedROINumber = rp_Ref_ROI_Numbers[idx]
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].NumberOfControlPoints = len(rp_line)
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].BrachyControlPointSequence.clear()
        for pt_idx, pt in enumerate( rp_line ):
            BCPPt = copy.deepcopy(BCPItemTemplate)
            BCPPt.ControlPointRelativePosition = rp_ControlPointRelativePositions[idx] + pt_idx * 5
            BCPPt.ControlPoint3DPosition[0] = pt[0]
            BCPPt.ControlPoint3DPosition[1] = pt[1]
            BCPPt.ControlPoint3DPosition[2] = pt[2]
            BCPStartPt = copy.deepcopy(BCPPt)
            BCPEndPt = copy.deepcopy(BCPPt)
            BCPStartPt.ControlPointIndex = 2 * pt_idx
            BCPEndPt.ControlPointIndex = 2 * pt_idx + 1
            rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].BrachyControlPointSequence.append(BCPStartPt)
            rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].BrachyControlPointSequence.append(BCPEndPt)

    pydicom.write_file(out_rp_filepath, rp_fp)

    pass



# FUNCTIONS - DICOM data processing Functions
def get_dicom_folder_pathinfo(folder):
    dicom_folder = {}
    ct_filelist = []
    rs_filepath = None
    rd_filepath = None
    rp_filepath = None
    for file in os.listdir(folder):
        filepath = os.path.join(folder ,file)
        try:
            ct_dicom = pydicom.read_file(filepath)
            """
            CT Computed Tomography
            RTDOSE Radiotherapy Dose
            RTPLAN Radiotherapy Plan
            RTSTRUCT Radiotherapy Structure Set
            """
            m = ct_dicom.Modality
            if (m == 'CT'):
                ct_filelist.append(filepath)
            elif (m == 'RTDOSE'):
                rd_filepath = filepath
            elif (m == 'RTSTRUCT'):
                rs_filepath = filepath
            elif (m == 'RTPLAN'):
                rp_filepath = filepath
        except Exception as e:
            pass
    dicom_folder['ct_filelist'] = ct_filelist
    dicom_folder['rs_filepath'] = rs_filepath
    dicom_folder['rd_filepath'] = rd_filepath
    dicom_folder['rp_filepath'] = rp_filepath
    return dicom_folder
def get_dicom_dict(folder):
    z_map = {}
    ct_filepath_map = {}
    out_dict = {}
    out_dict['z'] = z_map
    out_dict['ct_filepath'] = ct_filepath_map
    out_dict['metadata'] = {}
    out_dict['metadata']['folder'] = folder
    pathinfo = get_dicom_folder_pathinfo(folder)
    out_dict['pathinfo'] = pathinfo

    rs_filepath = out_dict['pathinfo']['rs_filepath']
    rs_fp = pydicom.read_file(rs_filepath)
    out_dict['metadata']['RS_StudyDate'] = rs_fp.StudyDate
    out_dict['metadata']['RS_PatientID'] = rs_fp.PatientID
    out_dict['metadata']['RS_SOPInstanceUID'] = rs_fp.SOPInstanceUID

    ct_filelist = pathinfo['ct_filelist']
    for ct_filepath in ct_filelist:
        # print('ct_filepath = ', ct_filepath)
        ct_fp = pydicom.read_file(ct_filepath)
        ct_obj = {}
        ct_obj['dicom_dict'] = out_dict
        ct_obj['filepath'] = ct_filepath
        ct_obj["pixel_array"] = copy.deepcopy(ct_fp.pixel_array)
        ct_obj["RescaleSlope"] = ct_fp.RescaleSlope
        ct_obj["RescaleIntercept"] = ct_fp.RescaleIntercept
        ct_obj["rescale_pixel_array"] = ct_fp.pixel_array * ct_fp.RescaleSlope + ct_fp.RescaleIntercept
        ct_obj['ps_x'] = ct_fp.PixelSpacing[0]
        ct_obj['ps_y'] = ct_fp.PixelSpacing[1]
        ct_obj['origin_x'] = ct_fp.ImagePositionPatient[0]
        ct_obj['origin_y'] = ct_fp.ImagePositionPatient[1]
        ct_obj['origin_z'] = ct_fp.ImagePositionPatient[2]
        ct_obj['SliceLocation'] = ct_fp.SliceLocation
        ct_obj['output'] = {} # put your output result in here

        z = ct_obj['SliceLocation']
        z_map[ z ] = ct_obj
        ct_filepath_map[ct_filepath] = ct_obj
        #print('ct_obj={}'.format(ct_obj))
    return out_dict
def generate_metadata_to_dicom_dict(dicom_dict):
    (view_min_y, view_max_y, view_min_x, view_max_x) = get_view_scope_by_dicom_dict(dicom_dict)
    metadata = dicom_dict['metadata']
    metadata['view_scope'] = (view_min_y, view_max_y, view_min_x, view_max_x)
    # metadata['view_scope'] = (view_min_y, view_max_y, view_min_x, view_max_x)
def print_info_by_folder(folder):
    out_dict = get_dicom_dict(folder)
    #print('aaa')
    z_map = out_dict['z']
    for z_idx, z in enumerate(sorted(z_map.keys())):
        ct_obj = z_map[z]
        print('z={}, {}'.format(z, ct_obj.keys()))

# FUNCTIONS - generate our expected output for each ct_obj in dicom_dict['z'].   (PS:z_map)
def generate_output_to_dicom_dict(dicom_dict):
    folder = dicom_dict['metadata']['folder']
    z_map = dicom_dict['z']
    for z_idx, z in enumerate(sorted(z_map.keys())):
        ct_obj = z_map[z]
        #print('z={}, {}'.format(z, ct_obj.keys()))
        generate_output_to_ct_obj(ct_obj)
        # information is in ct_obj['output']
def generate_output_to_ct_obj(ct_obj):
    out = ct_obj['output']
    rescale_pixel_array = ct_obj['rescale_pixel_array']
    (view_min_y, view_max_y, view_min_x, view_max_x) = ct_obj['dicom_dict']['metadata']['view_scope']
    #view_pixel_array = rescale_pixel_array[view_min_y:view_max_y, view_min_x:view_max_x]
    img = ct_obj['rescale_pixel_array']
    gray_img = convert_to_gray_image(img)
    gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
    img = img[view_min_y: view_max_y, view_min_x:view_max_x]
    filter_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    ct_obj['output']['contours'] = {}
    ct_obj['output']['contours']['algo01'] = get_contours_from_edge_detection_algo_01(img, filter_img)
    ct_obj['output']['contours']['algo02'] = get_contours_from_edge_detection_algo_02(img, filter_img)
    ct_obj['output']['contours']['algo03'] = get_contours_from_edge_detection_algo_03(img)
    ct_obj['output']['contours']['algo04'] = get_contours_from_edge_detection_algo_04(img)

    # Process to contours to fit global pixel img
    ct_obj['output']['contours512'] = {}
    for algo_key in sorted(ct_obj['output']['contours'].keys()):
        ct_obj['output']['contours512'][algo_key] =  copy.deepcopy(ct_obj['output']['contours'][algo_key] )
        contours = ct_obj['output']['contours512'][algo_key]
        for contour in contours:
            for [pt] in contour:
                pt[0] = view_min_x + pt[0]
                pt[1] = view_min_y + pt[1]

    # Generate contours infos like x,y mean and area_mm
    ct_obj['output']['contours_infos'] = {}
    ps_x = ct_obj['ps_x']
    ps_y = ct_obj['ps_y']
    for algo_key in (ct_obj['output']['contours'].keys()):
        contours = ct_obj['output']['contours'][algo_key]
        contours_infos = []
        for contour in contours:
            contours_info = {}
            #contours_infos.append(contours_info)
            (x,y) = get_contour_xy_mean(contour)
            global_x_pixel = x + view_min_x
            global_y_pixel = y + view_min_y
            area_mm2 = get_contour_area_mm2(contour, ps_x, ps_y)
            contours_info['mean'] = [global_x_pixel, global_y_pixel]
            contours_info['area_mm2'] = area_mm2
            contours_info['contour'] = contour
            contours_infos.append(contours_info)
        ct_obj['output']['contours_infos'][algo_key] = contours_infos
    pass

# FUNCTIONS - main function
def generate_contour_number_csv_report(f_list, csv_filepath = 'contours.csv'):
    output_csv_filepath = csv_filepath
    all_dicom_dict = {}
    all_sheet_dict = {}

    # Figure out all_dicom_dict and make empty sheet
    for folder in sorted(f_list):
        dicom_dict = get_dicom_dict(folder)
        generate_metadata_to_dicom_dict(dicom_dict)
        generate_output_to_dicom_dict(dicom_dict)
        all_dicom_dict[folder] = dicom_dict
    # Figure out sheet_width and algo_keys
    dicom_dict = all_dicom_dict[ sorted(all_dicom_dict.keys())[0]]
    z_map = dicom_dict['z']
    ct_obj = z_map[ sorted(z_map.keys())[0] ]
    sorted_algo_keys = sorted(ct_obj['output']['contours'].keys())
    sheet_width = 1 + len(sorted_algo_keys) # (z, algo01, algo02 ,... algo n)

    # Generate all_sheet_dict and fill all of value in it
    max_sheet_len = 0
    for folder_idx, folder in enumerate(sorted(all_dicom_dict.keys())):
        dicom_dict = all_dicom_dict[folder]
        header1 = [folder] + [''] * (sheet_width -1)
        header2 = ['z'] + sorted_algo_keys
        sheet_dict = {}
        sheet_dict['header'] = [header1] + [header2]
        sheet_dict['body'] = []
        z_map = dicom_dict['z']
        for z in sorted(z_map.keys()):
            ct_obj = z_map[z]
            body_row = [z]
            for algo in sorted(ct_obj['output']['contours'].keys()):
                contours_num = len( ct_obj['output']['contours'][algo])
                body_row.append(contours_num)
            sheet_dict['body'] = sheet_dict['body'] + [body_row]
        all_sheet_dict[folder] = sheet_dict
        sheet_len = len(sheet_dict['header']) + len(sheet_dict['body'])
        if sheet_len > max_sheet_len:
            max_sheet_len = sheet_len
    # For every sheet_dict whose sheet_len < max_sheet_len, append empty row for sheet, so that the len of sheet_dict will the same as to max_sheet_len
    for folder in sorted(all_sheet_dict.keys()):
        sheet_dict = all_sheet_dict[folder]
        sheet_len = len(sheet_dict['header']) + len(sheet_dict['body'])
        append_sheet_len = max_sheet_len - sheet_len
        if (append_sheet_len > 0):
            empty_body_row = [''] * sheet_width
            sheet_dict['body'] = sheet_dict['body'] + ([empty_body_row] * append_sheet_len)
        sheet_dict['csv'] = sheet_dict['header'] + sheet_dict['body']
        #print('len of all_sheet_dict[folder={}] = {}'.format(folder, len( all_sheet_dict[folder]['csv'] )) )

    # Generate spread sheet
    csv_data = []
    for idx in range(max_sheet_len):
        csv_row = []
        for folder in sorted(all_sheet_dict.keys()):
            sheet_dict = all_sheet_dict[folder]
            csv_row = csv_row + all_sheet_dict[folder]['csv'][idx]
        csv_data.append(csv_row)
    # Start to write csv from csv_data into output_csv_filepath
    with open(output_csv_filepath, mode='w', newline='') as csv_file:
        csv_writter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #csv_writter.writerow(out_dict['header'])
        for rowlist in csv_data:
            csv_writter.writerow(rowlist)
def generate_patient_mean_area_csv_report(folder, algo_key='algo01', csv_filepath = '29059811-1-algo01.csv'):
    output_csv_filepath = csv_filepath
    dicom_dict = get_dicom_dict(folder)
    generate_metadata_to_dicom_dict(dicom_dict)
    generate_output_to_dicom_dict(dicom_dict)
    sheet_width = 0
    sheet_height = 0
    z_map = dicom_dict['z']
    max_of_contours = 0
    for z in sorted(z_map.keys()):
        ct_obj = z_map[z]
        contours_infos = ct_obj['output']['contours_infos'][algo_key]
        len_contours = len(contours_infos)
        if len_contours > max_of_contours:
            max_of_contours = len_contours
    sheet_width = 2 + max_of_contours # 1st col is z, 2nd col is number of contours
    sheet_height = 3 + len(z_map.keys()) # 1st row -> folder name. 2nd row -> algo key. 3rd row->data header

    # declare sheet is list of list and will use it to write csv file
    sheet = []

    # write first 3 rows
    sheet.append([folder]) # idx = 0
    sheet.append([algo_key]) # idx = 1
    sheet.append(['z', 'contours#']) # idx = 2

    # fill first 2 columns
    for z in sorted(z_map.keys()):
        ct_obj = z_map[z]
        contours_infos = ct_obj['output']['contours_infos'][algo_key]
        contour_num = len(contours_infos)
        row = [z, contour_num]
        sheet.append(row)

    # fill space for empty cell in first 2 rows
    sheet[0] = sheet[0] + ['']*(sheet_width-len(sheet[0]))
    sheet[1] = sheet[1] + ['']*(sheet_width-len(sheet[1]))

    # write done for 3rd row
    header = sheet[2]
    for c_idx in range(sheet_width - 2):
        header.append('contour {}'.format(c_idx+1))

    # Now, 1,2,3 th row are finished. 1,2 th col are finished too.
    # another (row,col) is the cell to show contour's info

    # Process to write all contours info value
    for z_idx, z in enumerate(sorted(z_map.keys())):
        ct_obj = z_map[z]
        write_infos = []
        infos = copy.deepcopy(ct_obj['output']['contours_infos'][algo_key])
        infos.sort(key=lambda info: info['mean'][0]) # sorting infos by mean x
        for info_idx, info in enumerate(infos):
            mean_x = info['mean'][0]
            mean_y = info['mean'][1]
            area_mm2 = info['area_mm2']
            write_info = '({},{})px - {}mm2'.format(mean_x, mean_y, area_mm2)
            write_info_mean = '223,221'
            write_info_area = '23.1'

            write_infos.append(write_info)
        # Write infos into correct row_sheet
        sheet_row = copy.deepcopy(sheet[z_idx+3])
        sheet_row = sheet_row + write_infos
        sheet_row = sheet_row + ['']*(sheet_width-len(sheet_row))
        sheet[z_idx+3] = sheet_row
    with open(output_csv_filepath, mode='w', newline='') as csv_file:
        csv_writter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #csv_writter.writerow(out_dict['header'])
        for rowlist in sheet:
            csv_writter.writerow(rowlist)

    print('Time to lookup sheet variable')
def generate_all_patient_mean_area_csv_report(root_folder): # generate for each patient's data and put in more-infos folder
    root_folder = r'RAL_plan_new_20190905'
    f_list = [ os.path.join(root_folder, file) for file in os.listdir(root_folder) ]
    for folder in f_list:
        print(os.path.basename(folder))
        algo_keys = ['algo01', 'algo02', 'algo03', 'algo04']
        for algo_key in algo_keys:
            csv_filepath = r'more_infos/{}-{}.csv'.format(os.path.basename(folder), algo_key)
            generate_patient_mean_area_csv_report(folder, algo_key=algo_key, csv_filepath=csv_filepath)
    pass
def plot_with_contours(dicom_dict, z, algo_key):
    import matplotlib.pyplot as plt
    z_map = dicom_dict['z']
    ct_obj = z_map[z]
    print(ct_obj.keys())
    pixel_array = ct_obj['rescale_pixel_array']
    metadata = dicom_dict['metadata']
    contours = ct_obj['output']['contours512'][algo_key]
    img = copy.deepcopy(pixel_array)
    for contour in contours:
        cv2.drawContours(img, contour, -1, (0, 0, 255), 1)
    #plt.imshow(pixel_array, cmap=plt.cm.bone)
    folder_name = os.path.basename(dicom_dict['metadata']['folder'])
    plt.text(0, -2, 'z = {}, folder = {}, algo={}'.format(z, folder_name,algo_key), fontsize=10)

    plt.imshow(img, cmap=plt.cm.bone)
    plt.show()
    pass

def generate_brachy_rp_file(RP_OperatorsName, dicom_dict, out_rp_filepath, is_enable_print=False):
    if (is_enable_print == False):
        blockPrint()
    else:
        enablePrint()

    # Step 1. Get line of lt_ovoid, tandem, rt_ovoid by OpneCV contour material and innovated combination
    (lt_ovoid, tandem, rt_ovoid) = algo_to_get_pixel_lines(dicom_dict)

    # Step 2. Convert line into metric representation
    # Original line is array of (x_px, y_px, z_mm) and we want to convert to (x_mm, y_mm, z_mm)
    new_lines = []
    for line in [lt_ovoid, tandem, rt_ovoid]:
        new_line = []
        for pt in line:
            z = pt[2]
            ct_obj = dicom_dict['z'][z]
            x = pt[0] * ct_obj['ps_x'] + ct_obj['origin_x']
            y = pt[1] * ct_obj['ps_y'] + ct_obj['origin_y']
            new_line.append([x,y,z])
        new_lines.append(new_line)
    (metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = (new_lines[0], new_lines[1], new_lines[2])
    print('metric_lt_ovoid = {}'.format(metric_lt_ovoid))
    print('metric_tandem = {}'.format(metric_tandem))
    print('metric_rt_ovoid = {}'.format(metric_rt_ovoid))

    # Step 3. Reverse Order, so that first element is TIPS [from most top (z maximum) to most bottom (z minimum) ]
    metric_lt_ovoid.reverse()
    metric_tandem.reverse()
    metric_rt_ovoid.reverse()

    # Step 4. Get Applicator RP line
    tandem_rp_line = get_applicator_rp_line(metric_tandem, 4, 5)
    lt_ovoid_rp_line = get_applicator_rp_line(metric_lt_ovoid, 0, 5)
    rt_ovoid_rp_line = get_applicator_rp_line(metric_rt_ovoid, 0 ,5)
    print('lt_ovoid_rp_line = {}'.format(lt_ovoid_rp_line))
    print('tandem_rp_line = {}'.format(tandem_rp_line))
    print('rt_ovoid_rp_line = {}'.format(rt_ovoid_rp_line))

    # Step 5. Wrap to RP file

    print(dicom_dict['pathinfo']['rs_filepath'])
    print(dicom_dict['metadata'].keys())
    #print(pydicom.read_file(dicom_dict['pathinfo']['rs_filepath']).keys())

    rs_filepath = dicom_dict['pathinfo']['rs_filepath']

    print('out_rp_filepath = {}'.format(out_rp_filepath))
    wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line, out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, rt_ovoid_rp_line=rt_ovoid_rp_line)
    if (is_enable_print == False):
        enablePrint()



# FUNCTIONS - file dump function , so that you can accelerate develop speed
def contours_python_object_dump(root_folder, filename):
    # Step 1. declare all_dicom_dict
    all_dicom_dict = {}
    # Step 2. Generate all our target
    #root_folder = r'RAL_plan_new_20190905'
    f_list = [ os.path.join(root_folder, file) for file in os.listdir(root_folder) ]
    for folder in sorted(f_list):
        dicom_dict = get_dicom_dict(folder)
        generate_metadata_to_dicom_dict(dicom_dict)
        generate_output_to_dicom_dict(dicom_dict)
        all_dicom_dict[folder] = dicom_dict
        byte_filename = r'{}.bytes'.format(os.path.basename(folder))
        dump_filepath = os.path.join('contours_bytes', byte_filename)
        python_object_dump(dicom_dict, dump_filepath)
        print('Create {}'.format(dump_filepath))
    # Step 3. Use python_object_dump to dump it into some file
    print('Creating {} in very largest size'.format(filename))
    python_object_dump(all_dicom_dict, filename)
    print('Created {}'.format(filename))
def contours_python_object_load(filename):
    # load the file to load all contours algo's result
    obj = python_object_load(filename)
    return obj
def example_load_multiple_bytefile(): # example code of how to use contours_python_object_load _dump
    root_folder = r'RAL_plan_new_20190905'
    folders = os.listdir(root_folder)
    print('folders = {}'.format(folders))
    all_dicom_dict = {}
    for folder in folders:
        print('time={}: {}'.format(datetime.datetime.now(),'load start'))
        bytes_filepath = os.path.join('contours_bytes', r'{}.bytes'.format(folder))
        all_dicom_dict[folder] = python_object_load(bytes_filepath)
        print('time={}: {}'.format(datetime.datetime.now(),'load end'))
    return all_dicom_dict
def example_load_single_bytefile(): # example code of how to use contours_python_object_load _dump
    print('time={}: {}'.format(datetime.datetime.now(), 'Loading the file all_dicom_dict.bytes into all_dicom_dict object'))
    all_dicom_dict = contours_python_object_load('all_dicom_dict.bytes')
    print('time={}: {}'.format(datetime.datetime.now(),'Load done'))
    return all_dicom_dict
def example_dump_single_and_multiple_bytesfile():
    """
    From basic root_folder, Create all bytes files that are corresponding to every dicom_dict. Output log should like following
    Create contours_bytes\24460566-ctdate20191015.bytes
    Create contours_bytes\29059811-1.bytes
    Create contours_bytes\29059811-2.bytes
    ...
    Create contours_bytes\592697-1.bytes
    Create contours_bytes\592697-2.bytes
    Create contours_bytes\592697-3.bytes
    Creating all_dicom_dict.bytes in very largest size
    ...
    """
    root_folder = r'RAL_plan_new_20190905'
    contours_python_object_dump(root_folder, 'all_dicom_dict.bytes')
def example_create_all_rp_file():
    """
    ...
    The Code should show report like this

    [1/28]Create RP file -> all_rp_output\RP.24460566.20191015.f24460566-ctdate20191015.dcm -> 0:00:01.365078s [2019-10-30 10:05:43.507173-2019-10-30 10:05:44.872251]
    [2/28]Create RP file -> all_rp_output\RP.29059811.20190903.f29059811-1.dcm -> 0:00:00.886051s [2019-10-30 10:05:45.031260-2019-10-30 10:05:45.917311]
    ...
    [24/28]Create RP file -> all_rp_output\RP.413382.20190124.f413382-3.dcm -> 0:00:01.419081s [2019-10-30 10:06:11.038748-2019-10-30 10:06:12.457829]
    [25/28]Create RP file -> all_rp_output\RP.413382.20190122.f413382-4.dcm -> 0:00:01.343077s [2019-10-30 10:06:12.657840-2019-10-30 10:06:14.000917]
    [26/28]Create RP file -> all_rp_output\RP.592697.20190115.f592697-1.dcm -> 0:00:01.885108s [2019-10-30 10:06:14.330936-2019-10-30 10:06:16.216044]
    [27/28]Create RP file -> all_rp_output\RP.592697.20190110.f592697-2.dcm -> 0:00:01.592091s [2019-10-30 10:06:16.819078-2019-10-30 10:06:18.411169]
    FOLDER SUMMARY REPORT
    failed folders = ['34698361-3', '370648-3', '370648-4', '370648-5', '592697-2']
    failed / total = 5/28
    success /total = 23/28
    """
    root_folder = r'RAL_plan_new_20190905'
    print(os.listdir(root_folder))
    folders = os.listdir(root_folder)
    print('folders = {}'.format(folders))
    total_folders = []
    failed_folders = []
    success_folders = []
    for folder_idx, folder in enumerate(folders):
        total_folders.append(folder)
        try:
            bytes_filepath = os.path.join('contours_bytes', r'{}.bytes'.format(folder))
            dicom_dict = python_object_load(bytes_filepath)
            metadata = dicom_dict['metadata']
            # out_rp_filepath format is PatientID, RS StudyDate  and the final is folder name processing by coding
            out_rp_filepath = r'RP.{}.{}.f{}.dcm'.format(  metadata['RS_PatientID'],  metadata['RS_StudyDate'],  os.path.basename(metadata['folder']) )
            out_rp_filepath = os.path.join('all_rp_output', out_rp_filepath)
            time_start = datetime.datetime.now()
            print('[{}/{}]Create RP file -> {}'.format(folder_idx+1,len(folders), out_rp_filepath) ,end=' -> ')
            generate_brachy_rp_file(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath, is_enable_print=False)
            time_end = datetime.datetime.now()
            print('{}s [{}-{}]'.format(time_end-time_start, time_start, time_end), end='\n')
            success_folders.append(folder)
        except Exception as ex:
            print('Create Failed')
            failed_folders.append(folder)
    print('FOLDER SUMMARY REPORT')
    print('failed folders = {}'.format(failed_folders))
    print('failed / total = {}/{}'.format(len(failed_folders), len(total_folders) ))
    print('success /total = {}/{}'.format(len(success_folders), len(total_folders) ))


if __name__ == '__main__':
    example_dump_single_and_multiple_bytesfile()
    exit(0)

    #example_create_all_rp_file()
    #exit(0)

    root_folder = r'RAL_plan_new_20190905'
    print(os.listdir(root_folder))
    folders = os.listdir(root_folder)
    print('folders = {}'.format(folders))
    folder = '24460566-ctdate20191015'
    #folder = '35252020-2'
    #folder = '29059811-2'
    bytes_filepath = os.path.join('contours_bytes', r'{}.bytes'.format(folder))

    #plot_with_contours(dicom_dict, z=sorted(dicom_dict['z'].keys())[10], algo_key='algo03')
    dicom_dict = python_object_load(bytes_filepath)

    for z_idx, z in enumerate(sorted(dicom_dict['z'].keys())):
        #plot_with_contours(dicom_dict, z=sorted(dicom_dict['z'].keys())[z_idx], algo_key='algo01')
        continue

    metadata = dicom_dict['metadata']
    # out_rp_filepath format is PatientID, RS StudyDate  and the final is folder name processing by coding
    out_rp_filepath = r'RP.{}.{}.f{}.dcm'.format(  metadata['RS_PatientID'],  metadata['RS_StudyDate'],  os.path.basename(metadata['folder']) )
    out_rp_filepath = os.path.join('all_rp_output', out_rp_filepath)
    print('RP Create {}'.format(out_rp_filepath))
    time_start = datetime.datetime.now()
    print('Create RP file -> {}'.format(out_rp_filepath) ,end=' -> ')
    generate_brachy_rp_file(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath, is_enable_print=False)
    time_end = datetime.datetime.now()
    print('{}s [{}-{}]'.format(time_end-time_start, time_start, time_end), end='\n')




    exit(0)





