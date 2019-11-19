# Try to seperate program into clear verion and useful functions
import os
import errno
import pydicom
import numpy as np
import cv2
import copy
import math
from sys import exit
import sys
import datetime
import csv, codecs
from decimal import Decimal
from shutil import copyfile
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
def blockPrint(): # Disable printing
    sys.stdout = open(os.devnull, 'w')
def enablePrint(): # Restore for printing
    sys.stdout = sys.__stdout__
def create_directory_if_not_exists(path):
    """
    Creates 'path' if it does not exist
    If creation fails, an exception will be thrown
    :param path:    the path to ensure it exists
    """
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            print('An error happened trying to create ' + path)
            raise

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
    #return (156, 356, 156, 356)
    return (0, 512, 0, 512)
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
    #return (50, 50, 462, 462)
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
def get_contours_from_edge_detection_algo_05(img, contour_constant_value):
    (contours_without_filter, constant) = get_max_contours(img, constant_value=contour_constant_value, ContourRetrievalMode=cv2.RETR_TREE)
    contours = contours_without_filter
    return contours
def get_contours_from_edge_detection_algo_06(img, contour_constant_value):
    # the img should be rescale_pixel_array
    (contours_without_filter, constant) = get_max_contours(img, constant_value=contour_constant_value, ContourRetrievalMode=cv2.RETR_EXTERNAL)
    contours = contours_without_filter
    return contours
def get_contours_from_edge_detection_algo_07(img, contour_constant_value, ps_x, ps_y):
    (contours_without_filter, constant) = get_max_contours(img, constant_value=contour_constant_value, ContourRetrievalMode=cv2.RETR_EXTERNAL)
    needle_allowed_area_mm2 = 10
    needle_contours = [contour for contour in contours_without_filter if (get_contour_area_mm2(contour, ps_x, ps_y) < needle_allowed_area_mm2)]
    return needle_contours
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
def algo_to_get_pixel_lines(dicom_dict, needle_lines = []):
    # type: (dicom_dict) -> (lt_ovoid, tandem, rt_ovoid)
    # Step 1. Use algo05 to get center point of inner contour
    last_z_in_step1 = sorted(dicom_dict['z'].keys())[0]
    center_pts_dict = {} # The following loop will use algo03 to figure L't Ovoid, R't Ovoid and half tandem
    for z in sorted(dicom_dict['z'].keys()):
        #contours = dicom_dict['z'][z]['output']['contours512']['algo03']
        contours = dicom_dict['z'][z]['output']['contours512']['algo05']
        #plot_with_contours(dicom_dict, z=z, algo_key='algo03')
        #plot_with_contours(dicom_dict, z=z, algo_key='algo05')
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
    #allowed_distance_mm = 4.5 # allowed distance when trace from bottom to tips
    allowed_distance_mm = 4.5  # allowed distance when trace from bottom to tips
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
                print('Bottom-half Tandem say break in loop with z = {}'.format(z))
                break
            prev_pt = ( center_pts_dict[z][1][0], center_pts_dict[z][1][1], float(z))
            prev_info['pt'] = prev_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            tandem.append(prev_pt)
            continue
        prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
        prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
        print('aa and idx_z = ',idx_z, flush= True)
        if ( len(center_pts_dict[z]) <= 1 ):
            # to prevent out of range of list
            continue
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
            # To process first slice when there are no no three inner contour
            min_z = sorted(dicom_dict['z'].keys())[0]
            ct_obj = dicom_dict['z'][min_z]
            ps_x = ct_obj['ps_x']
            ps_y = ct_obj['ps_y']
            a_mm = 1.5 # allowed distance mm
            a_px_x = a_mm / ps_x # allowed distance pixel for x-axis
            a_px_y = a_mm / ps_y # allowed distance pixel for y-axis

            l_pt = lt_ovoid[0]
            r_pt = rt_ovoid[0]
            m_pt = None # middle pt
            for pt in inner_cen_pts:
                #if pt[0] > l_pt[0] + 4 and pt[0] < r_pt[0] - 4:
                if pt[0] > l_pt[0] + a_px_x and pt[0] < r_pt[0] - a_px_x:
                    #if (pt[1] > min([l_pt[1], r_pt[1]]) - 4 and pt[1] < max([l_pt[1], r_pt[1]]) + 4 ):
                    if (pt[1] > min([l_pt[1], r_pt[1]]) - a_px_y and pt[1] < max([l_pt[1], r_pt[1]]) + a_px_y):
                        m_pt = pt
                        break
            if (m_pt == None):
                # Use algo02 to find tandem
                # And remember to avoid needle_lines[][0]

                min_z = sorted(dicom_dict['z'].keys())[0]
                ct_obj = dicom_dict['z'][min_z]
                ps_x = ct_obj['ps_x']
                ps_y = ct_obj['ps_y']

                #ct_obj['output']['algo02']
                potential_contours = []
                for contour in ct_obj['output']['contours512']['algo02']:
                    rect_info = get_rect_info_from_cv_contour(contour)
                    cen_pt = (rect_info[2][0], rect_info[2][1])
                    inner_rect_gap_allowed_mm = 8.4
                    i_mm = inner_rect_gap_allowed_mm
                    i_px_x = i_mm / ps_x
                    i_px_y = i_mm / ps_y
                    #if cen_pt[0] > max(l_pt[0], r_pt[0])-28 or cen_pt[0]  < min(l_pt[0], r_pt[0])+28:
                    if cen_pt[0] > max(l_pt[0], r_pt[0]) - i_px_x or cen_pt[0] < min(l_pt[0], r_pt[0]) + i_px_x:
                        continue
                    #if cen_pt[1] > max(l_pt[1], r_pt[1])-28 or cen_pt[1]  < min(l_pt[1], r_pt[1])+28:
                    if cen_pt[1] > max(l_pt[1], r_pt[1]) - i_px_y or cen_pt[1] < min(l_pt[1], r_pt[1]) + i_px_y:
                        continue
                    cen_pt_is_on_needle = False
                    for needle_line in needle_lines:
                        n_pt = needle_line[0]
                        gap_of_needle_allowed_distance_mm = 1
                        dist_to_needle_mm = math.sqrt( ((n_pt[0]-cen_pt[0])*ps_x)**2 + ( (n_pt[1] - cen_pt[1])*ps_y)**2 )
                        if dist_to_needle_mm < gap_of_needle_allowed_distance_mm:
                            cen_pt_is_on_needle = True
                            break
                    if cen_pt_is_on_needle == True:
                        continue
                    potential_contours.append(contour)
                if len(potential_contours) == 1:
                    contour = potential_contours[0]
                    rect_info = get_rect_info_from_cv_contour(contour)
                    m_pt = (rect_info[2][0], rect_info[2][1])
                    tandem.append((m_pt[0], m_pt[1], float(z)))
                else:
                    if len(potential_contours) == 0 and len(needle_lines) == 1:
                        # In this case needle should be remove and change the needle to tandem
                        only_needle_line = needle_lines[0]
                        m_pt = only_needle_line[0]
                        needle_lines.remove(only_needle_line)
                        tandem.append( (m_pt[0], m_pt[1], float(z)) )
                        pass
                    else:
                        raise Exception

            else:
                tandem.append((m_pt[0], m_pt[1], float(z)))
        else :
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
    print('z_idx of last_z={} is {}'.format(last_z, idx_z) )
    upper_half_z_idx_start = (z_idx + 1 )# upper_half_z_idx_start is the next z of last_z in current tandem data.
    print('upper_half_z_idx_start = {}'.format(upper_half_z_idx_start))
    upper_half_z_idx_end = len(dicom_dict['z'].keys())
    print('upper_half_z_idx_end = {}'.format(upper_half_z_idx_end))
    print('upper_half_z_idx [start,end) = [{},{}) '.format(upper_half_z_idx_start, upper_half_z_idx_end))
    z_start = sorted(dicom_dict['z'].keys())[upper_half_z_idx_start]
    z_end = sorted(dicom_dict['z'].keys())[upper_half_z_idx_end-1]+0.001
    print('and [start_z, end_z)  = [{},{})'.format(z_start,z_end))
    #print('upper_half_z_idx [start,end) = [{},{}) and z = [{},{})'.format(upper_half_z_idx_start, upper_half_z_idx_end, sorted(dicom_dict['z'].keys())[upper_half_z_idx_start],  sorted(dicom_dict['z'].keys())[upper_half_z_idx_end]  ))


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
        if (z == -34):
            print('z is -34 arrive')
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
            cen_pt = (rect_info[2][0], rect_info[2][1])
            cen_pts.append(cen_pt)
        # Step 6.3.3. Find closed center point for these center pt. And append it into tandem line. But leave looping if there is no center pt
        # prev_info is like {'pt': (240, 226, -92.0), 'ps_x': "3.90625e-1", 'ps_y': "3.90625e-1"}
        # pt in cen_pts is like (240, 226)
        minimum_distance_mm = allowed_distance_mm + 1  # If minimum_distance_mm is finally large than allowed_distance_mm, it's mean there is no pt closed to prev_pt
        minimum_pt = (0, 0)
        print('cen_pts = {}'.format(cen_pts ))
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
            print('more than tip ')
            break
        else:
            tandem.append( (minimum_pt[0], minimum_pt[1],float(z)) )
            prev_info['pt'] = minimum_pt
            prev_info['ps_x'] = ps_x
            prev_info['ps_y'] = ps_y
            print('tandem = {}'.format(tandem))
    return (lt_ovoid, tandem, rt_ovoid)
def algo_to_get_needle_lines(dicom_dict):
    needle_lines = []
    # Step 1. Use algo07 to get center point of inner contour
    last_z_in_step1 = sorted(dicom_dict['z'].keys())[0]
    center_pts_dict = {}
    for z in sorted(dicom_dict['z'].keys()):
        contours = dicom_dict['z'][z]['output']['contours512']['algo07']
        #plot_with_contours(dicom_dict, z=z, algo_key='algo07')
        center_pts_dict[z] = []
        for contour in contours:
            rect_info = get_rect_info_from_cv_contour(contour)
            cen_pt = ( rect_info[2][0], rect_info[2][1] )
            center_pts_dict[z].append(cen_pt)
        center_pts_dict[z].sort(key=lambda pt:pt[0])
        print('center_pts_dict[{}] = {}'.format(z, center_pts_dict[z]))

    print('STEP 2.')

    min_z = sorted(center_pts_dict.keys())[0]
    allowed_distance_mm = 2.5 # allowed distance when trace from bottom to tips of L't Ovoid
    # Get first slice and see how many needle point in it. the index of needle point in first slice will be the needle_line_idx
    for needle_line_idx in range(len(center_pts_dict[min_z])):
        print('needle_line_idx = {}'.format(needle_line_idx))
        needle_line = []
        prev_info = {}
        prev_info['pt'] = None
        prev_info['ps_x'] = None
        prev_info['ps_y'] = None
        for idx_z, z in enumerate(sorted(center_pts_dict.keys())):
            ps_x = dicom_dict['z'][z]['ps_x']
            ps_y = dicom_dict['z'][z]['ps_y']
            if idx_z == 0:
                #prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
                prev_pt = (center_pts_dict[z][needle_line_idx][0], center_pts_dict[z][needle_line_idx][1], float(z))
                prev_info['pt'] = prev_pt
                prev_info['ps_x'] = ps_x
                prev_info['ps_y'] = ps_y
                #lt_ovoid.append(prev_pt)
                needle_line.append(prev_pt)
                continue
            prev_x_mm = prev_info['pt'][0] * prev_info['ps_x']
            prev_y_mm = prev_info['pt'][1] * prev_info['ps_y']
            #x_mm = center_pts_dict[z][0][0] * ps_x
            if( needle_line_idx >= len(center_pts_dict[z]) ):
                # It's mean there is no more point, so continue
                continue
            x_mm = center_pts_dict[z][needle_line_idx][0] * ps_x #Error
            #y_mm = center_pts_dict[z][0][1] * ps_y
            y_mm = center_pts_dict[z][needle_line_idx][1] * ps_y
            if math.sqrt( (x_mm-prev_x_mm)**2 + (y_mm-prev_y_mm)**2) < allowed_distance_mm:
                #prev_pt = ( center_pts_dict[z][0][0], center_pts_dict[z][0][1], float(z))
                prev_pt = (center_pts_dict[z][needle_line_idx][0], center_pts_dict[z][needle_line_idx][1], float(z))
                prev_info['pt'] = prev_pt
                prev_info['ps_x'] = ps_x
                prev_info['ps_y'] = ps_y
                #lt_ovoid.append(prev_pt)
                needle_line.append(prev_pt)
                print('needle_line (with idx={})  = {}'.format(needle_line_idx, needle_line))
            else:
                break
        needle_lines.append(needle_line)
    return needle_lines
def get_applicator_rp_line(metric_line, first_purpose_distance_mm, each_purpose_distance_mm):
    if (len(metric_line) == 0):
        return []
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
            print('Exception in get_metric_pt() of get_applicator_rp_line()')
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
def wrap_to_rp_file(RP_OperatorsName, rs_filepath, tandem_rp_line, out_rp_filepath, lt_ovoid_rp_line, rt_ovoid_rp_line, needle_rp_lines=[], applicator_roi_dict={}):
    # TODO wrap needles
    print('len(needle_rp_lines)={}'.format(len(needle_rp_lines)))
    # rp_template_filepath = r'RP_Template/Brachy_RP.1.2.246.352.71.5.417454940236.2063186.20191015164204.dcm'
    # rp_template_filepath = r'RP_Template_Brachy_24460566_implant-5_20191113/RP.1.2.246.352.71.5.417454940236.2060926.20191008103753.dcm'
    rp_template_filepath = r'RP_Template_34135696_20191115/RP.1.2.246.352.71.5.417454940236.2077416.20191115161213.dcm'
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
        try:
            val = getattr(rs_fp, attr)
            setattr(rp_fp, attr, val)
        except Exception as ex:
            print('Error is happend in for attr in directAttrSet. Sometimes RS file is out of control')
            print(ex)
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
    rp_lines = rp_lines + needle_rp_lines

    for idx, rp_line in enumerate(rp_lines):
        print('rp_line[{}] = {}'.format(idx, rp_line))


    #TODO rp_Ref_ROI_Numbers need to match to current RS's ROI number of three applicators
    #rp_Ref_ROI_Numbers = [17, 18, 19]
    #rp_Ref_ROI_Numbers = app_roi_num_list

    #enablePrint()
    SortedAppKeys = sorted(applicator_roi_dict.keys())
    app_roi_num_list = []
    print('mapping of [ROI Name => ROI Number]')
    for applicator_roi_name in SortedAppKeys:
        print('{}->{}'.format(applicator_roi_name, applicator_roi_dict[applicator_roi_name]))
        app_roi_num_list.append(applicator_roi_dict[applicator_roi_name])
    print('app_roi_num_list = {}'.format(app_roi_num_list))
    #rp_Ref_ROI_Numbers = sorted(app_roi_num_list, reverse=True)
    rp_Ref_ROI_Numbers = app_roi_num_list
    print('rp_Ref_ROI_Numbers = {}'.format(rp_Ref_ROI_Numbers))
    #blockPrint()
    rp_ControlPointRelativePositions = [3.5, 3.5, 3.5] # After researching, all ControlPointRelativePositions is start in 3.5
    rp_ControlPointRelativePositions = [3.5 for item in app_roi_num_list]

    #enablePrint()
    print('Dr. Wang debug message')
    for idx, rp_line in enumerate(rp_lines):
        print('\nidx={} -> rp_line = ['.format(idx))
        for pt in rp_line:
            print('\t, {}'.format(pt))
    #blockPrint()
    for idx,rp_line in enumerate(rp_lines):
        if (False and  len(needle_rp_lines) == 0):
            enablePrint()
            print('Case without needles')
            if (idx >= 3):
                break
            blockPrint()

        if (False and idx >= 1): #OneTandem
            enablePrint()
            print('Debug importing RP by only tandem')
            rp_fp.ApplicationSetupSequence[0].ChannelSequence = copy.deepcopy(rp_fp.ApplicationSetupSequence[0].ChannelSequence[0:1])
            blockPrint()
            break
        if (idx >= len(rp_Ref_ROI_Numbers)):
            print('the number of rp_line is larger than len(rp_Ref_ROI_Numbers)')
            break
        if (idx >= len(rp_fp.ApplicationSetupSequence[0].ChannelSequence) ):
            print('the number of rp_line is larger than len(rp_fp.ApplicationSetupSequence[0].ChannelSequence)')
            break
        # Change ROINumber of RP_Template_TestData RS into output RP output file
        # Do  I need to fit ROINumber in RS or not? I still have no answer
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].ReferencedROINumber = rp_Ref_ROI_Numbers[idx]
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].NumberOfControlPoints = len(rp_line)
        rp_fp.ApplicationSetupSequence[0].ChannelSequence[idx].BrachyControlPointSequence.clear()
        for pt_idx, pt in enumerate(rp_line):
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
def get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid):
    #(metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid)
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
    return (metric_lt_ovoid, metric_tandem, metric_rt_ovoid)
def get_metric_needle_lines_representation(dicom_dict, needle_lines):
    metric_needle_lines = []
    for line in needle_lines:
        metric_needle_line = []
        for pt in line:
            z = pt[2]
            ct_obj = dicom_dict['z'][z]
            x = pt[0] * ct_obj['ps_x'] + ct_obj['origin_x']
            y = pt[1] * ct_obj['ps_y'] + ct_obj['origin_y']
            metric_needle_line.append([x, y, z])
        metric_needle_lines.append(metric_needle_line)
    return metric_needle_lines

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

    # Set metadata for ROINumber list (for wrap rp data)
    rs_fp = pydicom.read_file(rs_filepath)
    if (rs_fp != None):

        # applicator_target_list = ['Applicator1', 'Applicator2', 'Applicator3']
        # Process to get applicator_target_list
        applicator_target_list = []
        for item in rs_fp.StructureSetROISequence:
            if 'Applicator' in item.ROIName:
                applicator_target_list.append(item.ROIName)


        applicator_roi_dict = {}
        for app_name in applicator_target_list:
            for item in rs_fp.StructureSetROISequence:
                if (item.ROIName == app_name):
                    applicator_roi_dict[app_name] = item.ROINumber
                    break
        #print('\napplicator_roi_dict = {}'.format(applicator_roi_dict))
        #display(applicator_roi_dict)
        #print(applicator_roi_dict.values())
        roi_num_list = [int(num) for num in applicator_roi_dict.values()]
        #print(roi_num_list)
        out_dict['metadata']['applicator123_roi_numbers'] = roi_num_list.copy()
        out_dict['metadata']['applicator_roi_dict'] = applicator_roi_dict

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

    # Figure out global_max_contour_constant_value
    A = dicom_dict['z'][ sorted(dicom_dict['z'].keys())[0] ]['rescale_pixel_array']
    data = A.ravel()
    sorted_data = np.copy(data)
    sorted_data.sort()
    global_max_contour_constant_value = sorted_data[-20] - 100
    metadata['global_max_contour_constant_value'] = global_max_contour_constant_value

    # metadata['view_scope'] = (view_min_y, view_max_y, view_min_x, view_max_x)
    #(contours_without_filter, constant) = get_max_contours(img, ContourRetrievalMode=cv2.RETR_TREE)

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
    global_max_contour_constant_value = ct_obj['dicom_dict']['metadata']['global_max_contour_constant_value']

    #view_pixel_array = rescale_pixel_array[view_min_y:view_max_y, view_min_x:view_max_x]
    img = ct_obj['rescale_pixel_array']
    gray_img = convert_to_gray_image(img)
    gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
    img = img[view_min_y: view_max_y, view_min_x:view_max_x]
    rescale_pixel_array = ct_obj['rescale_pixel_array']
    rescale_pixel_array = rescale_pixel_array[view_min_y: view_max_y, view_min_x:view_max_x]
    filter_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    ps_x = ct_obj['ps_x']
    ps_y = ct_obj['ps_y']
    ct_obj['output']['contours'] = {}
    ct_obj['output']['contours']['algo01'] = get_contours_from_edge_detection_algo_01(img, filter_img)
    ct_obj['output']['contours']['algo02'] = get_contours_from_edge_detection_algo_02(img, filter_img)
    ct_obj['output']['contours']['algo03'] = get_contours_from_edge_detection_algo_03(img)
    ct_obj['output']['contours']['algo04'] = get_contours_from_edge_detection_algo_04(img)
    contour_constant_value = ct_obj['dicom_dict']['metadata']['global_max_contour_constant_value']
    ct_obj['output']['contours']['algo05'] = get_contours_from_edge_detection_algo_05(rescale_pixel_array, contour_constant_value)
    ct_obj['output']['contours']['algo06'] = get_contours_from_edge_detection_algo_06(rescale_pixel_array, contour_constant_value)
    ct_obj['output']['contours']['algo07'] = get_contours_from_edge_detection_algo_07(rescale_pixel_array, contour_constant_value, ps_x, ps_y)

    #ct_obj['output']['contours']['algo05'] = get_contours_from_edge_detection_algo_05(img, contour_constant_vlaue = global_max_contour_constant_value)
    #ct_obj['output']['contours']['algo06'] = get_contours_from_edge_detection_algo_06(img, contour_constant_value = global_max_contour_constant_value)


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

# FUNCTIONS - main genearte function
def generate_needle_contours_infos_to_dicom_dict(dicom_dict):
    # Process to make needle_contours_infos
    # Step 1. find out the needle contour that focus on most light part
    for z in sorted(dicom_dict['z'].keys()):
        ct_obj = dicom_dict['z'][z]
        #needle_contours_infos = [info for info in ct_obj['output']['contours_infos']['algo04'] if (info['area_mm2'] < 10) ]
        needle_contours_infos = [info for info in ct_obj['output']['contours_infos']['algo06'] if(info['area_mm2'] < 10)]
        ct_obj['output']['needle_contours_infos'] = copy.deepcopy(needle_contours_infos)
        print('len(dicom_dict["z"][{}]["output"]["needle_contours_infos"]) = {}'.format(z, len(dicom_dict['z'][z]['output']['needle_contours_infos'])))
    # Step 2. pick up  15px * 15 px picture for each light point
    for z in sorted(dicom_dict['z'].keys()):
        ct_obj = dicom_dict['z'][z]
        ps_x = ct_obj['ps_x']
        ps_y = ct_obj['ps_y']
        for info in ct_obj['output']['needle_contours_infos']:
            print('({}, {}, {})'.format(info['mean'][0], info['mean'][1], z))
            mean_x = info['mean'][0]
            mean_y = info['mean'][1]
            rescale_pixel_array = ct_obj['rescale_pixel_array']
            x_min = mean_x - 7
            x_max = mean_x + 7
            y_min = mean_y - 7
            y_max = mean_y + 7
            if x_min < 0:
                x_min = 0
            if x_max >= 512:
                x_max = 512
            if y_min < 0:
                y_min = 0
            if y_max >= 512:
                y_max = 512
            #pick_picture = rescale_pixel_array[x_min:x_max, y_min:y_max]
            pick_picture = rescale_pixel_array[y_min:y_max, x_min:x_max]
            info['pick_picture'] = pick_picture
            img = pick_picture
            gray_img = convert_to_gray_image(img)
            filter_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
            contours = get_max_contours_by_filter_img(img, filter_img, ContourRetrievalMode=cv2.RETR_EXTERNAL)
            info['pick_picture_contours'] = contours
            info['pick_picture_contour_area_mm2s'] = [(cv2.contourArea(contour) * ps_x * ps_y) for contour in contours]




            #pick_area_mm2 = cv2.contourArea(contour) * ps_x * ps_y
def generate_brachy_rp_file(RP_OperatorsName, dicom_dict, out_rp_filepath, is_enable_print=False):
    if (is_enable_print == False):
        blockPrint()
    else:
        enablePrint()
    # Step 1. Get line of lt_ovoid, tandem, rt_ovoid by OpneCV contour material and innovated combination
    needle_lines = algo_to_get_needle_lines(dicom_dict)
    print('len(needle_lines) = {}'.format(len(needle_lines)))
    if len(needle_lines) > 0 :
        for idx, needle_line in enumerate(needle_lines):
            #print('needle_lines[0] = {}'.format(needle_lines[0]))
            print('needle_lines[{}] = {}'.format(idx, needle_lines[idx]))

    (lt_ovoid, tandem, rt_ovoid) = algo_to_get_pixel_lines(dicom_dict, needle_lines)

    # Step 2. Convert line into metric representation
    # Original line is array of (x_px, y_px, z_mm) and we want to convert to (x_mm, y_mm, z_mm)
    (metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid)
    print('metric_lt_ovoid = {}'.format(metric_lt_ovoid))
    print('metric_tandem = {}'.format(metric_tandem))
    print('metric_rt_ovoid = {}'.format(metric_rt_ovoid))


    metric_needle_lines = get_metric_needle_lines_representation(dicom_dict, needle_lines)

    # Step 2.1 Extend metric needle line with 2mm in the end point of metric_needle_lines
    needle_extend_mm = 2
    for line_idx, line in enumerate(metric_needle_lines):
        pt_s = line[0] # point start
        pt_e = line[-1] # point end
        pt_n = pt_e.copy() # point new. it will append in end of line
        cur_dist = math.sqrt( (pt_e[0]-pt_s[0])**2 + (pt_e[1]-pt_s[1])**2 + (pt_e[2]-pt_s[2])**2 )
        for i in range(3):
            pt_n[i] = pt_n[i] + ( (pt_e[i] - pt_s[i]) * (needle_extend_mm / cur_dist) )
        line.append(pt_n)

    print('len(metric_needle_lines) = {}'.format(len(metric_needle_lines)))
    for line_idx, line in enumerate(metric_needle_lines):
        print('metric_needle_lines[{}]= {}'.format(line_idx, line))

    # Step 3. Reverse Order, so that first element is TIPS [from most top (z maximum) to most bottom (z minimum) ]
    metric_lt_ovoid.reverse()
    metric_tandem.reverse()
    metric_rt_ovoid.reverse()
    for metric_line in metric_needle_lines:
        metric_line.reverse()

    # Step 4. Get Applicator RP line
    #tandem_rp_line = get_applicator_rp_line(metric_tandem, 4, 5)

    # for debug , so change about testing rp import correct or not. So change tandem start from 3mm to 13mm
    tandem_rp_line = get_applicator_rp_line(metric_tandem, 3, 5) # <-- change to reduce 1mm
    #tandem_rp_line = get_applicator_rp_line(metric_tandem, 13, 5)  # <-- change to reduce 1mm

    lt_ovoid_rp_line = get_applicator_rp_line(metric_lt_ovoid, 0, 5)
    rt_ovoid_rp_line = get_applicator_rp_line(metric_rt_ovoid, 0 ,5)
    rp_needle_lines = []
    for metric_line in metric_needle_lines:
        rp_needle_line = get_applicator_rp_line(metric_line, 0, 5)
        rp_needle_lines.append(rp_needle_line)

    print('lt_ovoid_rp_line = {}'.format(lt_ovoid_rp_line))
    print('tandem_rp_line = {}'.format(tandem_rp_line))
    print('rt_ovoid_rp_line = {}'.format(rt_ovoid_rp_line))
    print('len(rp_needle_lines) = {}'.format(len(rp_needle_lines)))
    for line_idx, line in enumerate(rp_needle_lines):
        print('rp_needle_lines[{}]= {}'.format(line_idx, line))

    # Step 5. Wrap to RP file
    # TODO for wrap rp_needle_lines into RP file
    print(dicom_dict['pathinfo']['rs_filepath'])
    print(dicom_dict['metadata'].keys())
    #print(pydicom.read_file(dicom_dict['pathinfo']['rs_filepath']).keys())
    rs_filepath = dicom_dict['pathinfo']['rs_filepath']
    print('out_rp_filepath = {}'.format(out_rp_filepath))

    applicator_roi_dict = dicom_dict['metadata']['applicator_roi_dict']
    # TODO will change the wrap_to_rp_file function, because we will wrap needle information into RP files
    #wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line, out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, rt_ovoid_rp_line=rt_ovoid_rp_line, app_roi_num_list=app_roi_num_list)
    #wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line,out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, needle_rp_lines=rp_needle_lines,rt_ovoid_rp_line=rt_ovoid_rp_line, app_roi_num_list=app_roi_num_list)
    wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line,
                    out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, needle_rp_lines=rp_needle_lines,
                    rt_ovoid_rp_line=rt_ovoid_rp_line, applicator_roi_dict=applicator_roi_dict)
    if (is_enable_print == False):
        enablePrint()
def generate_brachy_rp_file_without_needle(RP_OperatorsName, dicom_dict, out_rp_filepath, is_enable_print=False):
    if (is_enable_print == False):
        blockPrint()
    else:
        enablePrint()
    enablePrint()
    print('Call generate_brachy_rp_file_without_needle()')
    blockPrint()
    # Step 1. Get line of lt_ovoid, tandem, rt_ovoid by OpneCV contour material and innovated combination
    (lt_ovoid, tandem, rt_ovoid) = algo_to_get_pixel_lines(dicom_dict)
    # Step 2. Convert line into metric representation
    # Original line is array of (x_px, y_px, z_mm) and we want to convert to (x_mm, y_mm, z_mm)
    (metric_lt_ovoid, metric_tandem, metric_rt_ovoid) = get_metric_lines_representation(dicom_dict, lt_ovoid, tandem, rt_ovoid)
    print('metric_lt_ovoid = {}'.format(metric_lt_ovoid))
    print('metric_tandem = {}'.format(metric_tandem))
    print('metric_rt_ovoid = {}'.format(metric_rt_ovoid))

    # Step 3. Reverse Order, so that first element is TIPS [from most top (z maximum) to most bottom (z minimum) ]
    metric_lt_ovoid.reverse()
    metric_tandem.reverse()
    metric_rt_ovoid.reverse()

    # Step 4. Get Applicator RP line
    #tandem_rp_line = get_applicator_rp_line(metric_tandem, 4, 5)

    # for debug , so change about testing rp import correct or not. So change tandem start from 3mm to 13mm
    #tandem_rp_line = get_applicator_rp_line(metric_tandem, 3, 5) # <-- change to reduce 1mm
    tandem_rp_line = get_applicator_rp_line(metric_tandem, 13, 5)  # <-- change to reduce 1mm
    lt_ovoid_rp_line = get_applicator_rp_line(metric_lt_ovoid, 0, 5)
    rt_ovoid_rp_line = get_applicator_rp_line(metric_rt_ovoid, 0 ,5)
    print('lt_ovoid_rp_line = {}'.format(lt_ovoid_rp_line))
    print('tandem_rp_line = {}'.format(tandem_rp_line))
    print('rt_ovoid_rp_line = {}'.format(rt_ovoid_rp_line))

    # Step 5. Wrap to RP file
    # TODO for wrap rp_needle_lines into RP file
    print(dicom_dict['pathinfo']['rs_filepath'])
    print(dicom_dict['metadata'].keys())
    #print(pydicom.read_file(dicom_dict['pathinfo']['rs_filepath']).keys())

    rs_filepath = dicom_dict['pathinfo']['rs_filepath']

    print('out_rp_filepath = {}'.format(out_rp_filepath))
    app_roi_num_list = dicom_dict['metadata']['applicator123_roi_numbers']
    # TODO will change the wrap_to_rp_file function, because we will wrap needle information into RP files
    #wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line, out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, rt_ovoid_rp_line=rt_ovoid_rp_line, app_roi_num_list=app_roi_num_list)
    #wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line,out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, needle_rp_lines=rp_needle_lines,rt_ovoid_rp_line=rt_ovoid_rp_line, app_roi_num_list=app_roi_num_list)
    wrap_to_rp_file(RP_OperatorsName=RP_OperatorsName, rs_filepath=rs_filepath, tandem_rp_line=tandem_rp_line,out_rp_filepath=out_rp_filepath, lt_ovoid_rp_line=lt_ovoid_rp_line, needle_rp_lines=[], rt_ovoid_rp_line=rt_ovoid_rp_line, app_roi_num_list=app_roi_num_list)
    if (is_enable_print == False):
        enablePrint()


# FUNCTIONS - Some ploting utility functions support for you to check CT pictures with data
def generate_all_rp_process(
        root_folder=r'RAL_plan_new_20190905', rp_output_folder_filepath='all_rp_output',  bytes_dump_folder_filepath='contours_bytes',
        is_recreate_bytes=True, debug_folders=[]):
    print('Call generate_all_rp_process with the following arguments')
    print('root_folder = ', root_folder)
    print('rp_output_folder_filepath = ', rp_output_folder_filepath)
    print('bytes_dump_folder_filepath = ', bytes_dump_folder_filepath)


    create_directory_if_not_exists(bytes_dump_folder_filepath)
    create_directory_if_not_exists(rp_output_folder_filepath)

    print('[START] generate_all_rp_process()')
    all_dicom_dict = {}
    # Step 2. Generate all our target
    #root_folder = r'RAL_plan_new_20190905'
    f_list = [ os.path.join(root_folder, file) for file in os.listdir(root_folder) ]
    folders = os.listdir(root_folder)
    total_folders = []
    failed_folders = []
    success_folders = []
    sorted_f_list = copy.deepcopy(sorted(f_list))

    #for folder_idx, folder in enumerate(sorted_f_list):
    #    print(folder_idx, folder)
    #    print('[{}/{}] Loop info : folder_idx = {}, folder = {}'.format(folder_idx + 1, len(folders), folder_idx, folder),flush=True)

    #for folder_idx, folder in enumerate(sorted(f_list)):
    for folder_idx, folder in enumerate(sorted_f_list):
        enablePrint()
        #if (os.path.basename(folder) not in ['21569696', '33220132']):
        #    continue
        #if (os.path.basename(folder) not in ['21569696']):
        #    continue
        #if (os.path.basename(folder) not in ['487961']): # One Needle case
        #    continue
        #if (os.path.basename(folder) not in ['34982640']):
        #    continue
        #if (os.path.basename(folder) not in ['24460566-2']):
        #    continue
        if len(debug_folders) != 0:
            if (os.path.basename(folder) not in debug_folders):
                continue

        print('\n[{}/{}] Loop info : folder_idx = {}, folder = {}'.format(folder_idx + 1, len(folders), folder_idx, folder),flush=True)
        byte_filename = r'{}.bytes'.format(os.path.basename(folder))
        #dump_filepath = os.path.join('contours_bytes', byte_filename)
        dump_filepath = os.path.join(bytes_dump_folder_filepath, byte_filename)

        if (is_recreate_bytes == True):
            time_start = datetime.datetime.now()
            print('[{}/{}] Create bytes file {} '.format(folder_idx + 1, len(folders), dump_filepath), end=' -> ',flush=True)
            dicom_dict = get_dicom_dict(folder)
            generate_metadata_to_dicom_dict(dicom_dict)
            generate_output_to_dicom_dict(dicom_dict)
            all_dicom_dict[folder] = dicom_dict
            python_object_dump(dicom_dict, dump_filepath)
            time_end = datetime.datetime.now()
            print('{}s [{}-{}]'.format(time_end - time_start, time_start, time_end), end='\n', flush=True)
        else: # CASE is_recreate_bytes == False
            bytes_filepath = os.path.join(bytes_dump_folder_filepath, r'{}.bytes'.format(os.path.basename(folder)))
            bytes_file_exists = os.path.exists(bytes_filepath)
            if bytes_file_exists == True:
                #dicom_dict = python_object_load(bytes_filepath)
                #all_dicom_dict[folder] = dicom_dict
                print('[{}/{}] File have been created - {}'.format(folder_idx + 1, len(folders), dump_filepath), flush=True)
            else: #CASE When the file is not exist in bytes_filepath
                time_start = datetime.datetime.now()
                print('[{}/{}] Create bytes file {} '.format(folder_idx + 1, len(folders), dump_filepath), end=' -> ',flush=True)
                dicom_dict = get_dicom_dict(folder)
                generate_metadata_to_dicom_dict(dicom_dict)
                generate_output_to_dicom_dict(dicom_dict)
                all_dicom_dict[folder] = dicom_dict
                python_object_dump(dicom_dict, dump_filepath)
                time_end = datetime.datetime.now()
                print('{}s [{}-{}]'.format(time_end - time_start, time_start, time_end), end='\n', flush=True)
        # Change to basename of folder here
        fullpath_folder = folder
        folder = os.path.basename(folder)
        total_folders.append(folder)
        try:
            #bytes_filepath = os.path.join('contours_bytes', r'{}.bytes'.format(folder))
            bytes_filepath = os.path.join(bytes_dump_folder_filepath, r'{}.bytes'.format(folder))
            dicom_dict = python_object_load(bytes_filepath)

            if fullpath_folder not in all_dicom_dict.keys():
                all_dicom_dict[fullpath_folder] = dicom_dict

            metadata = dicom_dict['metadata']
            # out_rp_filepath format is PatientID, RS StudyDate  and the final is folder name processing by coding


            out_rp_filepath = r'RP.{}.{}.f{}.dcm'.format(  metadata['RS_PatientID'],  metadata['RS_StudyDate'],  os.path.basename(metadata['folder']) )
            #out_rp_filepath = os.path.join('all_rp_output', out_rp_filepath)
            out_rp_filepath = os.path.join(rp_output_folder_filepath, out_rp_filepath)
            time_start = datetime.datetime.now()
            print('[{}/{}] Create RP file -> {}'.format(folder_idx+1,len(folders), out_rp_filepath) ,end=' -> ', flush=True)
            #generate_brachy_rp_file_without_needle(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath,is_enable_print=False)
            generate_brachy_rp_file(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath, is_enable_print=False)
            #generate_brachy_rp_file(RP_OperatorsName='cylin', dicom_dict=dicom_dict, out_rp_filepath=out_rp_filepath, is_enable_print=True)
            time_end = datetime.datetime.now()
            print('{}s [{}-{}]'.format(time_end-time_start, time_start, time_end), end='\n', flush=True)
            success_folders.append(folder)

            # Added code for output RS file and output RP file into folder by each patient'study case
            import_root_folder = r'import_output'
            import_folder_filepath = os.path.join(import_root_folder, r'{}.{}.f{}'.format(metadata['RS_PatientID'],  metadata['RS_StudyDate'],  os.path.basename(metadata['folder'])))
            create_directory_if_not_exists(import_folder_filepath)
            src_rp_filepath = out_rp_filepath
            dst_rp_filepath = os.path.join(import_folder_filepath, os.path.basename(out_rp_filepath))
            print('src_rp_filepath = {}'.format(src_rp_filepath))
            print('dst_rp_filepath = {}'.format(dst_rp_filepath))
            copyfile(src_rp_filepath, dst_rp_filepath)
            src_rs_filepath = dicom_dict['pathinfo']['rs_filepath']
            dst_rs_filepath = os.path.join(import_folder_filepath, os.path.basename(src_rs_filepath))
            print('src_rs_filepath = {}'.format(src_rs_filepath))
            print('dst_rs_filepath = {}'.format(dst_rs_filepath))
            copyfile(src_rs_filepath, dst_rs_filepath)
        except Exception as debug_ex:
            print('Create RP file Failed')
            failed_folders.append(folder)
            print(debug_ex)
    print('FOLDER SUMMARY REPORT')
    print('failed folders = {}'.format(failed_folders))
    print('failed / total = {}/{}'.format(len(failed_folders), len(total_folders) ))
    print('success /total = {}/{}'.format(len(success_folders), len(total_folders) ))

    # Step 3. Use python_object_dump to dump it into some file
    try:
        print('Creating {} in very largest size'.format(filename))
        python_object_dump(all_dicom_dict, filename)
        print('Created {}'.format(filename))
    except Exception as ex:
        print('Create largest size dicom file failed')
    print('[END] generate_all_rp_process()')


if __name__ == '__main__':

    # 10 CASE
    print('root_folder = Study-RAL-implant_20191112 -> {}'.format([os.path.basename(item) for item in os.listdir('Study-RAL-implant_20191112')]))
    generate_all_rp_process(root_folder=r'Study-RAL-implant_20191112',
                            rp_output_folder_filepath='Study-RAL-implant_20191112_RP_Files',bytes_dump_folder_filepath='Study-RAL-implant_20191112_Bytes_Files',
                            is_recreate_bytes=True, debug_folders=['24460566'])
    # 31 CASE
    #print('root_folder = RAL_plan_new_20190905 -> {}'.format([os.path.basename(item) for item in os.listdir('RAL_plan_new_20190905')]))
    #generate_all_rp_process(root_folder=r'RAL_plan_new_20190905',
    #                        rp_output_folder_filepath='RAL_plan_new_20190905_RP_Files', bytes_dump_folder_filepath='RAL_plan_new_20190905_Bytes_Files',
    #                        is_recreate_bytes=False, debug_folders=[])

    # 22 CASE : the case of 33220132 is only one tandem and not with pipe. This case should be wrong
    #print('root_folder = Study-RAL-20191105 -> {}'.format([os.path.basename(item) for item in os.listdir('Study-RAL-20191105')]))
    #generate_all_rp_process(root_folder=r'Study-RAL-20191105',
    #                        rp_output_folder_filepath='Study-RAL-20191105_RP_Files', bytes_dump_folder_filepath='Study-RAL-20191105_Bytes_Files',
    #                        is_recreate_bytes=False, debug_folders=[])



    exit(0)




