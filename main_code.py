# Try to seperate program into clear verion and useful functions
import os,sys
import pydicom
import numpy as np
import cv2
import copy
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

def distance(pt1, pt2):
    import math
    axis_num = len(pt1)
    sum = 0.0
    for idx in range(axis_num):
        sum = sum + (pt1[idx] - pt2[idx]) ** 2
    ans = math.sqrt(sum)
    return ans

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

def run_and_make_rp(folder, out_rp_filepath):
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

run_and_make_rp(folder='RALmilo', out_rp_filepath=r'out.brachy.rp.withpoints.v04.dcm')






