# Try to seperate program into clear verion and useful functions
import os
import pydicom
import numpy as np
import cv2
import copy
import math
from sys import exit

import openpyxl
import csv, codecs

from decimal import Decimal
import random


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

    # Generate contours infos like x,y mean and area_mm
    ct_obj['output']['contours_infos'] = {}
    ps_x = ct_obj['ps_x']
    ps_y = ct_obj['ps_y']
    for algo_key in (ct_obj['output']['contours'].keys()):
        contours = ct_obj['output']['contours'][algo_key]
        contours_infos = []
        for contour in contours:
            contours_info = {}
            contours_infos.append(contours_info)
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
    sheet.append([folder]) # idx = 0
    sheet.append([algo_key]) # idx = 1
    sheet.append(['z', 'contours#']) # idx = 2
    for z in sorted(z_map.keys()):
        ct_obj = z_map[z]
        contours_infos = ct_obj['output']['contours_infos'][algo_key]
        contour_num = len(contours_infos)
        row = [z, contour_num]
        sheet.append(row)
    sheet[0] = sheet[0] + ['']*(sheet_width-len(sheet[0]))
    sheet[1] = sheet[1] + ['']*(sheet_width-len(sheet[1]))
    header = sheet[2]
    for c_idx in range(sheet_width - 2):
        header.append('contour {}'.format(c_idx+1))

    for z_idx, z in enumerate(sorted(z_map.keys())):
        ct_obj = z_map[z]
        contours_infos = ct_obj['output']['contours_infos'][algo_key]
        write_infos = []
        infos = copy.deepcopy(ct_obj['output']['contours_infos'][algo_key])
        infos.sort(key=lambda info: info['mean'][0]) # sorting infos by mean x
        for info_idx, info in enumerate(infos):
            mean_x = info['mean'][0]
            mean_y = info['mean'][1]
            area_mm2 = info['area_mm2']
            write_info = '({},{})pixel - {}mm2'.format(mean_x, mean_y, area_mm2)
            write_infos.append(write_info)
        # Write infos into correct row_sheet
        sheet_row = sheet[z_idx+3]
        sheet_row = sheet_row + write_infos
        sheet_row = sheet_row + ['']*(sheet_width-len(sheet_row))
    print('Time to lookup sheet variable')






    for z in sorted(z_map.keys()):
        ct_obj = z_map[z]
        infos = copy.deepcopy(ct_obj['output']['contours_infos'][algo_key])
        infos.sort(key=lambda info: info['mean'][0]) # sorting infos by mean x
        for info_idx, info in enumerate(infos):
            pass



    pass



if __name__ == '__main__':
    root_folder = r'RAL_plan_new_20190905'
    f_list = [ os.path.join(root_folder, file) for file in os.listdir(root_folder) ]

    folder = f_list[0]
    #print_info_by_folder(folder)
    dicom_dict = get_dicom_dict(folder)
    generate_metadata_to_dicom_dict(dicom_dict)
    # Now it is support metadata. You can see it at  dicom_dict['metadata']
    # And you can generate its output
    #print(dicom_dict['metadata'])
    generate_output_to_dicom_dict(dicom_dict)

    # It's start to travel each output of ct_obj in dicom_dict
    z_map = dicom_dict['z']
    for z_idx,z in enumerate(sorted(z_map.keys())):
        ct_obj = z_map[z]
        contours_dict = ct_obj['output']['contours']
        for idx, algo_key in enumerate(sorted(contours_dict.keys())):
            contours = contours_dict[algo_key]
            for cidx, contour in enumerate(contours):
                area_mm2 = get_contour_area_mm2(contour, ct_obj['ps_x'], ct_obj['ps_y'])
                print('len={}, area_mm2 = {}'.format(len(contour), area_mm2))
            #print(len(contours))
    #generate_contour_number_csv_report(f_list, csv_filepath = 'contours.csv')
    #print('write done for contours.csv')

    folder = f_list[0]
    print(os.path.basename(folder))
    algo_keys = ['algo01', 'algo02', 'algo03', 'algo04']
    for algo_key in algo_keys:
        csv_filepath = '{}-{}.csv'.format(os.path.basename(folder), algo_key)
        generate_patient_mean_area_csv_report(folder, algo_key=algo_key, csv_filepath=csv_filepath)
        break

