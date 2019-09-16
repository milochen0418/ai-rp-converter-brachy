
import matplotlib.pyplot as plt
import os
import pydicom
import numpy as np
import cv2
import copy
import seaborn as sns
from numpy.random import randn
import matplotlib as mpl
from scipy import stats


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
def get_max_contours_by_filter_img(A, filter_img):
    # gray_image = cv2.cvtColor(filter_img, cv2.COLOR_RGB2GRAY)
    gray_image = filter_img
    # findContours
    # _, contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours
def get_max_contours(A, constant_value=None):
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
    _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # return contours (list of np.array) and constant (you assume they are almsot highest)
    return (contours, constant)
def get_rect_infos_and_center_pts(contours,h_min=13, w_min=13, h_max=19, w_max=19):
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
        #if h >= 13 and h < 19 and w >= 13 and h < 19:
        if h >= h_min and h < h_max and w >= w_min and w < w_max:
            cen_pt = [x_mean, y_mean]
            app_center_pts.append(cen_pt)
        else:
            #print('(h={},{} , w={},{})'.format(h_max, h_min, w_max, w_min))
            #print('Not matching ! rect_info = ', rect_info)
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
def show_diagram(np_array):
    data = np_array.ravel()  # reshape to 1 dimesion array
    sorted_data = np.copy(data)
    sorted_data.sort()
    print("major value = ", sorted_data[-10])
    # the_bins = int( (np.max(data) - np.min(data)) / (12*12) )
    the_bins = 64
    plt.hist(data, bins=the_bins, color=sns.desaturate("indianred", .8), alpha=.4)
    plt.show()
def get_app_center_pts_of_first_slice(first_slice_dict):
    ps_x = first_slice_dict['PixelSpacing_x']
    ps_y = first_slice_dict['PixelSpacing_y']
    h_max = int((19.0*4.19921e-1) / ps_y)
    h_min = int((13.0*4.19921e-1) / ps_y)
    w_max = int((19.0*4.19921e-1) / ps_x)
    w_min = int((13.0*4.19921e-1) / ps_x)
    #print('(h={},{} , w={},{})'.format(h_max, h_min, w_max, w_min))

    (contours, constant) = get_max_contours(first_slice_dict['rescale_pixel_array'])

    #(sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
    (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours, h_max=h_max,h_min=h_min, w_max=w_max,w_min=w_min)
    print('\n\n')
    print(sorted_app_center_pts)
    #TODO After researching done, write the code to finish this task

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
    h_max = int((19.0*4.19921e-1) / ps_y)
    h_min = int((13.0*4.19921e-1) / ps_y)
    w_max = int((19.0*4.19921e-1) / ps_x)
    w_min = int((13.0*4.19921e-1) / ps_x)
    print('(h={},{} , w={},{})'.format(h_max, h_min, w_max, w_min))

    #(sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
    (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours,h_max=h_max,w_max=w_max,h_min=h_min,w_min=w_min)
    print('sorted_app_center_pts = ',sorted_app_center_pts)

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
def get_batch_process_dict_v01(root_folder):
    process_dict = {}
    if not os.path.isdir(root_folder):
        return process_dict

    dataset_folder_list = []
    for patient_id_file in os.listdir(root_folder):
        patient_id_filepath = r"{}/{}".format(root_folder, patient_id_file)
        if not os.path.isdir(patient_id_filepath):
            continue
        for study_date_file in os.listdir(patient_id_filepath):
            study_date_filepath = r"{}/{}".format(patient_id_filepath, study_date_file)
            if not os.path.isdir(study_date_filepath):
                continue
            dataset_folder_list.append(study_date_filepath)
    for folder in dataset_folder_list:
        input_dict = {}
        # TODO in future if you need
        process_dict[folder] = input_dict
    return process_dict
def process_first_slice_with_folder(folder):
    print("\n\n\n")
    print("process_with_folder: ", folder)
    # folder = r"AI_RS_Compare_20190724\628821\0620"
    ct_filelist = get_ct_filelist_by_folder(folder)
    ct_dicom_dict = gen_ct_dicom_dict(ct_filelist)
    sorted_ct_dicom_dict_keys = sorted(ct_dicom_dict['SliceLocation'].keys())
    first_slice_dict = ct_dicom_dict['SliceLocation'][sorted_ct_dicom_dict_keys[0]]
    main_center_pts = get_app_center_pts_of_first_slice(first_slice_dict)
    print('main_center_pts = ', main_center_pts)
    if len(main_center_pts) != 3:
        print('')
        print('WRONG CASE ! It is not our expected data')
        print('')

    # show first_slice_dict's medical image
    (view_min_y, view_max_y, view_min_x, view_max_x) = get_view_scope_by_slice(first_slice_dict)

    slice_dict = first_slice_dict

    print('z = ', slice_dict['SliceLocation'], 'filepath = ', slice_dict['filepath'])
    img = slice_dict['rescale_pixel_array']
    gray_img = convert_to_gray_image(img)

    fig = plt.figure(figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')
    threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    plt.subplot(1, 4, 1)
    plt.imshow(threshed_im, cmap=plt.cm.bone)
    gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
    img = img[view_min_y: view_max_y, view_min_x:view_max_x]
    plt.subplot(1, 4, 2)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.subplot(1, 4, 3)
    plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -22)
    # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
    threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    filter_img = threshed_im

    contours = get_max_contours_by_filter_img(img, filter_img)
    (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
    proc_img = np.copy(img)

    for contour in contours:
        if len(contour) < 5:
            # You need at least 5 points in contour, so that you can use fitEllipse
            continue
        ellipse = cv2.fitEllipse(contour)  # auto-figure the ellipse to fit contour
        # print(ellipse)
        ellipse_poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                        (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
        # Because cv2.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta)
        # So The center point is (int(ellipse[0][0]), int(ellipse[0][1]))

        # reshape to format of contour, so that your can use draw contour function to draw it
        reshape_poly = ellipse_poly.reshape(ellipse_poly.shape[0], 1, ellipse_poly.shape[1])

        cv2.drawContours(proc_img, reshape_poly, -1, (255, 0, 0), 1)
        # cv2.drawContours(proc_img, contour, -1, (255,0,0),1)

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

        # if h >= 13 and h < 19 and w >= 13 and w < 19:
        if h >= 5 and h < 19 and w >= 5 and w < 19:
            # cen_pt = [x_mean, y_mean]
            # app_center_pts.append(cen_pt)
            # cen_pt = [int(ellipse[0][0]), int(ellipse[0][1])]
            # app_center_pts.append(cen_pt)
            pass
            # print('sorted_app_center_pts =',sorted_app_center_pts)
    plt.subplot(1, 4, 4)
    plt.imshow(proc_img, cmap=plt.cm.bone)
    plt.show()

    pass
def process_with_folder(folder):
    print("\n\n\n")
    print("process_with_folder: ", folder)
    # folder = r"AI_RS_Compare_20190724\628821\0620"

    ct_filelist = get_ct_filelist_by_folder(folder)
    ct_dicom_dict = gen_ct_dicom_dict(ct_filelist)

    sorted_ct_dicom_dict_keys = sorted(ct_dicom_dict['SliceLocation'].keys())
    first_slice_dict = ct_dicom_dict['SliceLocation'][sorted_ct_dicom_dict_keys[0]]

    (view_min_y, view_max_y, view_min_x, view_max_x) = get_view_scope_by_slice(first_slice_dict)
    # Processing data from view_

    # The way to use return value is
    # gray_img = gray_img[ view_min_y: view_max_y, view_min_x:view_max_x]

    # for z in sorted(ct_dicom_dict['SliceLocation'].keys()):
    zlines = []  # every element in zlines is a list that link to all circle component
    zlines.append([])
    zlines.append([])
    zlines.append([])

    prev_slice_dict = None
    for z in sorted_ct_dicom_dict_keys:
        slice_dict = ct_dicom_dict['SliceLocation'][z]
        print('z = ', z, 'filepath = ', slice_dict['filepath'])
        img = slice_dict['rescale_pixel_array']
        gray_img = convert_to_gray_image(img)

        fig = plt.figure(figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')

        threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)

        plt.subplot(1, 4, 1)
        # plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
        plt.imshow(threshed_im, cmap=plt.cm.bone)
        # plt.show()

        gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
        img = img[view_min_y: view_max_y, view_min_x:view_max_x]

        plt.subplot(1, 4, 2)
        plt.imshow(img, cmap=plt.cm.bone)
        # plt.show()

        # show_diagram(img)
        plt.subplot(1, 4, 3)
        plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        # show_diagram(gray_img)

        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -22)
        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
        threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
        filter_img = threshed_im

        # contours = get_max_contours_by_filter_img(first_slice_dict['rescale_pixel_array'])
        contours = get_max_contours_by_filter_img(img, filter_img)
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
        proc_img = np.copy(img)

        # app_center_pts.append(cen_pt)

        for contour in contours:
            if len(contour) < 5:
                # You need at least 5 points in contour, so that you can use fitEllipse
                continue
            ellipse = cv2.fitEllipse(contour)  # auto-figure the ellipse to fit contour
            # print(ellipse)
            ellipse_poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                            (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360,
                                            5)
            # Because cv2.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta)
            # So The center point is (int(ellipse[0][0]), int(ellipse[0][1]))

            # reshape to format of contour, so that your can use draw contour function to draw it
            reshape_poly = ellipse_poly.reshape(ellipse_poly.shape[0], 1, ellipse_poly.shape[1])

            cv2.drawContours(proc_img, reshape_poly, -1, (255, 0, 0), 1)

            # cv2.drawContours(proc_img, contour, -1, (255,0,0),1)

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

            # if h >= 13 and h < 19 and w >= 13 and w < 19:
            if h >= 5 and h < 19 and w >= 5 and w < 19:
                # cen_pt = [x_mean, y_mean]
                # app_center_pts.append(cen_pt)
                # cen_pt = [int(ellipse[0][0]), int(ellipse[0][1])]
                # app_center_pts.append(cen_pt)
                pass

        print('sorted_app_center_pts =', sorted_app_center_pts)
        # print('app_center_pts = ', app_center_pts)
        prev_slice_dict = slice_dict
        plt.subplot(1, 4, 4)
        plt.imshow(proc_img, cmap=plt.cm.bone)
        # plt.show()
        plt.show()

    pass
def example_show_all_slice():
    process_dict = get_batch_process_dict(r"AI_RS_Compare_20190724")
    for i in sorted(process_dict.keys()):
        if i == r"AI_RS_Compare_20190724/35086187/0613":
            continue
        process_with_folder(i)
    pass
def example_first_slice():
    process_dict = get_batch_process_dict(r"AI_RS_Compare_20190724")
    for i in sorted(process_dict.keys()):
        if i == r"AI_RS_Compare_20190724/35086187/0613":
            continue
        # if i == r"AI_RS_Compare_20190724/30640451/0711":
        #    print('ignore the wrong case of process_first_slice_with_folder with i = ', i)
        #    continue
        process_first_slice_with_folder(i)
    pass

def distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
def get_most_closed_pt(src_pt, pts, allowed_distance=1000):
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
def pure_show_slice_dict(slice_dict, view_rect):
    (view_min_y, view_max_y, view_min_x, view_max_x) = view_rect
    # This will be first slice
    # So we just show image here
    img = slice_dict['rescale_pixel_array']
    gray_img = convert_to_gray_image(img)
    fig = plt.figure(figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')
    threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
    gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
    img = img[view_min_y: view_max_y, view_min_x:view_max_x]
    # Show threshed_im image , original img image and gray_img
    plt.subplot(1, 4, 1)
    plt.imshow(threshed_im, cmap=plt.cm.bone)
    plt.subplot(1, 4, 2)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.subplot(1, 4, 3)
    plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    # The process to show proc_img
    filter_img = threshed_im
    contours = get_max_contours_by_filter_img(img, filter_img)
    (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
    proc_img = np.copy(img)
    for contour in contours:
        if len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)  # auto-figure the ellipse to fit contour
        ellipse_poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                        (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
        reshape_poly = ellipse_poly.reshape(ellipse_poly.shape[0], 1, ellipse_poly.shape[1])
        # cv2.drawContours(proc_img, reshape_poly, -1, (255,0,0),1)
        cv2.drawContours(proc_img, contour, -1, (255, 0, 0), 1)
    plt.subplot(1, 4, 4)
    plt.imshow(proc_img, cmap=plt.cm.bone)
    # print out all image in the same row
    plt.show()
    pass
def algo_show_by_folder(folder, is_debug = False):
    ct_filelist = get_ct_filelist_by_folder(folder)
    ct_dicom_dict = gen_ct_dicom_dict(ct_filelist)
    sorted_ct_dicom_dict_keys = sorted(ct_dicom_dict['SliceLocation'].keys(), reverse=False)
    print('z = ',sorted_ct_dicom_dict_keys[0])
    first_slice_dict = ct_dicom_dict['SliceLocation'][sorted_ct_dicom_dict_keys[0]]
    based_center_pts = get_app_center_pts_of_first_slice(first_slice_dict)
    if len(based_center_pts) != 3:
        print('len(based_center_pts) is wrong, folder = ', folder)
        print('based_center_pts = ', based_center_pts)
        if is_debug == True:
            print('debug to show plt ')
            slice_dict = first_slice_dict
            (view_min_y, view_max_y, view_min_x, view_max_x) = get_view_scope_by_slice(first_slice_dict, padding=100)
            pure_show_slice_dict(slice_dict, (view_min_y, view_max_y, view_min_x, view_max_x))
        return
    else:
        print(based_center_pts)

    first_slice_dict['data'] = {}
    first_slice_dict['data']['center_pts'] = based_center_pts

    (view_min_y, view_max_y, view_min_x, view_max_x) = get_view_scope_by_slice(first_slice_dict, padding=100)

    prev_slice_dict = None
    for z in sorted_ct_dicom_dict_keys:
        slice_dict = ct_dicom_dict['SliceLocation'][z]
        print('z = ', z, 'filepath = ', slice_dict['filepath'])

        if 'data' in slice_dict and 'center_pts' in slice_dict['data']:
            prev_slice_dict = slice_dict
            pure_show_slice_dict(slice_dict, (view_min_y, view_max_y, view_min_x, view_max_x))
            # First slice
            print('center_pts = ', slice_dict['data']['center_pts'])
            continue

        img = slice_dict['rescale_pixel_array']
        gray_img = convert_to_gray_image(img)
        fig = plt.figure(figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')
        threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
        plt.subplot(1, 4, 1)
        plt.imshow(threshed_im, cmap=plt.cm.bone)
        gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
        img = img[view_min_y: view_max_y, view_min_x:view_max_x]
        plt.subplot(1, 4, 2)
        plt.imshow(img, cmap=plt.cm.bone)
        plt.subplot(1, 4, 3)
        plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -22)
        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
        threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
        filter_img = threshed_im

        # contours = get_max_contours_by_filter_img(first_slice_dict['rescale_pixel_array'])
        contours = get_max_contours_by_filter_img(img, filter_img)
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
        proc_img = np.copy(img)

        ellipse_center_pts = []
        draw_ellipse_center_pts = []
        for contour in contours:
            if len(contour) < 5:
                # You need at least 5 points in contour, so that you can use fitEllipse
                print('contour < 5 ')
                continue
            ellipse = cv2.fitEllipse(contour)  # auto-figure the ellipse to fit contour
            # print(ellipse)
            ellipse_poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                            (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360,
                                            5)
            # Because cv2.ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta)
            # So The center point is (int(ellipse[0][0]), int(ellipse[0][1]))
            # ellipse_center_pts.append([int(ellipse[0][0]), int(ellipse[0][1])])
            draw_x = int(ellipse[0][0])
            draw_y = int(ellipse[0][1])
            draw_ellipse_center_pts.append([draw_x, draw_y])
            x = int(ellipse[0][0]) + view_min_x
            y = int(ellipse[0][1]) + view_min_y
            ellipse_center_pts.append([x, y])

            # reshape to format of contour, so that your can use draw contour function to draw it
            reshape_poly = ellipse_poly.reshape(ellipse_poly.shape[0], 1, ellipse_poly.shape[1])
            cv2.drawContours(proc_img, reshape_poly, -1, (255, 0, 0), 1)
            # cv2.drawContours(proc_img, contour, -1, (255,0,0),1)
            # cv2.line(proc_img,(draw_x,draw_y),(draw_x,draw_y),(255,0,0),3)

        figure_center_pts = []
        for pt in prev_slice_dict['data']['center_pts']:
            if len(prev_slice_dict['data']['center_pts']) == 1:
                dst_pt = get_most_closed_pt(pt, ellipse_center_pts, allowed_distance=20)
            else:
                dst_pt = get_most_closed_pt(pt, ellipse_center_pts, allowed_distance=100)
            if dst_pt != None:
                figure_center_pts.append(dst_pt)

        slice_dict['data'] = {}
        slice_dict['data']['center_pts'] = figure_center_pts

        print('ellipse_center_pts = ', ellipse_center_pts)
        print('center_pts = ', slice_dict['data']['center_pts'])

        plt.subplot(1, 4, 4)
        for [x, y] in figure_center_pts:
            draw_x = x - view_min_x
            draw_y = y - view_min_y
            cv2.line(proc_img, (draw_x, draw_y), (draw_x, draw_y), (255, 0, 0), 3)

        plt.imshow(proc_img, cmap=plt.cm.bone)
        plt.show()

        #
        prev_slice_dict = slice_dict


def get_batch_process_dict_v02(root_folder):
    process_dict = {}
    if not os.path.isdir(root_folder):
        return process_dict

    dataset_folder_list = []
    for patient_id_file in os.listdir(root_folder):
        patient_id_filepath = r"{}/{}".format(root_folder, patient_id_file)
        if not os.path.isdir(patient_id_filepath):
            continue
        for study_date_file in os.listdir(patient_id_filepath):
            study_date_filepath = r"{}/{}".format(patient_id_filepath, study_date_file)
            if not os.path.isdir(study_date_filepath):
                continue
            dataset_folder_list.append(study_date_filepath)
    for folder in dataset_folder_list:
        input_dict = {}
        # TODO in future if you need
        process_dict[folder] = input_dict
    return process_dict


def example_get_batch_process_dict():
    f_list = []
    # process_dict = get_batch_process_dict(r"AI_RS_Compare_20190724")
    process_dict = get_batch_process_dict(r"RAL_plan_shift")
    for folder in sorted(process_dict.keys()):
        if folder == r"AI_RS_Compare_20190724/35086187/0613":
            continue
        # algo_show_by_folder(folder)
        print(folder)
        f_list.append(folder)
        continue
def make_lines_process(app_pts):
    lines = [[], [], []]
    sorted_app_pts_keys = sorted(app_pts.keys())
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
                # looking forware for candidate_pt
                for pt_idx in range(len(pts)):
                    pt = pts[pt_idx]
                    pt_x = pt[0]

                    if abs(last_line_pt_x - pt_x) < 5:
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
def show_lines(lines):
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        print('lines[{}] = '.format(line_idx))
        for pt in line:
            print('\t', pt)
        print('')
def example_show_lines(folder):
    app_pts = algo_run_by_folder(folder)
    lines = make_lines_process(app_pts)
    show_lines(lines)


folder = r"RAL_plan_shift/35086187/0101"
print('Usage of make_lines_process() with folder = ', folder)

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

    zlines = []  # every element in zlines is a list that link to all circle component
    zlines.append([])
    zlines.append([])
    zlines.append([])

    prev_slice_dict = None
    for z in sorted_ct_dicom_dict_keys:
        app_pts_dict[z] = []
        slice_dict = ct_dicom_dict['SliceLocation'][z]
        print('z = ', z, 'filepath = ', slice_dict['filepath'])

        if 'data' in slice_dict and 'center_pts' in slice_dict['data']:
            prev_slice_dict = slice_dict
            #pure_show_slice_dict(slice_dict, (view_min_y, view_max_y, view_min_x, view_max_x))
            # First slice
            print('center_pts = ', slice_dict['data']['center_pts'])
            for pt in slice_dict['data']['center_pts']:
                x = pt[0]
                y = pt[1]
                app_pts_dict[z].append([x,y,z])

            continue

        img = slice_dict['rescale_pixel_array']
        gray_img = convert_to_gray_image(img)
        #fig = plt.figure(figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')
        #threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
        gray_img = gray_img[view_min_y: view_max_y, view_min_x:view_max_x]
        img = img[view_min_y: view_max_y, view_min_x:view_max_x]

        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -22)
        # threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
        threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -22)
        # I'm not sure why it is the perfect value in our case.
        filter_img = threshed_im
        contours = get_max_contours_by_filter_img(img, filter_img)
        proc_img = np.copy(img)

        ellipse_center_pts = []
        draw_ellipse_center_pts = []
        for contour in contours:
            if len(contour) < 5:
                # You need at least 5 points in contour, so that you can use fitEllipse
                continue
            ellipse = cv2.fitEllipse(contour)  # auto-figure the ellipse to fit contour
            # print(ellipse)
            ellipse_poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),(int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360,5)
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
                dst_pt = get_most_closed_pt(pt, ellipse_center_pts, allowed_distance=20)
            else:
                dst_pt = get_most_closed_pt(pt, ellipse_center_pts, allowed_distance=100)
            if dst_pt != None:
                figure_center_pts.append(dst_pt)

        slice_dict['data'] = {}
        slice_dict['data']['center_pts'] = figure_center_pts

        print('ellipse_center_pts = ', ellipse_center_pts)
        print('center_pts = ', slice_dict['data']['center_pts'])

        #plt.subplot(1, 4, 4)
        for [x, y] in figure_center_pts:
            app_pts_dict[z].append([x,y,z])
            draw_x = x - view_min_x
            draw_y = y - view_min_y
            cv2.line(proc_img, (draw_x, draw_y), (draw_x, draw_y), (255, 0, 0), 3)

        #plt.imshow(proc_img, cmap=plt.cm.bone)
        #plt.show()

        #
        prev_slice_dict = slice_dict
    print(app_pts_dict)
    return app_pts_dict
# Implementation of get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)
def get_metric_pt(metric_line, pt_idx, pt_idx_remainder):
    # print('get_metric_pt(metric_line={}, pt_idx={}, pt_idx_remainder={})'.format(metric_line, pt_idx, pt_idx_remainder))
    pt = metric_line[pt_idx].copy()
    end_pt = metric_line[pt_idx + 1]

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
# The CT data is the format with 512 x 512, but we want to tranfer it into real metric space
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


def show_tandem(metric_line, first_purpose_distance_mm, each_purpose_distance_mm):
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
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,pt_idx_remainder, travel_dist)
    print(t_pt)

    for i in range(100):
        try:
            travel_dist = each_purpose_distance_mm
            (pt_idx, pt_idx_remainder) = (t_pt_idx, t_pt_idx_remainder)
            (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx,pt_idx_remainder,travel_dist)
            print(t_pt, t_pt_idx, t_pt_idx_remainder)
        except:

            break

def get_and_show_tandem( metric_line, first_purpose_distance_mm, each_purpose_distance_mm):
    tandem_rp_line = []
    def distance(pt1, pt2):
        import math
        # print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt( (pt1[0 ] -pt2[0] )**2 +  (pt1[1 ] -pt2[1] )**2 + (pt1[2 ] -pt2[2] )**2 )
        return ret_dist
    pt_idx = 0
    pt_idx_remainder = 0
    # first_purpose_distance_mm = 7 # get first RD point by 7mm
    # each_purpose_distance_mm = 5
    travel_dist = first_purpose_distance_mm
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)
    print(t_pt)
    tandem_rp_line.append(t_pt)

    for i in range(100):
        try :
            travel_dist = each_purpose_distance_mm
            (pt_idx, pt_idx_remainder) = (t_pt_idx, t_pt_idx_remainder)
            (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)
            print(t_pt, t_pt_idx, t_pt_idx_remainder)
            tandem_rp_line.append(t_pt)
        except:

            break
    return tandem_rp_line



def show_report_by_folder(folder):
    print('folder = ', folder )
    # the function will get all 3D pt of applicator
    app_pts = algo_run_by_folder(folder)
    # transform all 3D pt of applicator into each line for each applicator and the line have been sorted by z
    lines = make_lines_process(app_pts)
    # The CT data is the format with 512 x 512, but we want to tranfer it into real metric space
    metric_lines = convert_lines_in_metrics(lines, folder)
    # Show the lines information in metrics
    show_lines(metric_lines)
    metric_line = metric_lines[1].copy()
    print('metric_line = ',metric_line)

    def distance(pt1, pt2):
        import math
        #print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt( (pt1[0]-pt2[0])**2 +  (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 )
        return ret_dist
    pt_idx = 0
    pt_idx_remainder = 0
    purpose_distance_mm = 7
    max_mm = purpose_distance_mm
    orig_pt = metric_line[0]
    print('metric_line = ', metric_line)

    def distance(pt1, pt2):
        import math
        #print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt( (pt1[0]-pt2[0])**2 +  (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 )
        return ret_dist
    pt_idx = 0
    pt_idx_remainder = 0
    orig_pt = metric_line[0]
    purpose_distance_mm = 7
    travel_dist = purpose_distance_mm
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)
    print('{} -> {}'.format((t_pt, t_pt_idx, t_pt_idx_remainder), distance(orig_pt, t_pt)))

    tandem_rp_line = get_and_show_tandem(metric_line, 4.5, 5)
    #show_tandem(metric_line, 4.5, 5)
    print('tandem_rp_line[-1] = ', tandem_rp_line[-1])

    #max_mm = purpose_distance_mm
    #orig_pt = metric_line[0]
    #for mm in range(max_mm+1):
    #    travel_dist = mm
    #    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)
    #    print( '{} -> {}'.format((t_pt, t_pt_idx, t_pt_idx_remainder), distance(orig_pt,t_pt) )  )

#show_report_by_folder(folder)



# enable disable for print
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def predict_tandem_rp_line_by_folder(folder, start_mm, gap_mm, is_debug = False):
    tandem_rp_line = []
    print('folder = ', folder )
    if is_debug == False:
        blockPrint()

    # the function will get all 3D pt of applicator
    app_pts = algo_run_by_folder(folder)
    print('app_pts = {}'.format(app_pts))
    # transform all 3D pt of applicator into each line for each applicator and the line have been sorted by z
    lines = make_lines_process(app_pts)
    # The CT data is the format with 512 x 512, but we want to tranfer it into real metric space
    metric_lines = convert_lines_in_metrics(lines, folder)
    # Show the lines information in metrics
    show_lines(metric_lines)
    metric_line = metric_lines[1].copy()
    print('metric_line = ',metric_line)

    def distance(pt1, pt2):
        import math
        #print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt( (pt1[0]-pt2[0])**2 +  (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 )
        return ret_dist
    pt_idx = 0
    pt_idx_remainder = 0
    purpose_distance_mm = 7
    max_mm = purpose_distance_mm
    orig_pt = metric_line[0]
    print('metric_line = ', metric_line)

    def distance(pt1, pt2):
        import math
        #print(r"pt1 = {}, pt2 = {}".format(pt1, pt2))
        ret_dist = math.sqrt( (pt1[0]-pt2[0])**2 +  (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 )
        return ret_dist
    pt_idx = 0
    pt_idx_remainder = 0
    orig_pt = metric_line[0]
    purpose_distance_mm = 7
    travel_dist = purpose_distance_mm
    (t_pt, t_pt_idx, t_pt_idx_remainder, t_dist) = get_metric_pt_info_by_travel_distance(metric_line, pt_idx, pt_idx_remainder, travel_dist)
    print('{} -> {}'.format((t_pt, t_pt_idx, t_pt_idx_remainder), distance(orig_pt, t_pt)))

    #tandem_rp_line = get_and_show_tandem(metric_line, 4.5, 5)
    tandem_rp_line = get_and_show_tandem(metric_line, start_mm, gap_mm)
    #show_tandem(metric_line, 4.5, 5)
    print('tandem_rp_line[-1] = ', tandem_rp_line[-1])
    if is_debug == False:
        enablePrint()
    return tandem_rp_line



print('Hello ')

f_list = []
#process_dict = get_batch_process_dict(r"AI_RS_Compare_20190724")
#process_dict = get_batch_process_dict(r"RAL_plan_shift")




def get_batch_process_dict_v03(root_folder):
    process_dict = {}
    if not os.path.isdir(root_folder):
        return process_dict

    dataset_folder_list = []
    for patient_id_file in os.listdir(root_folder):
        patient_id_filepath = r"{}/{}".format(root_folder, patient_id_file)
        if not os.path.isdir(patient_id_filepath):
            continue
        dataset_folder_list.append(patient_id_filepath)

    for folder in dataset_folder_list:
        input_dict = {}
        # TODO in future if you need
        process_dict[folder] = input_dict
    return process_dict


import pickle
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


def get_man_dict():
    ret_dict_filename = 'man_patient_rp_data.bytes'
    ret_dict = python_object_load(ret_dict_filename)
    return ret_dict


def get_tandem_from_man(man_dict, folder):
    # Here show the different rules in different rp filepath to get tandem
    dict = man_dict[folder]
    if dict == None:
        print('[get_tandem_from_man()] {} is not exist in man_dict'.format(folder))
        return None
    line = None
    apps_list = dict['apps_list']
    for line_dict in apps_list:
        n = line_dict['name']
        if n == 'Tandom' or n == 'Tandem' or 'Tandem_' in n or 'Tabdem_' in n:
            # Some case is Tandom but some case is Tandem in rp filepath
            line = line_dict['points'].copy()
            break

    # Another rules to query Tandem line are by the case of ['Applicator1', 'Applicator2_R', 'Applicator3_L',...]
    if line == None:
        names = []
        for d in apps_list:
            n = d['name']
            names.append(n)

        queryName = None
        if 'Applicator1' in names and 'Applicator2_R' in names and 'Applicator3_L' in names:
            queryName = 'Applicator1'
        if queryName != None:
            for line_dict in apps_list:
                n = line_dict['name']
                if n == queryName:
                    # Some case is Tandom but some case is Tandem in rp filepath
                    line = line_dict['points'].copy()
                    break
    if line == None:
        names = []
        for ld in apps_list:
            names.append(ld['name'])
        print('line == None, and names = {}'.format(names))
    return line



def show_man_dict():
    man_dict = get_man_dict()
    for folder in sorted(man_dict.keys()):
        tandem = get_tandem_from_man(man_dict, folder)
        print('folder = {}, and tandem = {}'.format(folder,tandem))

process_dict = get_batch_process_dict_v03(r"RAL_plan_new_20190905")

for folder in sorted(process_dict.keys()):
    if folder == r"AI_RS_Compare_20190724/35086187/0613":
        continue
    # algo_show_by_folder(folder)
    print(folder)
    f_list.append(folder)
    continue

man_dict = get_man_dict()
folder_idx = 0
import math
def dist_3d(pt1, pt2):
    return math.sqrt( (pt1[0]-pt2[0])**2 +  (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2 )




broken_f_list = ['RAL_plan_new_20190905/29059811-2', 'RAL_plan_new_20190905/35252020-2']
debug_idx = 0
for folder in broken_f_list:
    if debug_idx != -1:
        print('debug for folder = ', folder)
        algo_show_by_folder(folder, is_debug = True)
    #ai_tandem_rp_line = predict_tandem_rp_line_by_folder(folder, start_mm=4.5, gap_mm=5, is_debug = True)
    debug_idx = debug_idx + 1
#exit(0)


broken_f_list = []
for folder in f_list:

    try:
        ai_tandem_rp_line = predict_tandem_rp_line_by_folder(folder, start_mm=4.5, gap_mm=5)
        man_tandem_rp_line = get_tandem_from_man(man_dict, folder)
        print('folder = {}, \nai_tandem_rp_line= {}, \nman_tandem_rp_line={}\n'.format(folder,ai_tandem_rp_line, man_tandem_rp_line))
    except:
        enablePrint() # Because predict_tandem_rp_line_by_folder() use blockPrint(), so enablePrint when catch exception
        print('folder  = {} is break'.format(folder))
        broken_f_list.append(folder)
        continue

print('len = {}, f_list = {}'.format(len(f_list), f_list))
print('len = {}, broken_f_list = {}'.format(len(broken_f_list), broken_f_list) )

exit(0)

broken_f_list = ['RAL_plan_new_20190905/29059811-2', 'RAL_plan_new_20190905/35252020-2']

folder_idx = 0
for folder in broken_f_list:
    if folder_idx != 0:
        ai_tandem_rp_line = predict_tandem_rp_line_by_folder(folder, start_mm=4.5, gap_mm=5, is_debug=False)
        print('processed error folder name = ', folder)
        break
    folder_idx = folder_idx + 1

#exit(0)




for folder in sorted(man_dict.keys()):
    print('folder = {}, with folder_idx = {}'.format(folder, folder_idx))
    # figure out the distance between ai tandem line and manual tandem line

    man_line = get_tandem_from_man(man_dict, folder)
    ai_line = []
    try:
        ai_line = predict_tandem_rp_line_by_folder(folder, start_mm=0.1, gap_mm=5)
    except:
        enablePrint()
        print('Why dead on case of folder = {}? finding it '.format(folder))

    print('folder = {}\nman_line={}\nai_line={}\n\n'.format(folder, man_line, ai_line))
    continue
    man_line_len = len(man_line)
    if man_line_len > len(ai_line):
        print('In case folder = {}, len of man line = {} > len of ai line = {}'.format(folder, man_line_len, len(ai_line)))
        continue
    man_1st_pt = man_line[0]
    ai_1st_pt = ai_line[0]
    man_list_pt = man_line[man_line_len - 1]
    ai_last_pt = ai_line[man_line_len -1]
    folder_idx = folder_idx + 1
    break


