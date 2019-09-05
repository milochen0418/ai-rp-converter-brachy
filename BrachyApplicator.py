




def show_all_cv_processing_output():
    # Start to implement new version of algorithm
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
    def get_rect_infos_and_center_pts(contours):
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
            if h >= 13 and h < 19 and w >= 13 and h < 19:
                cen_pt = [x_mean, y_mean]
                app_center_pts.append(cen_pt)
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
        (contours, constant) = get_max_contours(first_slice_dict['rescale_pixel_array'])
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
        x_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
        return x_sorted_pts
        pass
    def get_view_scope_by_slice(first_slice_dict, padding=30):
        (contours, constant) = get_max_contours(first_slice_dict['rescale_pixel_array'])
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
        print(sorted_app_center_pts)

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
    def get_batch_process_dict(root_folder):
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
    def algo_show_by_folder(folder):
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
    def example_show_root_folder():
        f_list = []
        process_dict = get_batch_process_dict(r"RAL_plan_shift")
        for folder in sorted(process_dict.keys()):
            if folder == r"AI_RS_Compare_20190724/35086187/0613":
                continue
            #algo_show_by_folder(folder)
            #print(folder)
            f_list.append(folder)
            continue
        #f_list = sorted(process_dict.keys())
        count = 0

        for folder in f_list:
            print(count , ' -> ', folder )
            count+=1
        count = 0
        algo_show_by_folder(f_list[1])
    example_show_root_folder()



def show_3d_plot_result():
    # 3D plot
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
    def get_rect_infos_and_center_pts(contours):
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
            if h >= 13 and h < 19 and w >= 13 and h < 19:
                cen_pt = [x_mean, y_mean]
                app_center_pts.append(cen_pt)
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
        (contours, constant) = get_max_contours(first_slice_dict['rescale_pixel_array'])
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
        x_sorted_pts = sorted(app_center_pts, key=lambda cen_pt: cen_pt[0], reverse=False)
        return x_sorted_pts
        pass
    def get_view_scope_by_slice(first_slice_dict, padding=30):
        (contours, constant) = get_max_contours(first_slice_dict['rescale_pixel_array'])
        (sorted_app_center_pts, rect_infos, app_center_pts) = get_rect_infos_and_center_pts(contours)
        print(sorted_app_center_pts)

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
    def get_batch_process_dict(root_folder):
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
    # strict 3D pt=[x,y,z] into [x,y]
    def get_most_closed_pt_strictly(src_pt, pts, allowed_distance=1000):
        if pts == None:
            return None
        if pts == []:
            return None
        strictly_src_pt = src_pt[0:2]
        strictly_pts = []
        for pt in pts:
            strictly_pt = pt[0:2]
            strictly_pts.append(strictly_pt)
        return get_most_closed_pt(strictly_src_pt, strictly_pts, allowed_distance)
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
    def algo_show_by_folder(folder):
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

            #threshed_im = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -22)
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
                    dst_pt = get_most_closed_pt(pt, ellipse_center_pts, allowed_distance=10)

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
    def app_pts_show3D(app_pts):
        # app_pts[DSfloat(z)] = [ [int(x), int(y), DSfloat(z)], ...  ]
        # Draw in 3D
        from mpl_toolkits import mplot3d
        import numpy as np
        import matplotlib.pyplot as plt

        obj3d = {}
        obj3d['x'] = []
        obj3d['y'] = []
        obj3d['z'] = []

        def drawPoint(obj3d,x,y,z):
            obj3d['x'].append(x)
            obj3d['y'].append(y)
            obj3d['z'].append(z)

        def drawShow(obj3d):
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.scatter3D(obj3d['x'], obj3d['y'], obj3d['z'], c=obj3d['z'], cmap='hsv')
            plt.show()

        keys = sorted(app_pts.keys())
        keys_len = len(keys)
        '''
        for z in sorted(app_pts.keys()):
            for pt in app_pts[z]:
                drawPoint(obj3d, pt[0], pt[1], pt[2])
        '''
        for idx,z in enumerate(keys):
            prev_z = None
            pprev_z = None # prev of prev z
            next_z = None
            if idx > 0:
                prev_z = keys[idx - 1]
            if idx > 1:
                pprev_z = keys[idx - 2]
            if idx < keys_len - 1:
                next_z = keys[idx + 1]
            prev_app_pts = None
            pprev_app_pts = None
            next_app_pts = None
            if next_z != None:
                next_app_pts = app_pts[next_z]
            if prev_z != None:
                prev_app_pts = app_pts[prev_z]
            if pprev_z != None:
                pprev_app_pts = app_pts[pprev_z]

            sorted_app_pts = sorted(app_pts[z], key = lambda pt:pt[0])
            #for pt in app_pts[z]:
            for pt in sorted_app_pts:
                # Case of 3 point and this is center point
                if len(sorted_app_pts) == 3 and pt == sorted_app_pts[1]: # if pt is center point
                    # TODO
                    drawPoint(obj3d, pt[0], pt[1], pt[2])
                    continue
                # Case of only 1 point and we assume it is center point
                if len(sorted_app_pts) == 1:
                    # TODO
                    drawPoint(obj3d, pt[0], pt[1], pt[2])
                    continue

                # The following code is fine tune code for Left and right applicator
                pprev_pt = None
                prev_pt = None
                next_pt = None
                if prev_app_pts != None:
                    prev_pt = get_most_closed_pt_strictly(pt,prev_app_pts, allowed_distance = 100)
                if next_app_pts != None:
                    next_pt = get_most_closed_pt_strictly(pt,next_app_pts, allowed_distance = 100)
                if pprev_app_pts != None and prev_pt != None:
                    pprev_pt = get_most_closed_pt_strictly(prev_pt, pprev_app_pts, allowed_distance = 100)

                # prev_pt (or next_pt) is 2D(pt[0]=x,pt[1]=y) point if  prev_pt (or next_pt) is not None
                if prev_pt != None and next_pt != None and pprev_pt != None:
                    prev_dist = distance(pt[0:2], prev_pt)
                    next_dist = distance(pt[0:2], next_pt)
                    pprev_dist =distance(prev_pt, pprev_pt)
                    np_dist = distance(next_pt, prev_pt) #distance of next_dist to prev_dist


                    # comment old tune decision
                    #tune_val = 50
                    #if prev_dist + next_dist > tune_val:
                    #    print('need to tune point at pt = ', pt)
                    #distance change speed < 5 dist unit / z
                    tune_val = 4*4 # distance change speed is allowed in 5
                    # if (abs(prev_dist - pprev_dist) + tune_val> abs(next_dist - prev_dist)):
                    #if ( abs(prev_dist - pprev_dist)  > abs(next_dist - prev_dist) + tune_val ):

                    is_tune = False
                    # the case of ppD = 0 will cause failed, so we set ppD <- 1 when ppD == 0
                    ref_pprev_dist = pprev_dist
                    if pprev_dist == 0:
                        ref_pprev_dist = 1

                    if prev_dist > ref_pprev_dist * 4 : # pD >> ppD
                        if next_dist > ref_pprev_dist * 4: # nD >> ppD
                            if np_dist < 2*ref_pprev_dist + 9 :# npD ~< 2 ppD

                                is_tune = True
                    #tune_val = pprev_dist + 3*3
                    #if prev_dist > tune_val and next_dist > tune_val:
                    if is_tune == True:
                        print('\nneed to tune point at pt = ', pt)
                        print('(pprev_pt,prev_pt,pt,next_pt)=',"({},{},{},{})".format(pprev_pt,prev_pt,pt,next_pt))
                        print('(pprev_z,prev_z,z,next_z)=', "({},{},{},{})".format(pprev_z, prev_z, z, next_z))
                        print('(pprev_dist,prev_dist,next_dist)=',"({},{},{})".format(ref_pprev_dist, prev_dist, next_dist))

                        # fine tune procedure
                        #(prev_pt[0] - pprev_pt[0])

                        #(pt[0] - prev_pt[0])
                        '''
                        offset_x = pt[0] - prev_pt[0]
                        offset_y = pt[1] - prev_pt[1]
                        '''

                        prev_offset_x = prev_pt[0] - pprev_pt[0]
                        prev_offset_y = prev_pt[1] - pprev_pt[1]

                        #prev_offset_x = prev_pt[0] - pprev_pt[0]
                        #prev_offset_y = prev_pt[1] - pprev_pt[1]

                        for p_pt in sorted_app_pts:
                            if pt[0] == p_pt[0] and pt[1] == p_pt[1]:
                                print('before, z = ', z, 'pts = ', sorted_app_pts)
                                p_pt[0] = prev_pt[0] + prev_offset_x
                                p_pt[1] = prev_pt[1] + prev_offset_y
                                print(' after, z = ', z, 'pts = ', sorted_app_pts)
                                pass

                drawPoint(obj3d, pt[0], pt[1], pt[2])

        drawShow(obj3d)

    f_list = []
    process_dict = get_batch_process_dict(r"RAL_plan_shift")
    for folder in sorted(process_dict.keys()):
        if folder == r"AI_RS_Compare_20190724/35086187/0613":
            continue
        # algo_show_by_folder(folder)
        print(folder)
        f_list.append(folder)
        continue

    folder = f_list[1]
    app_pts = algo_run_by_folder(folder)
    app_pts_show3D(app_pts)
    print('folder = ', folder)





if __name__ == '__main__':
    print('Hello World 2')
    #show_all_cv_processing_output()
    show_3d_plot_result()