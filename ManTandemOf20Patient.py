
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


def get_process_list(root_folder):
    import pydicom
    import os
    import copy
    process_list = []
    for file in sorted(os.listdir(root_folder)):
        basedir_filepath = r"{}/{}".format(root_folder, file)
        # print('basedir of data = ', basedir_filepath)
        ct_filelist = []  # ct file list
        rs_filepath = None  # Structure filepath
        rp_filepath = None  # Plan file filepath
        rd_filepath = None  # Dose file filepath

        for file in sorted(os.listdir(basedir_filepath)):
            dicom_filepath = r"{}/{}".format(basedir_filepath, file)
            try:
                fp = pydicom.read_file(dicom_filepath)
                if fp.Modality == "CT":
                    ct_filelist.append(dicom_filepath)
                elif fp.Modality == "RTDOSE":
                    rd_filepath = dicom_filepath
                elif fp.Modality == "RTPLAN":
                    rp_filepath = dicom_filepath
                elif fp.Modality == "RTSTRUCT":
                    rs_filepath = dicom_filepath
                else:
                    print(fp.Modality)
                pass
            except:
                print('file process error')
                break
                pass
        if rs_filepath == None or rp_filepath == None or rd_filepath == None or len(ct_filelist) == 0:
            print(r"data is incompleted in {}, so that it not recorded".format(basedir_filepath))
        i_dict = {}
        i_dict['rs_filepath'] = rs_filepath
        i_dict['rp_filepath'] = rp_filepath
        i_dict['rd_filepath'] = rd_filepath
        i_dict['ct_filelist'] = ct_filelist
        o_dict = {}

        p_dict = {}
        p_dict['input'] = i_dict
        p_dict['output'] = o_dict
        process_list.append(copy.deepcopy(p_dict))
    return process_list


def get_apps_list(rp_filepath):
    apps_list = []

    # make result_list by the following code
    def get_dict_of_ChannelSequenceItem(idx, cseqItem):
        c_dict = {}
        c_dict['idx'] = idx
        c_dict['name'] = cseqItem.SourceApplicatorID
        points = []
        for cp in cseqItem.BrachyControlPointSequence:

            # if len(points) == 0 or points[-1] != cp.ControlPoint3DPosition:
            if True:
                points.append(cp.ControlPoint3DPosition)
        c_dict['points'] = points
        return c_dict

    # rd_filepath = r'D:\kai\DICOMtoNPZ\ral\RP.1.2.246.352.71.5.417454940236.2035996.20190820095628.dcm'
    rp_fp = pydicom.read_file(rp_filepath)
    cseq = rp_fp.ApplicationSetupSequence[0].ChannelSequence
    for idx in range(len(cseq)):
        c_dict = get_dict_of_ChannelSequenceItem(idx, cseq[idx])
        apps_list.append(c_dict)
    return apps_list


def show_apps_list(apps_list):
    for app_dict in apps_list:
        for app_key in app_dict.keys():
            app_val = app_dict[app_key]
            if (type(app_val) == str):
                print('{}->{}'.format(app_key, app_val))
            elif type(app_val) == list:
                print('{}->'.format(app_key))
                for pt in app_val:
                    print('\t', pt)
            elif app_key == 'idx':
                print('{}->{}'.format(app_key, app_val))
            else:
                print("The non-process key = {} and val's type = ", app_key, type(app_val))
                print(app_val)
def example_case():
    rd_filepath = r'D:\kai\DICOMtoNPZ\ral\RP.1.2.246.352.71.5.417454940236.2035996.20190820095628.dcm'
    apps_list = get_apps_list(rd_filepath)
    show_apps_list(apps_list)

#example_case()

def example_case_2():
    root_folder = "RAL_plan"
    process_list = get_process_list(root_folder)
    p_list = get_process_list(root_folder)
    for d in p_list:
        rp_filepath = d['input']['rp_filepath']
        print('\n\nrp_filepath = ', rp_filepath)
        apps_list = get_apps_list(rp_filepath)
        show_apps_list(apps_list)

def example_case_3():
    root_folder = "RAL_plan_new_20190905"
    process_list = get_process_list(root_folder)
    p_list = get_process_list(root_folder)
    idx_cnt = 0
    for d in p_list:

        rp_filepath = d['input']['rp_filepath']
        print('\n\n[{}] rp_filepath = ',idx_cnt,  rp_filepath)
        apps_list = get_apps_list(rp_filepath)
        show_apps_list(apps_list)
        idx_cnt = idx_cnt + 1
example_case_3()
