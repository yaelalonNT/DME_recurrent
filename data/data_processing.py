# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:10:20 2023

@author: yael
"""
import pyodbc 
import pandas as pd
import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def connect_to_DB(DB_ip):
    """
    Initiation of connection to the DB
    """
    sub_DB = 'OCTanalysis' 
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                  'Server=%s;'
                  'Database=%s;'
                  'uid=sa;pwd=Vision321' % (DB_ip,sub_DB))
    return conn
    
def query():
    query = """
    select distinct _patient.StudySubjectID as Patient,Scan.Eye,
    Scan.UniqueIdentifier as Scan from Scan
    join Session on scan.SessionID = session.SessionID
    join _patient on session.PatientID = _Patient.PatientID
    join _PatientEyeStudy on _PatientEyeStudy.PatientID = _Patient.PatientID
    join _Study on _PatientEyeStudy.StudyID = _Study.StudyID
    where _Study.Name in ('C2022.001','C2021.003','RGX-314-5101','GR43828','C2022.001')
    order by _patient.studysubjectid, scan.eye
    """
    return query

def query_class_type():
    query = """
    select distinct VG_aup.ID,_patient.studysubjectid, scan.eye,Scan.UniqueIdentifier,VG_BScanVSROutput.VisFrameID,
    VG_BScanOutput.ClassTypeReg from scan
    join analysisunitprocess as DN_aup on scan.ScanID = DN_aup.ScanID
    join analysisunitprocess as VG_aup ON VG_aup.ScanID = scan.ScanID
    join DN_ScanOutput on DN_ScanOutput.analysisunitprocessid = DN_aup.id
    JOIN VG_BScanVSROutput ON VG_BScanVSROutput.AnalysisUnitProcessID = VG_aup.ID
    join VG_BScanOutput ON VG_BScanOutput.AnalysisUnitProcessID = VG_aup.ID AND VG_BScanOutput.RastIndex = VG_BScanVSROutput.RastIndex
    join Session on scan.SessionID = session.SessionID
    join _patient on session.PatientID = _Patient.PatientID
    join _PatientEyeStudy on _PatientEyeStudy.PatientID = _Patient.PatientID
    join _Study on _PatientEyeStudy.StudyID = _Study.StudyID
    where _Study.Name in ('C2021.003','RGX-314-5101','GR43828','C2022.001')
    and VG_aup.RunModeTypeID = DN_aup.RunModeTypeID AND VG_aup.RunModeTypeID !=0 
    order by _patient.studysubjectid, scan.eye,Scan.UniqueIdentifier,VG_BScanVSROutput.VisFrameID

    """ 
    return query
    
def save_data_summary(scans_path,save_path,save_name):
    """
    The function saves a summary of the data - patient,eye and scan name, 
    alongside train-validation split of eyes
    """
    mat_files = os.listdir(scans_path)
    scans = [mat_file.split('.mat')[0] for mat_file in mat_files]
    scans = [scan + '                ' for scan in scans]
    my_query = query()
    df = pd.io.sql.read_sql(my_query, conn)
    df = df[df.Scan.isin(scans)].reset_index(drop=True)
    df['Patient_Eye'] = df.Patient + df.Eye
    pt_eye_unique = df['Patient_Eye'].unique()
    train, validation = train_test_split(pt_eye_unique, test_size=0.2,random_state=100)
    for n in range(0,len(df)):
        df.loc[n,'data set'] = 'train' if df['Patient_Eye'][n] in train else 'validation'
    df.to_csv(os.path.join(save_path,save_name))
    
    return df

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return

def save_numpy_files(df,mat_file_path,save_numpy_path):
    """
    The function saves the data of the images in a seperate np file for raw B-scan
    and corresponding segmentaion GS, after filtering out B-scans that we didn't
    manage to register correctly (classes 3 or 4 in registration)
    """
    
    mkdir(save_numpy_path)
    Images_folder = 'I_VS'
    Seg_folder = 'Seg'
    
    I_path = os.path.join(mat_file_path,Images_folder)
    seg_path = os.path.join(mat_file_path,Seg_folder)
    I_save_path = os.path.join(save_numpy_path,Images_folder)
    mkdir(I_save_path)
    seg_save_path = os.path.join(save_numpy_path,Seg_folder) 
    mkdir(seg_save_path)
    scans = df.Scan
    my_query_class = query_class_type()
    df_classes = pd.io.sql.read_sql(my_query_class, conn)
    
    print('Starting to save numpy files')
    for n,scan in enumerate(tqdm(scans)):
        df_classes_scan = df_classes[df_classes.UniqueIdentifier == scan].reset_index()
        df.loc[n,'Number lines'] = len(df_classes_scan)
        scan = scan.replace(' ','')
        scan_I_save_path = os.path.join(I_save_path,scan)
        mkdir(scan_I_save_path)
        scan_seg_save_path = os.path.join(seg_save_path,scan)
        mkdir(scan_seg_save_path)
        I_all = scipy.io.loadmat(os.path.join(I_path,scan + '.mat'))['I']
        seg_all = scipy.io.loadmat(os.path.join(seg_path,scan + '.mat'))['Ilab']
        num_b = np.shape(I_all)[0]
        
        if num_b!=len(df_classes_scan):
            raise Exception("Error - mismatch between the number of B-scans that is saved\
                            and the number accroding to sql for scan %s" %scan)
                            
        for n in range(0,num_b):
            if df_classes.ClassTypeReg[n] in [1,2]:
                I = I_all[n,:,:].astype('float16')
                I_seg = seg_all[n,:,:].astype('float16')
                np.save(os.path.join(scan_I_save_path,'%d.npy'%n),I)
                np.save(os.path.join(scan_seg_save_path,'%d.npy' %n),I_seg)
                
    df.to_csv(os.path.join(save_csv_path,save_name))
    return df   

if __name__ =="__main__":     
    scans_path = r'\\172.17.102.175\Data\DME_recurrent\Data\matfile\I_VS'
    mat_file_path = r'\\172.17.102.175\Data\DME_recurrent\Data\matfile'
    save_path_main = r'\\nv-nas01\Data\DME_recurrent\Model'
    
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H")
    save_csv_path =  os.path.join(save_path_main,date_str)
    mkdir(save_csv_path)
    save_numpy_path = os.path.join(save_csv_path,'numpy')
    save_name = 'data_summary.csv'
    conn = connect_to_DB(DB_ip = '172.30.2.246')

    df = save_data_summary(scans_path,save_csv_path,save_name)
    save_numpy_files(df,mat_file_path,save_numpy_path)    
    

    

