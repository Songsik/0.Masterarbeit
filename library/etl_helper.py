import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from library import faps_color as fapsc
from tqdm.notebook import tqdm_notebook as tq


def build_dataframe(input_list, downsample):
    
    df = pd.concat(input_list, axis=1, ignore_index=True)
    df = df[:][::downsample]
    df = df.reset_index(drop=True)
    df[df < 0] = np.nan
    df = df.fillna(0)
    
    return df


def load_data(main_path, folders, fname, downsample, file_amount):
    
    torque = []
    #angle = []
    len_curves = []

    for i in tq(range(file_amount)):
        try:
            file_path = main_path + folders + fname + str(i) + ".csv"
            data = pd.read_csv(file_path, encoding = 'ISO-8859-1')[["Drehmoment(N·m)", "Winkel(°)"]]
            torque.append(data["Drehmoment(N·m)"])
            #angle.append(data["Winkel(°)"])
            len_curves.append(len(data))
        except:
            print(i)
        
    df = build_dataframe(torque, downsample)
    #ang = build_dataframe(angle, downsample)
    
    return df, len_curves


def set_time(df, downsample=20):
    
    original_time_step = 0.000175
    len_datapoints = len(df) * downsample        # down sample value -> 20
    max_time = original_time_step * len_datapoints
    time_steps = round(max_time/len(df), 4)
    
    timeline = []

    for i in range(len(df)):
        i = round(i*time_steps, 4)
        timeline.append(i)
        
    return np.array(timeline).reshape(-1)


def check_low_short(df, len_curves):
    
    short_curves = []
    low_curves = []
   
    for i, length in enumerate(len_curves): 
        if length < np.max(len_curves)*0.8:
            short_curves.append(i)
    
    for col in df:
        if max(df[col]) < 1.0:
            low_curves.append(col)
    
    print(f"short curves: {short_curves}")
    print(f"low curves: {low_curves}")


def boxplot(rising_point, figsize=(4,5)):

    plt.figure(figsize=figsize)
    plt.boxplot(rising_point, showmeans=True)
    plt.grid()
    plt.show()


def get_rising_point(df):

    rising_point = []

    for col in df:
        for i, val in enumerate(df[col]):
            if val > 0.8:
                rising_point.append(i)
                break
    #print(f"mean rising point: {np.mean(rising_point)}")
    #boxplot(rising_point)

    return rising_point


def get_max_values(df):
    
    max_values = []

    for col in df:
        max_values.append(max(df[col]))

    #print(f"mean max_values: {np.mean(max_values)}")
    #boxplot(max_values)

    return max_values



def rename_csv(main_path, folder, fname_new, name_start, name_end):
    
    for i in range(name_start, name_end):
        old = os.path.join(main_path, folder, os.listdir(main_path + folder)[i-name_start])
        new = os.path.join(main_path, folder, fname_new + str(i) + ".csv")
        #print(i)
        os.rename(old, new)


def save_df(df, name):
    
    df.to_pickle(f"dataframes/{name}.pkl")

    print(f"dataframe {name} is saved")



def plot_df(df, len_curves, color, label, incorrect_curves):

    plt.figure(figsize=(14,7), dpi=70)
    plt.plot(set_time(df, len_curves), df[df.columns], color=color)
    plt.plot([],[], color, label=label)

    plt.plot(set_time(df, len_curves), df[df.columns[incorrect_curves]], color="r")
    plt.plot([],[], "r", label="Fehlerhafte Kurven")

    plt.xlabel('Zeit in Sek', fontsize=15)
    plt.ylabel('Drehmoment in kN', fontsize=15)
    plt.grid(True, linestyle='-', color='grey', alpha=0.7)
    plt.legend(loc='upper left', fontsize="xx-large")
    plt.show()



def plot_curves_df(df, class_sector, classes, max_len_curves):

    plt.figure(figsize=(16,8), dpi=70)
    plt.xlabel("Zeit in Sekunden", fontsize="xx-large")
    plt.ylabel("Drehmoment in kN", fontsize="xx-large")

    for i in range(class_sector[0], class_sector[1]):
        plt.plot(set_time(df, max_len_curves), df[df.columns[i]], fapsc.green)
    plt.plot([], [], fapsc.green, label=classes[0])

    for i in range(class_sector[1], class_sector[2]):
        plt.plot(set_time(df, max_len_curves), df[df.columns[i]], fapsc.dark_green)
    plt.plot([], [], fapsc.dark_green, label=classes[1])

    for i in range(class_sector[2], class_sector[3]):
        plt.plot(set_time(df, max_len_curves), df[df.columns[i]], fapsc.yellow)
    plt.plot([], [], fapsc.yellow, label=classes[2])

    for i in range(class_sector[3], class_sector[4]):
        plt.plot(set_time(df, max_len_curves), df[df.columns[i]], fapsc.orange)
    plt.plot([], [], fapsc.orange, label=classes[3])

    for i in range(class_sector[4], class_sector[5]):
        plt.plot(set_time(df, max_len_curves), df[df.columns[i]], fapsc.blue)
    plt.plot([], [], fapsc.blue, label=classes[4])

    for i in range(class_sector[5], class_sector[6]):
        plt.plot(set_time(df, max_len_curves), df[df.columns[i]], fapsc.grey_3)
    plt.plot([], [], fapsc.grey_3, label=classes[3])

    
    leg = plt.legend(loc='upper left', fontsize="xx-large")
    for line in leg.get_lines():
        line.set_linewidth(8)

    plt.grid(color="grey", linestyle='-')
    plt.show()




def concat_df(df_list, shift_curves=None):
    
    df = pd.concat(df_list, axis=1)
    df = df.set_axis([x for x in np.arange(0, len(df.columns))], axis=1, inplace=False)
    
    if shift_curves==True:
        for col in df:
            shift_nr = df[col].isna().sum()
            df[col] = df[col].shift(periods=shift_nr)
    else:
        pass
    
    df = df.fillna(0)
            
    return df




def plot_representative_curves(df, class_sector, classes, max_len_curves):

    plt.figure(figsize=(16,8), dpi=70)
    plt.xlabel("Zeit in Sekunden", fontsize="xx-large")
    plt.ylabel("Drehmoment in kN", fontsize="xx-large")
    linewidth = 3

    plt.plot(set_time(df, max_len_curves), df[df.columns[class_sector[0]+1]], fapsc.green, label=classes[0], linewidth=linewidth)
    plt.plot(set_time(df, max_len_curves), df[df.columns[class_sector[1]+1]], fapsc.dark_green, label=classes[1], linewidth=linewidth)
    plt.plot(set_time(df, max_len_curves), df[df.columns[class_sector[2]+2]], fapsc.yellow, label=classes[2], linewidth=linewidth)
    plt.plot(set_time(df, max_len_curves), df[df.columns[class_sector[3]+3]], fapsc.orange, label=classes[3], linewidth=linewidth)
    plt.plot(set_time(df, max_len_curves), df[df.columns[class_sector[4]]], fapsc.blue, label=classes[4], linewidth=linewidth)
    plt.plot(set_time(df, max_len_curves), df[df.columns[class_sector[5]+4]], fapsc.grey_3, label=classes[5], linewidth=linewidth)

    leg = plt.legend(loc='upper left', fontsize="xx-large")
    for line in leg.get_lines():
        line.set_linewidth(8)
    plt.grid(color="grey", linestyle='-')
    plt.show()




def check_max_length(class_name, main_path, fname, file_amount):
    len_files = []
    short_curve_index = []
    
    for i in range(1, file_amount+1):
        file_path =  main_path + class_name + fname + "_" + str(i) +".csv"
        df = pd.read_csv(file_path, encoding = 'ISO-8859-1')["Drehmoment(N·m)"]
        len_files.append(len(df))
        
    for index, length in enumerate(len_files):
        if length < np.max(len_files)*0.8:
            short_curve_index.append(index)
    
    
    return len_files, short_curve_index



def save_df(df, name):
    
    df.to_pickle(f"dataframes/{name}.pkl")

    print(f"dataframe {name} is saved")



def detect_curves_wo_peak(df, upper_limit):
    
    index_list = []

    for i in range(len(df.columns[1:])):
        if max(df[i]) < upper_limit:
            index_list.append(i)
        else:
            pass
        
    print(len(index_list))
    return index_list



def remove_curves(df, remove_list):
    
    df = df.drop(remove_list, axis=1)
    
    return df 



def get_points_of_fall(df, pre_remove_list):
    
    points_of_fall = []
    
    for i in range(len(df.columns[1:])+len(pre_remove_list)):
        for j in range(600, len(df)):
            if i in pre_remove_list:
                points_of_fall.append(0)
                break
            else:
                df_pct_change = df[i].pct_change()
                if df_pct_change[j]==-1:
                      points_of_fall.append(j)
                        
    print(len(points_of_fall))
    return points_of_fall



def get_points_of_rise(df, pre_remove_list, upper_limit):

    points_of_rise = []

    for i in range(len(df.columns[1:])+len(pre_remove_list)):
        for j in range(len(df)):
            if i in pre_remove_list:
                points_of_rise.append(0)
                break
            else:
                if df[i][j] > upper_limit:
                    points_of_rise.append(j)
                    break
                    
    print(len(points_of_rise))   
    return points_of_rise



def calc_wo_0(lst, start=None, end=None):
    
    lst_wo_0 = []
    
    for i in lst:
        if i != 0:
            lst_wo_0.append(i)
            
    if (start != None) & (end!=None):
        lst_wo_0 = lst_wo_0[start:end]
            
    print(f"length: {len(lst_wo_0)}")
    print(f"min: {min(lst_wo_0)}")
    print(f"mean: {int(np.mean(lst_wo_0))}")
    print(f"max: {max(lst_wo_0)}")



def shift_curves(df, pre_remove_list, list_rise_fall, reference_point, rel_error):
    diffs = []
    for col, rise_fall_point in enumerate(list_rise_fall):
        if col in pre_remove_list:
            pass
        
        else:
            if (reference_point > rise_fall_point) & ((abs(reference_point-rise_fall_point)/reference_point) > rel_error):

                diff = abs(rise_fall_point-reference_point)
                diffs.append(diff)
                df[col] = df[col].shift(periods=diff)

            elif (reference_point < rise_fall_point) & ((abs(reference_point-rise_fall_point)/reference_point) > rel_error):

                diff = abs(rise_fall_point-reference_point)
                diffs.append(-diff)
                df[col] = df[col].shift(periods=-diff)
    
    df = df.fillna(0)
            
    return df, diffs
