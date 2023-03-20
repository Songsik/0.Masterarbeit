import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import tensorflow as tf
import pickle
from tqdm.notebook import tqdm_notebook as tq
from warnings import filterwarnings
filterwarnings('ignore')
import importlib
from library import faps_color as fapsc


def pad_curves(arr, new_shape1):

    pad_arr = np.zeros(arr.shape[0]*new_shape1).reshape(-1, new_shape1)
    pad_arr[:arr.shape[0], :arr.shape[1]] = arr
    print(pad_arr.shape)

    return pad_arr


def plot_error_hist(train_error, test_error, bin_num1, bin_num2, val=True, figsize=(8,4), dpi=80):

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    axs[0].hist(train_error, color=fapsc.dark_green, bins=bin_num1)
    axs[0].set_title("Trainingsdaten", fontsize=16)
    #axs[0].set_xlabel("MAE", fontsize=16)
    axs[0].set_ylabel("Anzahl der Fehler", fontsize=16)
    axs[0].tick_params(axis='both', labelsize=14)

    axs[1].hist(test_error, color=fapsc.green, bins=bin_num2)
    if val:
        axs[1].set_title("Validierungsdaten", fontsize=16)
    else:
        axs[1].set_title("Tesdaten", fontsize=16)
    #axs[1].set_xlabel("MAE", fontsize=16)
    axs[1].tick_params(axis='both', labelsize=14)

    plt.tight_layout(pad=1)
    plt.show()


def plot_clean_fraud(y_test, test_error, bin_num0, bin_num1, bin_width0=0.0005, threshold=None, figsize=(10,5), dpi=80):

    clean = test_error[y_test==0]
    fraud = test_error[y_test>0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(clean, bins=bin_num0, width=bin_width0, label="Kurven der Klasse i.O.", alpha=1, color=fapsc.green)
    ax.hist(fraud, bins=bin_num1, label="Kurven der Fehlerklassen", alpha=0.5, color=fapsc.red)
    if threshold:
        ax.axvline(threshold, label=f"Schwellenwert: {threshold}", color=fapsc.black, linewidth=2)

    ax.set_xlabel("Rekonstruktionsfehler in MAE", fontsize=16)
    ax.set_ylabel("Anzahl der Fehler", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    plt.rcParams['font.family'] = ['Arial']
    plt.legend(loc='upper right', fontsize=16)
    plt.show()



def plot_calc_cm_binary(y_true, y_pred, figsize=(6,6), dpi=80):
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(ax=ax, colorbar=False, cmap="Greens")

    plt.xticks(ticks=[0,1], labels=["0_normal", "1_fehlerhaft"])
    plt.yticks(ticks=[0,1])
    plt.tick_params(axis='both', labelsize=16)
    
    plt.xlabel("Vorhergesagte Klasse", fontsize=16)
    plt.ylabel("Wahre Klasse", fontsize=16)
    plt.rcParams['font.size'] = 20
    #fig.tight_layout(pad=2)
    plt.show()
    
    print(classification_report(y_true, y_pred))


def plot_calc_cm(y_true, y_pred, label, dpi=80):
    
    fig, ax = plt.subplots(figsize=(7,7), dpi=dpi)
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(ax=ax, colorbar=False, cmap="Greens")

    plt.xticks(ticks=[x for x in range(len(label))])
    plt.yticks(ticks=[x for x in range(len(label))], labels=label)
    plt.tick_params(axis='both', labelsize=15)
    
    plt.xlabel("Vorhergesagte Klasse", fontsize=16)
    plt.ylabel("Wahre Klasse", fontsize=16)
    plt.show()
    
    print(classification_report(y_true, y_pred))


def reconstruct_curves(model, curves, max_train, len_curve, scaled=None):
    
    if scaled == True:
        reconstructed_curves = model.predict(curves)
        reconstructed_curves_inverse = reconstructed_curves.reshape(-1, len_curve) * max_train
      
        print(f"reconstructed_curves_inverse shape: {reconstructed_curves_inverse.shape}")
        return reconstructed_curves_inverse


def calc_mae(feat, recon_feat):
    
    feat_mae = np.mean(np.abs(feat-recon_feat), axis=1)
    return feat_mae 


def calc_mse(feat, recon_feat):

    feat_mse = np.mean(np.square(feat-recon_feat), axis=1)
    return feat_mse


def calc_rmse(feat, recon_feat):

    feat_rmse = np.sqrt(np.mean((feat-recon_feat)**2, axis=1))
    return feat_rmse



def calc_reconstruction_loss(feat, recon_feat):

    mean_rmse = np.mean(calc_rmse(feat, recon_feat))
    mean_mae = np.mean(calc_mae(feat, recon_feat))
    mean_mse = np.mean(calc_mse(feat, recon_feat))

    return mean_rmse, mean_mae, mean_mse


def find_threshold_with_val(start, end, y_val, val_loss, steps = 0.00005):
    
    start = start
    end = end
    steps = steps
    val_threshold = {"threshold": [], "tpr": [], "fpr": [], "f1": [], "acc": []}

    for i in np.arange(start, end, steps):
        val_threshold["threshold"].append(i)
        tpr, fpr, f1, acc = get_predictions(y_val, threshold=i, loss=val_loss)
        val_threshold["tpr"].append(tpr)
        val_threshold["fpr"].append(fpr)
        val_threshold["f1"].append(f1)
        val_threshold["acc"].append(acc)

    return val_threshold


def prepare_train_test(model, x_train, x_test, max_train, len_curve, scaled=None):
    
    if scaled == True:
        reconstructed_train = model.predict(x_train)
        reconstructed_train_inverse = reconstructed_train.reshape(-1, len_curve) * max_train
        
        reconstructed_test = model.predict(x_test)
        reconstructed_test_inverse = reconstructed_test.reshape(-1, len_curve) * max_train
        
        print(f"reconstructed_train_inverse shape: {reconstructed_train_inverse.shape}")
        print(f"reconstructed_test_inverse shape: {reconstructed_test_inverse.shape}")
        
        return reconstructed_train_inverse, reconstructed_test_inverse
        
    if scaled == False:
        reconstructed_train = model.predict(x_train)
        reconstructed_train = reconstructed_train.reshape(-1, len_curve)
        
        reconstructed_test = model.predict(x_test)
        reconstructed_test = reconstructed_test.reshape(-1, len_curve)
        
        print(reconstructed_train.shape)
        print(reconstructed_test.shape)
        
        return reconstructed_train, reconstructed_test



def get_preds(y_test_binary, threshold, loss):

    anomaly_mask = pd.Series(loss) > threshold
    anomaly_prediction = np.array(anomaly_mask.map(lambda x: 1.0 if x == True else 0)).astype(int)  # anomaly=1, ok-process=0
    
    #print(anomaly_prediction)
    cm = confusion_matrix(y_test_binary, anomaly_prediction)
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    tn = cm[0][0]
       
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    f1 = f1_score(y_test_binary, anomaly_prediction)
    acc = accuracy_score(y_test_binary, anomaly_prediction)
     
    return tpr, fpr, f1, acc, cm


def get_predictions(y_test_binary, threshold, loss):

    anomaly_mask = pd.Series(loss) > threshold
    anomaly_prediction = np.array(anomaly_mask.map(lambda x: 1.0 if x == True else 0)).astype(int)  # anomaly=1, ok-process=0
    
    #print(anomaly_prediction)
    cm = confusion_matrix(y_test_binary, anomaly_prediction)
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
       
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    f1 = f1_score(y_test_binary, anomaly_prediction)
    acc = accuracy_score(y_test_binary, anomaly_prediction)
     
    return tpr, fpr, f1, acc


def get_anomaly_pred_acc(y_test_binary, threshold, loss):

    anomaly_mask = pd.Series(loss) > threshold
    anomaly_prediction = np.array(anomaly_mask.map(lambda x: 1.0 if x == True else 0)).astype(int)  # anomaly=1, ok-process=0
    
    #print(anomaly_prediction)
    acc = accuracy_score(y_test_binary, anomaly_prediction)
    rec = recall_score(y_test_binary, anomaly_prediction)
    pre = precision_score(y_test_binary, anomaly_prediction)
    f1 = f1_score(y_test_binary, anomaly_prediction)
    
    print(f"Recall: {rec}" )
    print(f"Precision: {pre}" )
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")

    return rec, pre, f1, acc, anomaly_prediction
    
    
    