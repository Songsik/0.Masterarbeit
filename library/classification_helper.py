import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, hamming_loss
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from tqdm.notebook import tqdm_notebook as tq
from warnings import filterwarnings
filterwarnings('ignore')
import importlib
from library import faps_color as fapsc
from library import etl_helper as etl



def plot_calc_cm(y_true, y_pred, class_name, cm=np.array([0]), title=None, figsize=(7,7), dpi=100, fontsize=20, fontsize_cm=15):
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if cm.sum()==0:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm=cm
        
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(ax=ax, colorbar=False, cmap="Greens")

    plt.xticks(ticks=[x for x in range(len(class_name))], labels=class_name)
    plt.yticks(ticks=[x for x in range(len(class_name))], labels=class_name)
    plt.tick_params(axis='both', labelsize=fontsize)
    
    plt.title(title, fontsize=fontsize+8)
    plt.xlabel("Vorhergesagte Klasse", fontsize=fontsize)
    plt.ylabel("Wahre Klasse", fontsize=fontsize)
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['font.size'] = fontsize + fontsize_cm
    plt.show()
    
    #print(classification_report(y_true, y_pred))



def plot_history(history, figsize=(6,4)):

    plot_loss(history, figsize=figsize)
    get_metric_values(history, figsize=figsize)


def plot_loss(history, figsize=(6,4)):

    plt.figure(figsize=(figsize))
    plt.title('Loss', fontsize=30)
    plt.plot(history.history['loss'], label='train loss', linewidth=3)
    plt.plot(history.history['val_loss'], label='val loss', linewidth=3)
    plt.legend(loc='upper right', fontsize=20)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()


def get_metric_values(history, figsize=(6,4)):

    if len(list(history.history.keys())) == 4:
        met1train = str(list(history.history.keys())[1])
        met1val = str(list(history.history.keys())[1 + len(list(history.history.keys()))//2])
        plot_metric(history, met1train, met1val, figsize)

    elif len(list(history.history.keys())) == 6:

        met1train = str(list(history.history.keys())[1])
        met1val = str(list(history.history.keys())[1 + len(list(history.history.keys()))//2])
        plot_metric(history, met1train, met1val, figsize)

        met2train = str(list(history.history.keys())[2])
        met2val = str(list(history.history.keys())[2 + len(list(history.history.keys()))//2])
        plot_metric(history, met2train, met2val, figsize)


def plot_metric(history, met_train, met_val, figsize):

    plt.figure(figsize=figsize)
    #plt.title('Accuracy', fontsize=30)
    plt.plot(history.history[met_train], label=met_train, linewidth=3)
    plt.plot(history.history[met_val], label=met_val, linewidth=3)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()


def transform_multilabel(labelrow):
    return str([i for i,j in enumerate(labelrow) if j>0]).strip('[]').replace(",", "").replace(" ", "")


def check_multilabel(y_true):
    
    labels = {"class0":0, "class1":0,"class2":0,"class3":0,"class4":0,"class5":0,"class6":0,"class7":0,
              "class16":0,"class24":0,"class35":0,"class37":0}

    for row in y_true:
        if transform_multilabel(row)=="0":
            labels["class0"] += 1
        if transform_multilabel(row)=="1":
            labels["class1"] += 1
        if transform_multilabel(row)=="2":
            labels["class2"] += 1
        if transform_multilabel(row)=="3":
            labels["class3"] += 1
        if transform_multilabel(row)=="4":
            labels["class4"] += 1
        if transform_multilabel(row)=="5":
            labels["class5"] += 1
        if transform_multilabel(row)=="6":
            labels["class6"] += 1
        if transform_multilabel(row)=="7":
            labels["class7"] += 1
        if transform_multilabel(row)=="16":
            labels["class16"] += 1
        if transform_multilabel(row)=="24":
            labels["class24"] += 1
        if transform_multilabel(row)=="35":
            labels["class35"] += 1
        if transform_multilabel(row)=="37":
            labels["class37"] += 1

    for i in labels:
        print(f"number of {i}: {labels[i]:2}")

    for i in range(y_true.shape[1]):
        if i==0:
            print("")
        else:
            print(f"occurence of class {i}: {y_true[:,i].sum()}")



def plot_loss_acc_parallel(history, figsize=(20,7), dpi=80):

    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    ax[0].set_title('Loss', fontsize=30)
    ax[0].plot(history.history['loss'], label='train loss', linewidth=4)
    ax[0].plot(history.history['val_loss'], label='val loss', linewidth=4)
    ax[0].legend(loc='upper right', fontsize=24)
    ax[0].tick_params(axis="x", labelsize=20)
    ax[0].tick_params(axis="y", labelsize=20)
    ax[0].set_xlabel("epochs", fontsize=24)

    ax[1].set_title('Accuracy', fontsize=30)
    ax[1].plot(history.history['accuracy'], label='train acc', linewidth=4)
    ax[1].plot(history.history['val_accuracy'], label='val acc', linewidth=4)
    ax[1].legend(loc='lower right', fontsize=24)
    ax[1].tick_params(axis="x", labelsize=20)
    ax[1].tick_params(axis="y", labelsize=20)
    ax[1].set_xlabel("epochs", fontsize=24)

    fig.tight_layout(pad=3)
    plt.show()


def make_pred_mlc(prediction, threshold):
    
    pred = prediction >= threshold
    pred_round = pred.astype(int)

    return pred_round


def mlc_result(y_true, y_pred):
    
    acc = accuracy_score(y_true, y_pred)
    ham = hamming_loss(y_true, y_pred)
    
    print(f"\n")
    print(f"acc: {acc}")
    print(f"hamming loss: {ham}\n")
    print(classification_report(y_true, y_pred, digits=3))
    
    return acc, ham


def convert_and_zip_label(y_true, y_pred):

    y_true_converted = []
    y_pred_converted = []

    for val in zip(y_true, y_pred):
        y_true_converted.append(str([i for i,j in enumerate(val[0]) if j>0]).strip('[]').replace(",", "").replace(" ", ""))

        if val[1].sum()==0:
            y_pred_converted.append("noclass")
        else:
            y_pred_converted.append(str([i for i,j in enumerate(val[1]) if j>0]).strip('[]').replace(",", "").replace(" ", ""))
            
    zipped = np.column_stack((np.array(y_true_converted), np.array(y_pred_converted)))
    df = pd.DataFrame(zipped, columns=['ytrue', 'ypred'])
    df['len'] = df['ypred'].str.len()
    df = df.sort_values(by=['len', 'ypred'], ascending=[True, True])
    df.drop(columns=['len'], inplace=True)
    
    return df


def cm_style(cm, fontsizestr="16px"):
    th_css = [
        {"selector": "th",
            "props": f"font-size:{fontsizestr}; font-weight: bold"
        },
        {"selector" :"td",
             "props": f"font-size:{fontsizestr}; font-weight: bold"
        }]

    theme = sns.light_palette("green", as_cmap=True)
    cm = cm.style.background_gradient(cmap=theme)

    #s = s.style
    cm = cm.set_precision(1).set_table_styles(th_css)
    return cm


def cm_multi(dfzip, fontsizestr="16px"):
    
    y_true_name = sorted(np.unique(dfzip["ytrue"]), key=len)
    y_pred_name = sorted(np.unique(dfzip["ypred"]), key=len)
    
    cm = pd.crosstab(pd.Series(dfzip["ytrue"], name="True"), pd.Series(dfzip["ypred"], name="Pred"))
    cm = cm.reindex(index=y_true_name, columns=y_pred_name)
    cm.loc[:,'Total'] = cm.sum(numeric_only=True, axis=1)
    cm = cm_style(cm, fontsizestr)
    return cm




def find_wrong_classification(y_true, y_pred, x_test, df_list):
    
    misclassified = {"true_label_0": [], "true_label_1": [], "true_label_2": [],
                     "true_label_3": [], "true_label_4": [], "true_label_5": []}
    
    for num, val in enumerate(zip(y_true, y_pred)):
        if val[0]!=val[1]:
            misclassified[f"true_label_{val[0]}"].append(num)
            
    df_x_test = pd.DataFrame(x_test.reshape(-1, x_test.shape[1]).transpose())
    df_miss = pd.DataFrame()

    for key in misclassified:
        df_miss = pd.concat([df_miss, df_x_test[misclassified[key]]], axis=1)

    misclassified_curve_index_each_df = find_curve_index_in_each_df(misclassified, df_miss, df_list)

    return misclassified_curve_index_each_df, misclassified, df_miss, df_x_test



def find_curve_index_in_each_df(misclassified, df_miss, df_list):

    misclassified_curve_index_each_df = {"df0": [], "df1": [], "df2": [],
                                         "df3": [], "df4": [], "df5": []}

    for num, key in enumerate(misclassified):
        if misclassified[key]==[]:
            pass
        
        elif (misclassified[key]!=[]) & (num==0):
            for col1 in df_miss:
                for col2 in df_list[0]:
                    if round(df_miss.iloc[:len(df_list[0])][col1], 4).equals(round(df_list[0][col2], 4)):
                        misclassified_curve_index_each_df["df0"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==1):
            for col1 in df_miss:
                for col2 in df_list[1]:
                    if round(df_miss.iloc[:len(df_list[1])][col1], 4).equals(round(df_list[1][col2], 4)):
                        misclassified_curve_index_each_df["df1"].append(col2)
            
        elif (misclassified[key]!=[]) & (num==2):
            for col1 in df_miss:
                for col2 in df_list[2]:
                    if round(df_miss.iloc[:len(df_list[2])][col1], 4).equals(round(df_list[2][col2], 4)):
                        misclassified_curve_index_each_df["df2"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==3):
            for col1 in df_miss:
                for col2 in df_list[3]:
                    if round(df_miss.iloc[:len(df_list[3])][col1], 4).equals(round(df_list[3][col2], 4)):
                        misclassified_curve_index_each_df["df3"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==4):
            for col1 in df_miss:
                for col2 in df_list[4]:
                    if round(df_miss.iloc[:len(df_list[4])][col1], 4).equals(round(df_list[4][col2], 4)):
                        misclassified_curve_index_each_df["df4"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==5):
            for col1 in df_miss:
                for col2 in df_list[5]:
                    if round(df_miss.iloc[:len(df_list[5])][col1], 4).equals(round(df_list[5][col2], 4)):
                        misclassified_curve_index_each_df["df5"].append(col2)
                    
    return misclassified_curve_index_each_df
    
        


def final_evaluation(testing_model, feature, label, num_trials, epochs, batch_size, param_grid):

    result_dict = {"acc": [],
                   "rec": [],
                   "f1" : []}

    for i in tq(range(num_trials)):
        print(f"Start {i}ter Lauf=========================================================================================")

        xtrain, xtest, ytrain, ytest = train_test_split(feature, label, random_state=i, test_size=0.25, shuffle=True)
        feature_shuffle = np.concatenate((xtrain, xtest))
        label_shuffle = np.concatenate((ytrain, ytest))
        
        cv_outer = KFold(n_splits=4, shuffle=True, random_state=1)

        for train, test in cv_outer.split(label_shuffle):
            x_train, x_test = feature_shuffle[train, :], feature_shuffle[test, :]
            y_train, y_test = label_shuffle[train], label_shuffle[test]

            cv_inner = KFold(n_splits=4, shuffle=True, random_state=1)

            model = KerasClassifier(build_fn=testing_model, epochs=epochs, batch_size=batch_size, verbose=1)

            clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_inner, refit=True)
            result = clf.fit(x_train, y_train)
            best_model = result.best_estimator_

            y_true = np.argmax(y_test, axis=1)
            y_pred = best_model.predict(x_test)

            result_dict['acc'].append(accuracy_score(y_true, y_pred))
            result_dict['rec'].append(recall_score(y_true, y_pred, average='weighted'))
            result_dict['f1'].append(f1_score(y_true, y_pred, average='weighted'))
                
        print(f"Ende {i}ter Lauf=========================================================================================\n")


    return result_dict



def box_plot_color(ax, data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops={"marker":"o",
                               "markerfacecolor":"white", 
                               "markeredgecolor":fapsc.black,
                               "markersize":"10"})
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp


def boxplot_model_results(result_dict, title, xlist, color, size=(8, 6), dpi=80, lower=None, upper=None):

    #data = [result_dict["acc"], result_dict["rec"], result_dict["f1"]]
    #data_name = ["Accuracy", "Weighted_Recall", "Weighted_F1"]
    data = []
    for key in result_dict:
        data.append(result_dict[key])
    data_name = xlist

    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    bp = box_plot_color(ax, data, fapsc.black, color)

    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel('Ergebnisse der Testdaten', fontsize=15)
    ax.set_ylim(lower, upper)
    ax.set_xticklabels(data_name)


    ax.yaxis.grid()
    ax.tick_params(axis='both', labelsize=15)

    plt.show()




def mlc_cnn_structure(conv, maxpool, bn_conv, dropout_conv, dense, bn_dense, dropout_dense, strides):
    
    model = Sequential()
    
    if conv >= 1:
        model.add(Conv1D(32, kernel_size=8, strides=strides, activation="relu", input_shape=(920, 1)))
        if maxpool[0]:
            model.add(MaxPooling1D(3))
        if bn_conv[0]:
            model.add(BatchNormalization())
        if dropout_conv[0]:
            model.add(Dropout(0.1))
        
    if conv >= 2:
        model.add(Conv1D(32, kernel_size=8, strides=strides, activation="relu"))
        if maxpool[1]:
            model.add(MaxPooling1D(3))
        if bn_conv[1]:
            model.add(BatchNormalization())
        if dropout_conv[1]:
            model.add(Dropout(0.1))
    
    if conv >= 3:
        model.add(Conv1D(32, kernel_size=8, strides=strides, activation="relu"))
        if maxpool[2]:
            model.add(MaxPooling1D(3))
        if bn_conv[2]:
            model.add(BatchNormalization())
        if dropout_conv[2]:
            model.add(Dropout(0.1))
            
    if conv >= 4:
        model.add(Conv1D(32, kernel_size=8, strides=strides, activation="relu"))
        if maxpool[3]:
            model.add(MaxPooling1D(3))
        if bn_conv[3]:
            model.add(BatchNormalization())
        if dropout_conv[3]:
            model.add(Dropout(0.1))

    
    model.add(Flatten())
    
    if dense >= 1:
        model.add(Dense(80, activation="relu"))
        if bn_dense[0]:
            model.add(BatchNormalization())
        if dropout_dense[0]:
            model.add(Dropout(0.1))
            
    if dense >= 2:
        model.add(Dense(80, activation="relu"))
        if bn_dense[1]:
            model.add(BatchNormalization())
        if dropout_dense[1]:
            model.add(Dropout(0.1))
    
    if dense >= 3:
        model.add(Dense(80, activation="relu"))
        if bn_dense[2]:
            model.add(BatchNormalization())
        if dropout_dense[2]:
            model.add(Dropout(0.1))
            
    if dense >= 4:
        model.add(Dense(80, activation="relu"))
        if bn_dense[3]:
            model.add(BatchNormalization())
        if dropout_dense[3]:
            model.add(Dropout(0.1))

    model.add(Dense(8, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    print(model.summary())
    return model


def mlc_cnn_structure_optimization(model_dict, x_train, y_train, epochs, batch_size, patience, result_file_name): #file name as .json file

    model = KerasClassifier(build_fn=mlc_cnn_structure, epochs=epochs, batch_size=batch_size)

    clf = GridSearchCV(estimator=model, param_grid=model_dict, cv=5, refit=False, verbose=10) #scoring_default -> accuracy
    result = clf.fit(x_train, y_train, validation_split=0.2,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=1)]) 

    # save model structure and cv results as mean test score in acc
    if os.path.exists(f"results/{result_file_name}") == False:
        with open(f"results/{result_file_name}", "w") as f:
            json.dump([model_dict, result.cv_results_["mean_test_score"][0]], f)
            f.close()
    else: 
        with open(f"results/{result_file_name}", "r") as f:
            data = json.load(f)
            data.append(model_dict)
            data.append(result.cv_results_["mean_test_score"][0])

        with open(f"results/{result_file_name}", "w") as f:
            json.dump(data, f)
            f.close()

    print(result.cv_results_["mean_test_score"][0])
    #return pd.DataFrame(result.cv_results_)




# gets the score from the cv train log
def get_score(cvtextfile, cv_iter, cv=5):

    score = {"acc1": [], "acc2": [], "acc3": [], "acc4": [], "acc5": [], "mean_acc": []}

    for l in tq(cvtextfile):
        for i in range(1, cv_iter):
            for j in range(1, cv+1):
                if (f"[CV {j}/5; {i}/500]" in l) & ("score" in l):
                    if j==1:
                        score["acc1"].append(float(l.split("score")[1].split()[0].replace("=", "")))
                    if j==2:
                        score["acc2"].append(float(l.split("score")[1].split()[0].replace("=", "")))
                    if j==3:
                        score["acc3"].append(float(l.split("score")[1].split()[0].replace("=", "")))
                    if j==4:
                        score["acc4"].append(float(l.split("score")[1].split()[0].replace("=", "")))
                    if j==5:
                        score["acc5"].append(float(l.split("score")[1].split()[0].replace("=", "")))
                        score["mean_acc"].append((score["acc1"][-1]+score["acc2"][-1]+score["acc3"][-1]+score["acc4"][-1]+score["acc5"][-1])/5)
                else:
                    pass
                
    for key in score.keys():
        if key != "acc5":
            while len(score["acc5"]) < len(score[key]):
                score[key].pop()
                print(f"one value removed from {key}")
                
    dfscore = pd.DataFrame.from_dict(score)            
    return dfscore



# gets the hyperparameters from cv train log
def get_hp(cvtextfile, cv_iter):

    hp = {"batch_size":[], "dense1":[], "dense2":[], "dense3":[], "drop1":[], "drop2":[], "filter1":[], "filter2":[],
          "filter3":[], "kernel1":[], "kernel2":[], "kernel3":[], "learning_rate":[], "maxpool":[]}

    for l in tq(cvtextfile):
        for i in range(1, cv_iter):

            if (f"[CV 1/5; {i}/500]" in l) & ("score" in l):
                for key in hp.keys():
                    if (key!="drop1") & (key!="drop2") & (key!="learning_rate") & (key!="score") & (key!= "maxpool"):
                        hp[key].append(int(l.split(key)[1].split()[0].replace("=", "").replace(",", "")))

                    if (key=="drop1") | (key=="drop2") | (key=="learning_rate"):
                        hp[key].append(float(l.split(key)[1].split(",")[0].replace("=", "")))

                    if key=="maxpool":
                        hp[key].append(int(l.split(key)[1].split()[0].replace("=", "").replace(",", "").replace(";", "")))
    
    dfhp = pd.DataFrame.from_dict(hp)
    return dfhp



def collect_mcc_results(y_true, y_pred, algorithm, params, best_param, mean_val_acc):
    
    res = {"algorithm":algorithm, "params":params, "best_param":best_param, "mean_val_acc":mean_val_acc,
           "rec_0":None, "rec_macro":None, "f1_0":None, "f1_macro":None, "pre_macro":None, "acc":None}
    
    y_pred0 = (y_pred>0).astype(int)
    y_true0 = (y_true>0).astype(int)
    
    res["rec_0"] = recall_score(y_true0, y_pred0)
    res["f1_0"] = f1_score(y_true0, y_pred0)
    res["rec_macro"]= recall_score(y_true, y_pred, average='macro')
    res["f1_macro"]= f1_score(y_true, y_pred, average='macro')
    res["pre_macro"]= precision_score(y_true, y_pred, average='macro')
    res["acc"] = accuracy_score(y_true, y_pred)
    
    for val in res.keys():
        if res[val]==None:
            print("somethong is wrong")
    
    return res


def grouped_barplot(leftlist, leftlabel, leftcolor, rightlist, rightlabel, rightcolor, title, xtickslist, ylim=(0,1), 
                    size=(8,6), dpi=100, fontsize=20):

    labels = xtickslist
    left = leftlist
    right = rightlist

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    rects1 = ax.bar(x - width/2, left, width, color=leftcolor, label=leftlabel)
    ax.bar_label(rects1, padding=3, fontsize=fontsize)

    rects2 = ax.bar(x + width/2, right, width, color=rightcolor, label=rightlabel)
    ax.bar_label(rects2, padding=3, fontsize=fontsize)

    ax.set_title(title, fontsize=fontsize+10)
    ax.set_ylabel(f"Results in %", fontsize=fontsize)
    ax.set_xticks(x, labels, fontsize=fontsize)
    ax.set_ylim(ylim)

    plt.rcParams['font.family'] = ['Arial']
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    #plt.tight_layout(rect=(0, 0, 0.8, 0.8))
    plt.show()



def single_barplot(values, valuecolor, title, xtickslist, size=(8,6), width=0.4, rotation=45, dpi=100, ylim=1.25):

    labels = xtickslist

    x_position = np.arange(len(labels))  # the label, bar locations
    
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    rects = ax.bar(x_position, values, width, color=valuecolor)
    ax.bar_label(rects, padding=3, fontsize=12)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(f"Mean 5 Fold Cross\n Validation Accuracy", fontsize=14)
    ax.set_xticks(x_position, labels, fontsize=12, rotation=rotation)
    ax.set_ylim(0, ylim)

    #plt.legend(loc='upper right', fontsize=12)

    plt.show()


def calc_metrics_occ(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1,1):
        npv = 1
        tnr = 1
    else: 
        tp = cm[1][1]
        fn = cm[1][0]
        fp = cm[0][1]
        tn = cm[0][0]
        # negative predictive value
        npv = tn/(tn+fn) 
        # true negative rate
        tnr = tn/(tn+fp)
    
    return npv, tnr



def collect_results_occ(y_true, y_pred, algorithm, params, best_param, cm):
    
    res = {"algorithm":algorithm, "params":params, "best_param":best_param, "cm":cm,
           "rec":None, "pre":None, "f1":None, "acc":None}

    res["rec"] = recall_score(y_true, y_pred)
    res["pre"] = precision_score(y_true, y_pred)
    res["f1"] = f1_score(y_true, y_pred)
    res["acc"] = accuracy_score(y_true, y_pred)
    
    return res


# swaping the zeros and ones of the label
# zero->no defect; one->defect 
def swap_zero_one(lab, cl_nr):  #cl_nr: 0-7
    
    y = lab.copy()[:,cl_nr]  #.copy() is important, without it the original label will be changed
    y[y==1] = -1
    y[y==0] = 1
    y[y==-1] = 0
    
    return y


def collect_results_mlc(y_true, y_pred, algorithm, params, best_param):
    
    res = {"algorithm":algorithm, "params":params, "best_param":best_param,
           "rec_0":None, "pre_0":None, "f1_0":None, "rec_weighted":None, "pre_weighted":None, "f1_weighted":None, "acc":None}
    
    y_true0 = swap_zero_one(y_true, 0)
    y_pred0 = swap_zero_one(y_pred, 0)
    
    res["rec_0"] = recall_score(y_true0, y_pred0)
    res["pre_0"] = precision_score(y_true0, y_pred0)
    res["f1_0"] = f1_score(y_true0, y_pred0)
    res["rec_weighted"]= recall_score(y_true, y_pred, average='weighted')
    res["f1_weighted"]= f1_score(y_true, y_pred, average='weighted')
    res["pre_weighted"]= precision_score(y_true, y_pred, average='weighted')
    res["acc"] = accuracy_score(y_true, y_pred)
    
    return res


def save_results_json(filename, key_algorithm, hyperopt_result):
    
    if os.path.exists(f"results/{filename}") == False:
        with open(f"results/{filename}", "w") as f:
            json.dump({f"{key_algorithm}":hyperopt_result}, f)
            f.close()
        print(f"file {filename} new created and results of {key_algorithm} saved.")
    else:
        with open(f"results/{filename}", "r") as f:
            data = json.load(f)
            data.update({f"{key_algorithm}":hyperopt_result})

        with open(f"results/{filename}", "w") as f:
            json.dump(data, f)
            f.close()
        print(f"existing file {filename} opened and results of {key_algorithm} saved.") 



def find_misclassification_with_duplicates(y_test, y_pred, df_list, wrong_pred_dict, x_test_df, rounding=4):
    
    for num, key in enumerate(wrong_pred_dict):
        for col1, val in enumerate(zip(y_test, y_pred)):
            if num==0:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==1:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:920][col1], rounding).equals(round(df_list[num].iloc[:920][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==2:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==3:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==4:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==5:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])

            elif num==6:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])

            elif num==7:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
    
    return wrong_pred_dict



def plot_wrong_preds(dfa, dfwrong, colora, labela, dfb, colorb, labelb, figsize=(8,4), dpi=80):

    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    ax[0].plot(etl.set_time(dfa), dfa[dfwrong].values, color=colora, linewidth=3)
    ax[0].plot([], [], color=colora, label=labela)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].legend(loc="upper left", fontsize=16)
    ax[0].grid()
    ax[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    ax[1].plot(etl.set_time(dfb), dfb.values, color=colorb, linewidth=3)
    ax[1].plot([], [], color=colorb, label=labelb)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].legend(loc="upper left", fontsize=16)
    ax[1].grid()
    ax[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.tight_layout(pad=1)
    plt.show()