#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:42:48 2020

@author: carolinepacheco
"""

import argparse
import glob
import logging as log
import math
import os
import sys

import chocolate as choco
import cv2
import numpy as np
import pandas as pd
import pybgs as bgs
from numba import jit

if sys.version_info >= (3, 0):
    from six.moves import xrange

print("OpenCV Version: {}".format(cv2.__version__))


#%%
################################## SPLIT ##################################
import multiprocessing
import time
import queue  # imported for using queue.Empty exception
from multiprocessing import Process, Queue, current_process


def do_job(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
            i = task[0]
            equation = task[1]
            args = task[2]
            args.pname = current_process().name
            print(args.pname + " " + str(i) + " " + equation)
            if args.eval_mutation:
                result = process_structure(i, equation, args)  # best_loss, best_params, best_mutation, max_iterations
            if args.eval_equation:
                result = process_equation(i, equation, args)  # loss
            result = list(result) + [equation]
            print(args.pname + " " + str(i) + " result: " + str(result))
        except queue.Empty:
            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            print(args.pname + " " + equation + " is done")
            tasks_that_are_done.put(result)
            time.sleep(.5)
    return True


# all equations, all cpus available
def main(args):
    filepath = args.file
    if not os.path.isfile(filepath):
        log.error("File " + str(filepath) + " does not exists!")
        sys.exit(-1)

    if args.njobs is not None:
        number_of_processes = args.njobs
    else:
        number_of_processes = multiprocessing.cpu_count()
    print("number_of_processes: " + str(number_of_processes))
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    number_of_task = sum(1 for line in open(filepath))
    print("number_of_task: " + str(number_of_task))

    with open(filepath) as fp:
        for i, equation in enumerate(fp):
            equation = equation.replace("\n", "")
            tasks_to_accomplish.put([i, equation, args])

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    data = []
    while not tasks_that_are_done.empty():
        output = tasks_that_are_done.get()
        if args.eval_mutation:
            best_loss, best_params, best_mutation, max_iterations, equation = output
            data.append([equation, best_mutation, best_loss, max_iterations])
        if args.eval_equation:
            loss, equation = output
            data.append([equation, loss])
    save(data, args=args)
    print("Finished")
    return True


def save(data, i="", args=None):
    print(i, " Summary:")
    if args is None or args.eval_mutation is True:
        df = pd.DataFrame(data, columns=['structure', 'best_mutation', 'loss', 'iterations'])
    if args is not None and args.eval_equation is True:
        df = pd.DataFrame(data, columns=['equation', 'loss'])
    df.to_csv("autolbp.csv", sep='\t', encoding='utf-8', index=False, header=True)
    print(df)


################################# PROCESS #################################
# eval on + numba off = 6sec
# eval off + numba off = 4sec
# eval off + numba on = 1sec

@jit(nopython=True, parallel=True)
def compute_lbp(img_gray, lbp_str, neighboor=3, s=0, a_=0.01):
    img_lbp = np.zeros_like(img_gray)
    a = np.zeros((1, 1, 1), dtype=np.float)
    a[0] = a_
    for ih in range(0, img_gray.shape[0] - neighboor):  # loop by line
        for iw in range(0, img_gray.shape[1] - neighboor):
            # Get matrix image 3 by 3 pixel
            Z = (img_gray[ih:ih + neighboor, iw:iw + neighboor])  # .astype(np.float)
            C = (Z[1, 1])  # float
            s = eval(lbp_str)
            lbp = ((s >= 0) * 1.0)  # .astype(np.uint8)
            img_vector = np.delete(lbp.T.flatten(), 4)
            # Convert the binary operated values to a decimal (digit)
            where_lbp_vector = np.where(img_vector)[0]
            num = np.sum(2 ** where_lbp_vector) if len(where_lbp_vector) >= 1 else 0
            img_lbp[ih + 1, iw + 1] = num
    return img_lbp


@jit(nopython=True, parallel=True)
def compare_fg_gt(img_fg, img_gt, img_gt_path):
    # TP - # of foreground pixels classified as foreground pixels.
    # FP - # of background pixels classified as foreground pixels.
    # TN - # of background pixels classified as background pixels.
    # FN - # of foreground pixels classified as background pixels.
    TP = .0  # True positive pixels
    FP = .0  # False positive pixels
    TN = .0  # True negative pixels
    FN = .0  # False negative pixels
    # blue  = [255, 0, 0]
    green = [0, 255, 0]  # for FN
    red = [0, 0, 255]  # for FP
    white = [255, 255, 255]  # for TP
    black = [0, 0, 0]  # for TN
    rows, cols = img_gt.shape
    img_sc = np.zeros((rows, cols, 3), np.uint8)
    for i in xrange(rows):
        for j in xrange(cols):
            pixel_gt = img_gt[i, j]
            pixel_fg = img_fg[i, j]
            if pixel_gt == 85:
                break
            if (pixel_gt == 255 and pixel_fg == 255):
                TP = TP + 1
                img_sc[i, j] = white
            if (pixel_gt == 0 and pixel_fg == 255):
                FP = FP + 1
                img_sc[i, j] = red
            if (pixel_gt == 0 and pixel_fg == 0):
                TN = TN + 1
                img_sc[i, j] = black
            if (pixel_gt == 255 and pixel_fg == 0):
                FN = FN + 1
                img_sc[i, j] = green
        if pixel_gt == 85:
            break
    return img_sc.copy(), TP, FP, TN, FN


eq_symbols = [chr(i) for i in range(32, 127)]  # 95
ss_symbols = ['+', '-', '*', '/']  # 4
no_symbols = ['(', 'Z', 'C', 'a', ')', ' ', '"', "'", ',', '.', '\\', '[', ']', '`', '^', '{', '}', ':', ';', '-', '=',
              '<', '>', '~', '|', '_'] + ss_symbols  # 30
va_symbols = list(set(eq_symbols) - set(no_symbols))  # 66 (max 33)
va_symbols.sort()


def process_structure(i, equation, args):
    print(args.pname + " processing structure: " + equation)
    eq_split = equation.split("o")  # ['((Z', 'C)', '(Z', 'C))']
    search_space = {}
    new_equation = eq_split[0]
    for j in range(len(eq_split) - 1):
        j_symbol = va_symbols[j]
        search_space[j_symbol] = choco.choice(ss_symbols)
        new_equation = new_equation + j_symbol + eq_split[j + 1]
    # linear:
    # max_iterations = (len(eq_split)-1) * len(ss_symbols)
    #
    # exponential:
    max_iterations = int(math.pow(len(ss_symbols), (len(eq_split) - 1)))
    #
    # limit: 4^5
    if max_iterations > 1024:
        max_iterations = 1024
    #
    print(args.pname + " " + equation + " max_iterations: " + str(max_iterations))
    best_loss, best_params, best_mutation = mutate_equation(i, new_equation, search_space, max_iterations, args)
    return best_loss, best_params, best_mutation, max_iterations


def process_equation(i, equation, args):
    print(args.pname + " processing equation: " + equation)
    loss = eval_equation(i, equation, args)
    return loss


def mutate_equation(i, new_equation, search_space, max_iterations, args):
    database_url = "sqlite:///db/chocolate_" + str(i) + ".db"
    conn = choco.SQLiteConnection(url=database_url)
    conn.clear()
    sampler = choco.MOCMAES(conn, search_space, mu=2)
    best_loss = 1
    best_params = None
    best_equation = None
    try:
        for n in range(max_iterations):
            token, params = sampler.next()
            print(args.pname + " " + new_equation + " iter " + str(n + 1) + " of " + str(max_iterations))
            loss, eq = score_equation(new_equation, args, params)
            if loss < best_loss:
                best_loss = loss
                best_params = params
                best_equation = eq
            # print(n, " loss: ", loss)
            sampler.update(token, loss)
    # except:
    #     pass
    except Exception as err:
        print(args.pname + " unexpected error:\n", err)
    return [best_loss, best_params, best_equation]


def eval_equation(i, equation, args):
    loss = 1
    try:
        loss, eq = score_equation(equation, args)
    except Exception as err:
        print(args.pname + " unexpected error:\n", err)
    return [loss]


# 128x96, 150 images, disabled imshow
# 64x48, 150 images, disabled imshow
def score_equation(equation, args, params=None):
    if params is not None:
        for key in params:
            equation = equation.replace(key, params[key])
        print(args.pname + " scoring mutation: " + equation)
    else:
        print(args.pname + " scoring equation: " + equation)
    img_in_folder = args.inputfolder
    img_in_array = sorted(glob.iglob(img_in_folder + '/*.jpg'))
    print(args.pname + " in folder " + img_in_folder + " " + str(len(img_in_array)))
    img_gt_folder = args.groundtruthfolder
    img_gt_array = sorted(glob.iglob(img_gt_folder + '/*.png'))
    print(args.pname + " gt folder " + img_gt_folder + " " + str(len(img_gt_array)))
    #
    # background subtraction algorithm
    algorithm = bgs.LBP_MRF()
    # print("Running ", algorithm.__class__)
    # TP - # of foreground pixels classified as foreground pixels.
    # FP - # of background pixels classified as foreground pixels.
    # TN - # of background pixels classified as background pixels.
    # FN - # of foreground pixels classified as background pixels.
    # PR - # of true positive pixels / (# of true positive pixels + # of false positive pixels).
    # RE - # of true positive pixels / (# of true positive pixels + # of false negative pixels).
    # FS = 2 × (PR × RE)/(PR + RE).
    # *** Also known as F1 score, F-score or F-measure.
    TP = .0  # True positive pixels
    FP = .0  # False positive pixels
    TN = .0  # True negative pixels
    FN = .0  # False negative pixels
    RE = .0  # TP / (TP + FN)
    PR = .0  # TP / (TP + FP)
    FS = .0  # 2*(PR*RE)/(PR+RE)
    enable_imshow = args.imshow
    # loop x times as files in our folder
    start = time.time()
    for x in range(0, len(img_in_array)):
        start_in = time.time()
        # we can loop now through our array of images
        img_in_path = img_in_array[x]  # input/in000300.jpg
        # read file into opencv
        img_gray = cv2.imread(img_in_path, cv2.IMREAD_GRAYSCALE)
        if args.resize is not None:
            img_gray = cv2.resize(img_gray, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        if x == 0:
            print(args.pname + " img in size " + str(img_gray.shape))
        if enable_imshow:
            cv2.imshow('img_gray', img_gray)
        # compute lbp
        img_gray = img_gray.astype(np.float)
        if args.disable_lbp:
            img_lbp = img_gray.copy()
        else:
            try:
                img_lbp = compute_lbp(img_gray, lbp_str=equation)
            except:
                img_lbp = np.zeros_like(img_gray)
        img_lbp = img_lbp.astype(np.uint8)
        if enable_imshow:
            cv2.imshow('img_lbp', img_lbp)
        # background subtraction
        img_gt_path = img_gt_array[x]
        img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)
        if args.resize is not None:
            img_gt = cv2.resize(img_gt, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_NEAREST)
        if x == 0:
            print(args.pname + " img gt size " + str(img_gt.shape))
        img_fg = algorithm.apply(img_lbp)
        img_bg = algorithm.getBackgroundModel()
        # show images in python imshow window
        if enable_imshow:
            cv2.imshow('img_gt', img_gt)
            cv2.imshow('img_fg', img_fg)
        if (args.skip == -1) or (args.skip > 0 and x > (args.skip - 1)):
            img_sc, tp, fp, tn, fn = compare_fg_gt(img_fg, img_gt, img_gt_path)
        else:
            print(args.pname + " ignoring frame " + str(x))
            img_sc = np.zeros_like(img_gray)
            tp, fp, tn, fn = [0, 0, 0, 0]
        print(args.pname + " " + str(x) + " elapsed time: " + str(time.time() - start_in))
        TP = TP + tp
        FP = FP + fp
        TN = TN + tn
        FN = FN + fn
        if enable_imshow:
            cv2.imshow('img_sc', img_sc)
        if enable_imshow:
            cv2.waitKey(100)
            time.sleep(.1)
        if args.saveoutput:
            if not os.path.isdir(args.output_folder):
                os.mkdir(args.output_folder)
            img_fg_folder = os.path.join(args.output_folder, "fg")
            img_sc_folder = os.path.join(args.output_folder, "sc")
            if not os.path.isdir(img_fg_folder):
                os.mkdir(img_fg_folder)
            if not os.path.isdir(img_sc_folder):
                os.mkdir(img_sc_folder)
            cv2.imwrite(os.path.join(img_fg_folder, os.path.basename(img_in_path)), img_fg)
            cv2.imwrite(os.path.join(img_sc_folder, os.path.basename(img_in_path)), img_sc)
            if not args.disable_lbp:
                img_lbp_folder = os.path.join(args.output_folder, "lbp")
                if not os.path.isdir(img_lbp_folder):
                    os.mkdir(img_lbp_folder)
                cv2.imwrite(os.path.join(img_lbp_folder, os.path.basename(img_in_path)), img_lbp)
        # break
    print(args.pname + " " + equation + " elapsed time " + str(time.time() - start))
    try:
        RE = TP / (TP + FN)
    except:
        pass
    try:
        PR = TP / (TP + FP)
    except:
        pass
    try:
        FS = (2 * PR * RE) / (PR + RE)
    except:
        pass
    loss = (1 - FS)
    print(args.pname + " " + equation + " loss: " + str(loss))
    if args.saveoutput:
        output_file = os.path.join(args.output_folder, "results.txt")
        with open(output_file, "w") as text_file:
            if not args.disable_lbp:
                text_file.write("{0}\n".format(equation))
            text_file.write("RE: {0}\n".format(RE))
            text_file.write("PR: {0}\n".format(PR))
            text_file.write("FS: {0}\n".format(FS))
            text_file.write("loss: {0}\n".format(loss))
    if enable_imshow:
        cv2.waitKey(5000)
        time.sleep(5)
        cv2.destroyAllWindows()
    return [loss, equation]


################################## MAIN ###################################

def print(*args):
    # Print each argument separately so caller doesn't need to
    # stuff everything to be printed into a single string
    for arg in args:
        log.info(arg)


# print = log.info
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="the path to file", required=True)
    parser.add_argument("-i", "--inputfolder", type=str, help="the path to the input folder", required=True)
    parser.add_argument("-g", "--groundtruthfolder", type=str, help="the path to the groundtruth folder", required=True)
    parser.add_argument("-r", "--resize", type=float, help="resize the images between 0 to 1")
    parser.add_argument("-k", "--skip", type=int, help="skip the first N frames", default=-1)
    parser.add_argument("-j", "--njobs", type=int, help="number of parallel processes")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-s", "--imshow", help="enable imshow for visual debugging", action="store_true")
    parser.add_argument("--disable_lbp", help="disable lbp computation (uses grayscale)", action="store_true")
    parser.add_argument("--saveoutput", action="store_true", required=False)
    parser.add_argument("-o", "--output_folder", required='--saveoutput' in sys.argv,
                        help="the path to the output folder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-m", "--eval_mutation", help="mutate and eval equations", action="store_true")
    group.add_argument("-e", "--eval_equation", help="process raw equations", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=log.DEBUG)
        print("Verbosity turned ON")
    else:
        log.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=log.INFO)
    logger = log.getLogger()
    logger.addHandler(log.FileHandler('autolbp.log', 'a'))

    if args.imshow:
        print("Visual debugging is ON")
    if args.resize:
        print("Resize ON")
    if args.eval_mutation:
        print("Mutate and evaluation is ON")
    if args.eval_equation:
        print("Evaluate raw equations is ON")
    if args.file:
        print("Processing: " + args.file)

    main(args)
