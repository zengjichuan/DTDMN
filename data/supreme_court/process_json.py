"""
process supreme court dataset
1. split conversation by case id and after pre
Process data format:
{
    "dataset": "court",
    "version": "0.1",
    "case_lst": [
        [conv1, conv2]
    ]
}
"""
import json
import os
import numpy as np
import copy
import re
from collections import Counter
import random

def read_file(fn):
    with open(fn) as fin:
        lines = fin.readlines()
    case_dict = {}
    conv = []
    case_id = ""
    for line in lines:
        items = line.strip().split(" +++$+++ ")
        case_id = items[0]
        after_pre = items[2]    # TRUE or FALSE
        speaker = items[3]      # name of speaker
        is_justice = items[4]   # JUSTICE or NOT JUSTICE
        fv = items[5]         # PETITIONER,RESPONDENT,NA
        side = items[6]         # PETITIONER, RESPONDENT or empty
        utt = items[7]
        if case_id not in case_dict:
            case_dict[case_id] = []
        if after_pre == "FALSE":
            if len(conv) >= 4:   # filter less than 4 turns
                case_dict[case_id].append(conv)
            conv = []
        elif count_word(utt) > 5:   # content should not less than 5
            conv.append({"is_justice": is_justice, "speaker": speaker, "utt": utt, "side": side, "fv": fv})
    if len(conv) >= 4:
        case_dict[case_id].append(conv)

    return case_dict

def process(case_dict):
    rst_dict = {"dataset": "court", "version": "0.1"}
    data_lst = []
    for case_id, conv_lst in case_dict.items():
        case_d = {}
        case_d["case_id"] = case_id
        case_d["case_convs"] = []
        win_counter = Counter()
        for c_i, conv in enumerate(conv_lst):
            jst_dict = {}
            conv_lst = []

            side_counter = Counter()
            for utt in conv:   # take a try, see which side win
                if utt["is_justice"] == "JUSTICE":
                    if utt["fv"] and utt["speaker"] not in jst_dict:
                        jst_dict[utt["speaker"]] = 1
                        win_counter[utt["fv"]] += 1
                    if utt["side"]:
                        side_counter[utt["side"]] += 1
            if len(side_counter) > 0:
                side = side_counter.most_common(1)[0][0]
                for utt in conv:
                    if utt["is_justice"] == "NOT JUSTICE" and utt["side"] == side:
                        conv_lst.append(utt["utt"]) # only record the side announcement
                if len(conv_lst) > 5:
                    case_d["case_convs"].append({"side": side, "utt_lst": conv_lst, "win": False})
        win_side = win_counter.most_common(1)[0][0]
        for conv in case_d["case_convs"]:
            if conv["side"] == win_side:
                conv["win"] = True
        data_lst.append(case_d)
    num_data = len(data_lst)
    train_lst = data_lst[:int(num_data * 0.8)]
    test_lst = data_lst[int(num_data * 0.8):]
    rst_dict["train"] = train_lst
    rst_dict["test"] = test_lst
    json.dump(rst_dict, open("court.json", "w"), indent=4)


def statistics(case_dict):
    """
    Statistics of dataset
    1. Average # of conv (P,R,J)
    2. Distribution of side in conv
    :param case_dict:
    :return:
    """
    conv_lens = []
    p_nums = []
    j_nums = []
    r_nums = []

    for case_id, conv_lst in case_dict.items():
        for c_i, conv in enumerate(conv_lst):
            conv_len = len(conv)
            j_num = 0
            p_num = 0
            r_num = 0
            j_s_lst = []

            for utt in conv:
                if utt["is_justice"] == "JUSTICE":
                    j_num += 1
                    if utt["side"] == "PETITIONER":
                        j_s_lst.append("P")
                    elif utt["side"] == "RESPONDENT":
                        j_s_lst.append("R")
                elif utt["side"] == "PETITIONER":
                    p_num += 1
                elif utt["side"] == "RESPONDENT":
                    r_num += 1
                else:
                    # print("no justice, no side")
                    pass
            # dist of j_s_lst
            print("".join(j_s_lst))

            conv_lens.append(conv_len)
            p_nums.append(p_num)
            j_nums.append(j_num)
            r_nums.append(r_num)

    print("total # of case: %d" % len(case_dict))
    print("total # of conv: %d" % len(conv_lens))
    print("avg # of conv len: %.3f" % np.mean(conv_lens))
    print("avg # of petitioner utt in conv: %.3f" % np.mean(p_nums))
    print("avg # of respondent utt in conv: %.3f" % np.mean(r_nums))
    print("avg # of justice utt in conv: %.3f" % np.mean(j_nums))

# simple count word by split
def count_word(sent):
    if sent:
        return len(sent.split())
    else:
        return 0

def process_text(words):
    re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{1,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    sent = re.sub(re_url, " <url> ", words)
    sent = re.sub(r"-?\d+(\.\d+)?(?!\w)", "<digit>", sent)
    sent = re.sub(r"&gt;.*\n\n", " <quote> ", sent)     # place the quotation
    return sent

if __name__ == '__main__':
    conv_dict = read_file("origin/supreme.conversations.txt")
    process(conv_dict)
    statistics(conv_dict)