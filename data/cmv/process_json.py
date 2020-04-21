"""
data preprocessing for CMV dataset (1,2 follow Tan's paper)
1. len > 30 to avoid trivial response (question clarification)
2. turns should larger than 2
3. # of participants should larger than 10
4. only select persuasive dialog (with delta end), with select non-persuasive dialog

data format:
{
    "dataset": "cmv,
    "version": "0.1",
    "data_lst": [
        {
            "title": ,
            "content": ,
            "author": ,
            "comments":
                [
                    {"utt_lst": [ , , ], "win": True or False},

                ]
            ,
        },
        {},
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

rst_json = {"dataset": "cmv", "version": "0.1", "data_lst": []}



def read_file(fn):
    rst_lst = []
    turn_lens = []
    delta_op = 0
    delta_cm = 0
    num_turns = 0
    num_conv = 0
    with open(fn) as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            json_data = json.loads(line)
            ori_content = process_text(json_data["selftext"])
            ori_title = process_title(json_data["title"])
            op_delta = json_data["link_flair_text"]
            op_id = json_data["name"]
            num_comments = json_data["num_comments"]
            op_author = process_text(json_data["author"])
            # get comments
            if not json_data["comments"]:
                continue
            # process comments
            comments, delta_cm_ = process_comments(json_data["comments"], op_author, op_id)
            turn_lens.extend([len(comment["utt_lst"]) for comment in comments])
            num_turns += sum([len(comment["utt_lst"]) for comment in comments])
            num_conv += len(comments)
            delta_cm += delta_cm_
            # if op_delta == "[Deltas Awarded]":
            if delta_cm_ > 0:
                delta_op += 1
            if delta_cm_ == 0 or len(comments) == 0:
                continue
            # if delta_cm > 0 and op_delta != "[Deltas Awarded]":
            #     print("something happen!")
            rst_lst.append({
                "title": ori_title,
                "content": ori_content,
                # "delta": op_delta == "[Deltas Awarded]" and True or False,
                "author": op_author,
                "num_comments": num_comments,
                "comments": comments,
            })
    len_hist = Counter(turn_lens)
    print(len_hist)
    print("num of dialogs: %d, num of delta dialogs: %d(%.4f), num of turns %d, num of delta turns: %d(%.4f), "
          "num of avg num of turns per dialog: %.4f" % (num_conv, delta_op, float(delta_cm)/num_conv, num_turns,
                                             delta_cm, float(delta_cm)/num_turns, float(np.mean(turn_lens))))
    return rst_lst


def process_text(words):
    re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{1,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    sent = re.sub(re_url, " <url> ", words)
    sent = re.sub(r"-?\d+(\.\d+)?(?!\w)", "<digit>", sent)
    sent = re.sub(r"&gt;.*\n\n", " <quote> ", sent)     # place the quotation
    return sent

def process_title(title):
    return re.sub(r"CMV:|cmv:", "", title).strip()

def dfs_iter(conv_tree, paths):
    stack = [(conv_tree, [])]
    while stack:
        tree, path = stack.pop()
        if len(tree) == 0:
            # print(path)
            paths.append(path)
        else:
            for id, sub_tree in tree.items():
                    stack.append((sub_tree, path + [id]))

# simple count word by split
def count_word(sent):
    if sent:
        return len(sent.split())
    else:
        return 0

def process_comments(json_lst, op_author, op_id):
    # record cid, pid
    id_lst = []
    ele_dict = {}   # for quick lookup
    author_dict = Counter()
    delta_cm = 0
    for json_dict in json_lst:
        cid = json_dict["name"]
        pid = json_dict["parent_id"]
        content = process_text(json_dict.get("body")) if json_dict.get("body") else None
        author = process_text(json_dict.get("author")) if json_dict.get("author") else None
        if author != "DeltaBot":
            author_dict[author] += 1
        ele = {"cid": cid, "pid": pid, "author": author, "content": content, "delta": False}
        ele_dict[cid] = ele
        id_lst.append((cid, pid))

    # find out delta
    for cid, ele in ele_dict.items():
        if ele["author"] == "DeltaBot" and ele["content"].startswith("Confirmed:"):
            # trace back
            delta_author = ele["content"].split()[5][3:-1]
            cur_ele = ele
            while cur_ele["pid"] != op_id:
                if cur_ele["pid"] not in ele_dict:
                    print("cannot find parent node, corrupt tree")
                    break
                parent_ele = ele_dict[cur_ele["pid"]]
                if parent_ele["author"] == delta_author:
                    parent_ele["delta"] = True
                    delta_cm += 1
                    break
                else:
                    cur_ele = parent_ele    # roll back
    # filtering participents less than 10 (exclude op) and no delta post
    if delta_cm == 0 or len(author_dict) <= 5:
        return [], 0
    # construct tree
    conv_tree_dict = {op_id: {}}
    conv_pos_dict = {op_id: conv_tree_dict[op_id]}
    remain_set = set(id_lst)
    last_set_len = len(remain_set) + 1
    while len(remain_set) != 0 and last_set_len != len(remain_set):
        last_set_len = len(remain_set)
        new_re_set = remain_set.copy()
        for cid, pid in remain_set:
            if pid in conv_pos_dict:
                conv_pos_dict[pid][cid] = {}
                conv_pos_dict[cid] = conv_pos_dict[pid][cid]
                new_re_set.remove((cid, pid))
        remain_set = new_re_set

    # remove the descendents of delta
    for cid, pos in conv_pos_dict.items():
        if cid != op_id and ele_dict[cid]["delta"]:
            conv_pos_dict[cid].clear()

    paths = []
    dfs_iter(conv_tree_dict, paths)
    # reconstruct tree
    comment_lst = []
    cid_recorder = []
    remain_lst = []
    remain_dict = {}
    for path in paths:
        assert len(path) > 1
        post_lst = []
        for id in path[1:]: # elem #0 is op
            ele = ele_dict[id]
            if ele["author"] != op_author and count_word(ele["content"]) > 30 or ele["delta"]:
                post_lst.append(ele)

        if len(post_lst) > 1:   # at least 3 turns and
            if post_lst[-1]['delta']:   # end with delta post
                comment_lst.append({"utt_lst": [post["content"] for post in post_lst], "win": True})
                cid_recorder.append([term['cid'] for term in post_lst])     # record the root node
            else:
                # should avoid duplicate terms after filtering
                cid_str = "".join([term['cid'] for term in post_lst])
                if cid_str not in remain_dict:
                    remain_dict[cid_str] = 1
                    remain_lst.append(post_lst)
    # random pick another 5 conversation for negative label, similarity should be less than 0.5
    if len(comment_lst) > 0:  # more than 1 delta
        remain_sample = 10
        delta_cm = len(comment_lst)
        while len(remain_lst) > 0 and remain_sample > 0:
            idx = np.random.randint(len(remain_lst))

            if get_sim([term['cid'] for term in remain_lst[idx]], cid_recorder) <= 0.5:
                comment_lst.append({"utt_lst": [post["content"] for post in remain_lst[idx]], "win": False})
                remain_sample -= 1
                del remain_lst[idx]
            else:
                del remain_lst[idx]
    return comment_lst, delta_cm


def get_sim(cid_lst1, cid_recorder):
    max = 0
    for cid_lst in cid_recorder:
        sim = len(set(cid_lst1) & set(cid_lst)) / float(len(cid_lst1))
        if max < sim:
            max = sim
    return max


train_lst = read_file(os.path.join("all", "train_period_data.jsonlist"))
test_lst = read_file(os.path.join("all", "heldout_period_data.jsonlist"))

# resplit data to 80/20
all_lst = train_lst + test_lst
num_data = len(all_lst)
# shuffle
random.shuffle(all_lst)

train_lst = all_lst[:int(num_data * 0.8)]
test_lst = all_lst[int(num_data * 0.8):]


rst_json["train"] = train_lst
rst_json["test"] = test_lst
with open("cmv.json", "w") as fout:
    json.dump(rst_json, fout, indent=4)
