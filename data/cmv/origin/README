Cornell ChangeMyView Data v1.0 (released January 2016)

Distributed together with:

Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions
Chenhao Tan, Vlad Niculae, Cristian Danescu-Niculescu-Mizil, Lillian Lee. 
In Proceedings of the 25th International World Wide Web Conference (WWW'2016).

The paper, data, and associated materials can be found at:
http://chenhaot.com/pages/changemyview.html

If you use this data, please cite:
@inproceedings{tan+etal:16a, 
    author = {Chenhao Tan and Vlad Niculae and Cristian Danescu-Niculescu-Mizil and Lillian Lee}, 
    title = {Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions}, 
    year = {2016}, 
    booktitle = {Proceedings of WWW} 
}

The size of the compressed file is 321M. It includes 6 files in 3 directories:

all/
    train_period_data.jsonlist.bz2
    heldout_period_data.jsonlist.bz2

    This directory contains all of the discussion in /r/ChangeMyView during the monitored period, as extracted from the Reddit API. Both files have the same format and store data for the training period (2013/01/01-2015/05/07) and the heldout period (2015/05/08-2015/09/01) respectively. Each line is a json object that includes all information for a submission. Each submission has the following fields:
        domain, banned_by, media_embed, subreddit, selftext_html, selftext, likes, suggested_sort, user_reports, secure_media, link_flair_text, id, from_kind, gilded, archived, clicked, report_reasons, author, media, comments, name, score, approved_by, over_18, hidden, thumbnail, subreddit_id, edited, link_flair_css_class, author_flair_css_class, downs, mod_reports, secure_media_embed, saved, removal_reason, stickied, from, is_self, from_id, permalink, hide_score, created, url, author_flair_text, quarantine, title, created_utc, ups, num_comments, visited, num_reports, distinguished.
    All fields except "comments" come directly from Reddit API. "comments" is a list of replies to the submission. Each reply has the following fields:
        subreddit_id, banned_by, removal_reason, link_id, likes, replies, user_reports, saved, id, gilded, archived, report_reasons, author, parent_id, score, approved_by, controversiality, body, edited, author_flair_css_class, downs, body_html, subreddit, score_hidden, name, created, author_flair_text, created_utc, ups, mod_reports, num_reports, distinguished.

pair_task/
    train_pair_data.jsonlist.bz2
    heldout_pair_data.jsonlist.bz2

    This directory contains pairs of argumentative threads made in reply to the same original post, one successful and one not (more details in Section 4 of the paper). Both files have the same format and store data for training and heldout testing respectively. Each line is a json object for a pair. A pair has the following fields:
        op_author, op_text, op_name, op_title, positive, negative.
    "positive" is a list of replies in a rooted path-unit that won a delta from OP, while "negative" is a matching rooted path-unit that did not win a delta. "op_author", "op_text", and "op_title" give information for the original post. "op_name" is an identifier that can be used to find more submission-related information in the corresponding file in all/.

op_task/
    train_op_data.jsonlist.bz2
    heldout_op_data.jsonlist.bz2

    This directory contains the selected original post (OP) data used to investigate malleability (more details in Section 5 of the paper). Both files have the same format and store data for training and heldout testing respectively. Each line is a json object for an original post. An original post has the following fields:
        selftext, name, title, delta_label.
    "delta_label" shows whether the OP changed her opinion (True if she changed her mind, False if not). "selftext" and "title" provide textual information. "name" is an identifier that can be used to find more submission-related information in the corresponding file in all/.


Please email any questions to: chenhao@chenhaot.com

