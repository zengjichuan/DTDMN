Supreme Court Dialogs Corpus v1.01 (released September 2012)

Distributed together with:

"Echoes of power: Language effects and power differences in social interaction"
Cristian Danescu-Niculescu-Mizil, Bo Pang, Lillian Lee and Jon Kleinberg
WWW 2012

This corpus builds upon and enriches the data initially used in:

"Computational analysis of the conversational dynamics of the United States Supreme Court"
Timothy Hawes
Master's Thesis, 2009

(Both papers are included in this .zip file.)


NOTE: If you have results to report on this corpus, please send email to cristian@cs.cornell.edu so we can add you to our list of people using this data.  Thanks!


Contents of this README:

	A) Brief description
	B) Files description
	C) Contact

A) Brief description:

This corpus contains a collection of conversations from the U.S. Supreme Court Oral Arguments (http://www.supremecourt.gov/oral_arguments/) with metadata:

- 51,498 utterances making up 50,389 conversational exchanges
- from 204 cases involving 11 Justices and 311 other participants (lawyers or amici curiae)
- metadata includes: 
	- case outcome
	- vote of the Justice 
	- section in which the conversation took place
	- gender annotation

Case outcome and vote data were extracted from the the Spaeth Supreme Court database (http://scdb.wustl.edu/)



B) Files description:


<===> supreme.conversations.txt

This file contains 51,498 utterances, one per line, in the following format (the field separator is: "+++$+++"): 

CASE_ID +++$+++ UTTERANCE_ID +++$+++ AFTER_PREVIOUS +++$+++ SPEAKER +++$+++ IS_JUSTICE +++$+++ JUSTICE_VOTE +++$+++ PRESENTATION_SIDE +++$+++ UTTERANCE

where:

CASE_ID unique id of the case
UTTERANCE_ID unique id of the utterance
AFTER_PREVIOUS in {TRUE, FALSE}: TRUE if this utterance is part of the same conversation as the previous utterances, FALSE if this utterance starts a new conversation
SPEAKER the name of the speaker
IS_JUSTICE in {JUSTICE, NOT JUSTICE}: JUSTICE if the speaker is a Justice, NOT JUSTICE otherwise
JUSTICE_VOTE in {PETITIONER,RESPONDENT,NA}: PETITIONER if the Justice eventually voted for the petitioner, RESPONDENT if the Justice eventually voted for the respondent, NA if the Justice did not vote on this case or the speaker is not a Justice
PRESENTATION_SIDE in {PETITIONER, RESPONDENT}: PETITIONER if the line was uttered in the sections supporting the petitioner's case, RESPONDENT if the line was uttered in the section supporting the respondent's case. (The field is left empty if the utterance appears outside these sections, such as in the preamble of the argument.)  Note that while all Justices can speak in both sections, only lawyers supporting the petitioner's case will speak in the PETITIONER section, and only lawyers supporting the respondent's case will speak in the RESPONDENT section.  A detailed description of the structure of the Oral Arguments can be found at: http://www.supremecourt.gov/oral_arguments/
UTTERANCE the text produced by the speaker


Example lines:

02-1472 +++$+++ 13 +++$+++ TRUE +++$+++ JUSTICE SOUTER +++$+++ JUSTICE +++$+++ PETITIONER +++$+++ PETITIONER +++$+++ So -- so we know -- they -- they knew at the moment the contract was signed what it was going to cost.
02-1472 +++$+++ 14 +++$+++ TRUE +++$+++ MR. MILLER +++$+++ NOT JUSTICE +++$+++ NA +++$+++ PETITIONER +++$+++ That is correct, Justice Souter. In the case of the Shoshone-Paiute contract, when the parties decided to renegotiate the contract amount, they entered into an amendment to specify the new, updated contract amount.

The first line (line 13) was uttered by Justice Souter in the PETITIONER's side of case 02-1472.  The second line (line 14) is Mr. Miller's reply; since Mr. Miller is not a Justice and speaks in the PETITIONER's side, one can infer that Mr. Miller is representing the PETITIONER (lawyers do not speak in the section supporting the opposite case).  Justice Souter eventually voted for the PETITIONER's side supported by Mr. Miller.


----------------------
<===> supreme.outcome.txt

This file indicates the winning side for each case, one case per line, in the following format:

CASE_ID +++$+++ WINNING_SIDE

where:

WINNING_SIDE in {PETITIONER, RESPONDENT}: the side that eventually won the case

Example line:

02-1472 +++$+++ PETITIONER

Case 02-1472 was eventually won by the PETITIONER (represented in the example above by Mr. Miller)

(Note that this information is available for only 196 of the 204 cases.)


--------------------
<===> supreme.votes.txt

This file indicates the votes for each case, one case by line, in the following format

CASE_ID +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR +++$+++ JUSTICE_NAME::SIDE_VOTED_FOR

JUSTICE_NAME in {'THOMAS', 'STEVENS', 'REHNQUIST', 'KENNEDY', 'GINSBURG', 'BREYER', 'OCONNOR', 'ROBERTS', 'SOTOMAYOR', 'ALITO', 'SOUTER', 'KAGAN', 'SCALIA'}: the name of the Justice casting the vote
SIDE_VOTED_FOR in {PETITIONER, RESPONDENT,NA}: the side for which the respective Justice eventually voted

example:

02-1472 +++$+++ THOMAS::PETITIONER +++$+++ STEVENS::PETITIONER +++$+++ KENNEDY::PETITIONER +++$+++ REHNQUIST::NA +++$+++ GINSBURG::PETITIONER +++$+++ OCONNOR::PETITIONER +++$+++ BREYER::PETITIONER +++$+++ SOUTER::PETITIONER +++$+++ SCALIA::PETITIONER

in case 02-1472 all the Justices but Justice Rehnquist voted for the PETITIONER; Justice Rehnquist did not cast a vote.


---------------------
<===> supreme.gender.txt

Gender of the speakers, one per line:

SPEAKER +++$+++ GENDER

where GENDER in {male, female, NA}

examples:

JUSTICE O'CONNOR +++$+++ female
BRUNSTAD +++$+++ male
MS. PERRY +++$+++ female

Gender was automatically derived from the Mr./Ms. prefixes when available, and manually annotated elsewhere.



C) Contact:

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)




This material is based upon work supported in part by the National Science Foundation under grant IIS-0910664.  Any opinions, findings, and conclusions or recommendations expressed above are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.