import pyopenms
import os
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt

# Reference dictionary for annotations, sub-types, and tissues
reference_dict = {
    '20221222_GLIOMIC_MSMS_1016-75_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_1063-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_1077-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_362-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_600-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_604-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_615-15_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_619-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_621-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_623-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_628-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_638-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_647-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_651-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_665-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_666-05_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_673-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_697-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_699-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_701-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_706-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_713-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_725-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PA 38:3', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_734-95_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_747-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PG 34:1', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_751-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PA 40:4', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_755-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_757-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_763-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_785-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_788-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_806-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_810-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_834-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 40:6', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_857-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_876-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_885-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_888-65_NEG.mzML': {'Tissus': '', 'Annotations': 'SHexCer d42:2', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_892-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_901-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_902-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 44:0', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_904-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PC 42:7', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_906-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PC 42:6', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_913-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_918-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_920-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_934-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_957-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_965-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230102_GLIOMIC_MSMS_902-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 44:0', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_621-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_625-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_631-15_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_649-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_727-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_766-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_766-55_NEG2.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_802-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PC 38:7', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_824-75_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_835-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_917-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_931-95_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_934-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_1018-75_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_1028-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_609-15_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_628-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_628-5_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_687-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PA P-36:0', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_694-75_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_715-15_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_744-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_780-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_785-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_794-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 40:4 ', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_804-35_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_848-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_860-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_861-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_882-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_890-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 44:6', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_932-05_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_974-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_1063-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_1084-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_601-15_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_631-15_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_631-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_655-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_657-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_663-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_667-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_693-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_722-15_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_723-45_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_742-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 36:2', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_878-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PE 46:4', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_909-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PI 40:6', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_922-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_934-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_601-55_NEG.mzML': {'Tissus': '', 'Annotations': 'DG O-36:4', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_631-25_NEG.mzML': {'Tissus': '', 'Annotations': 'PA 32:8', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_661-55_NEG.mzML': {'Tissus': '', 'Annotations': 'DG 40:9', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_729-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_729-25_NEG2.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_746-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_751-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PA 40:4', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_753-65_NEG.mzML': {'Tissus': '', 'Annotations': 'DG 46:5', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_760-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 34:1', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_779-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PA 42:4', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_790-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS 36:0 (PS 18:0_18:0)', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_812-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 38:3', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_814-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 38:2', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_821-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PI O-34:1', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_826-55_NEG.mzML': {'Tissus': '', 'Annotations': 'PC 40:9', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_844-75_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_846-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_883-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PI 38:5', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_901-65_NEG.mzML': {'Tissus': '', 'Annotations': 'PI P-40:2', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_920-75_NEG.mzML': {'Tissus': '', 'Annotations': 'PS 46:5', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_932-75_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_303-25_NEG.mzML': {'Tissus': 'Partout', 'Annotations': 'FA 20:4', 'Sous-type': ''},
    '20211026_OG_MSMS_642-55_NEG.mzML': {'Tissus': 'Partout', 'Annotations': 'HexCer 30:1;2', 'Sous-type': ''},
    '20211026_OG_MSMS_687-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA O-36:1 (PA O-18:1_18:0)', 'Sous-type': 'PCC'},
    '20211026_OG_MSMS_699-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA 36:2 ', 'Sous-type': 'ADK'},
    '20211026_OG_MSMS_701-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA 36:1 (PA 18:1_18:0)', 'Sous-type': ''},
    '20211026_OG_MSMS_742-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 36:2', 'Sous-type': ''},
    '20211026_OG_MSMS_748-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PS O-34:0', 'Sous-type': ''},
    '20211026_OG_MSMS_750-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-38:5 (PE O-18:0_20:4) ', 'Sous-type': 'PCC'},
    '20211026_OG_MSMS_766-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 38:4 (PE 18:0_20:4)', 'Sous-type': 'PCC'},
    '20211026_OG_MSMS_771-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_853-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_863-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_906-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230316_OG_MSMS_1010-65_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'Hex2Cer 42:0 (adduit chlore)', 'Sous-type': ''},
    '20230316_OG_MSMS_635-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230316_OG_MSMS_659-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA O-34:1', 'Sous-type': ''},
    '20230316_OG_MSMS_679-45_NEG.mzML': {'Tissus': 'Partout', 'Annotations': 'DG 38:4 CL', 'Sous-type': ''},
    '20230316_OG_MSMS_682-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230316_OG_MSMS_698-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'Cer (d18:2_24:0(2OH)) CL', 'Sous-type': ''},
    '20230316_OG_MSMS_700-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'Cer (d18:1_24:0(2OH)) CL', 'Sous-type': ''},
    '20230316_OG_MSMS_701-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA 36:1 (PA 18:1_18:0)', 'Sous-type': ''},
    '20230316_OG_MSMS_713-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA O-38:2', 'Sous-type': 'ADK'},
    '20230316_OG_MSMS_718-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 34:0 (PE 18:0_16:0)', 'Sous-type': ''},
    '20230316_OG_MSMS_720-45_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 32:2 CL', 'Sous-type': ''},
    '20230316_OG_MSMS_728-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-36:2 (PE P-18:1_18:0) ', 'Sous-type': 'ADK'},
    '20230316_OG_MSMS_730-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230316_OG_MSMS_733-45_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PA 36:3 CL', 'Sous-type': ''},
    '20230316_OG_MSMS_738-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 36:4 (PE 16:0_20:4)', 'Sous-type': 'ADK'},
    '20230316_OG_MSMS_742-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 36:2', 'Sous-type': ''},
    '20230316_OG_MSMS_744-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 36:1 (PE 18:0_18:1)', 'Sous-type': ''},
    '20230316_OG_MSMS_748-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PS O-34:0', 'Sous-type': ''},
    '20230316_OG_MSMS_766-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 38:4 (PE 18:0_20:4)', 'Sous-type': ''},
    '20230316_OG_MSMS_768-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 38:3', 'Sous-type': ''},
    '20230316_OG_MSMS_770-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 38:2', 'Sous-type': ''},
    '20230316_OG_MSMS_788-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS 36:1 (PS 18:0_18:1)', 'Sous-type': ''},
    '20230316_OG_MSMS_790-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS 36:0 (PS 18:0_18:0)', 'Sous-type': ''},
    '20230316_OG_MSMS_792-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 40:5', 'Sous-type': ''},
    '20230316_OG_MSMS_794-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 40:4 ', 'Sous-type': 'ADK'},
    '20230316_OG_MSMS_799-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230316_OG_MSMS_802-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230316_OG_MSMS_804-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-42:6', 'Sous-type': ''},
    '20230316_OG_MSMS_822-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 36:1 CL', 'Sous-type': ''},
    '20230316_OG_MSMS_835-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PI 34:1 (PI 16:0_18:1)', 'Sous-type': ''},
    '20230316_OG_MSMS_844-65_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 38:4 CL', 'Sous-type': ''},
    '20230316_OG_MSMS_846-65_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'HexCer(d18:1/24:0) CL', 'Sous-type': ''},
    '20230316_OG_MSMS_878-65_NEG.mzML': {'Tissus': 'Partout', 'Annotations': 'PS 40:2 CL', 'Sous-type': ''},
    '20230316_OG_MSMS_930-65_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PC 44:4', 'Sous-type': ''},
    '20230316_OG_MSMS_996-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_709-25_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_1006-65_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'Hex2Cer 38:2', 'Sous-type': ''},
    '20230707_OG_MSMS_722-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-36:5 (PE P-16:0_20:4)', 'Sous-type': ''},
    '20230707_OG_MSMS_726-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-36:3 (PE P-18:0_18:2)', 'Sous-type': 'ADK'},
    '20230707_OG_MSMS_734-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 36:6 (PE 16:2_20:4)', 'Sous-type': ''},
    '20230707_OG_MSMS_740-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_772-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 38:1 (PE 18:1_20:0)', 'Sous-type': ''},
    '20230707_OG_MSMS_778-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS 36:6', 'Sous-type': ''},
    '20230707_OG_MSMS_797-65_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'TG 48:4', 'Sous-type': 'PCC'},
    '20230707_OG_MSMS_820-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 42:5', 'Sous-type': ''},
    '20230707_OG_MSMS_836-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_869-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PI 37:5', 'Sous-type': 'PCC'},
    '20230707_OG_MSMS_897-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230711_OG_MSMS_626-55_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230711_OG_MSMS_639-25_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PG 26:7', 'Sous-type': 'ADK'},
    '20230711_OG_MSMS_671-45_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA 34:2', 'Sous-type': 'ADK'},
    '20230711_OG_MSMS_678-45_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 32:6', 'Sous-type': 'PCC'},
    '20230711_OG_MSMS_684-65_NEG.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230711_OG_MSMS_728-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-36:2 (PE P-18:1_18:0) ', 'Sous-type': 'ADK'},
    '20230711_OG_MSMS_762-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 38:6', 'Sous-type': 'ADK'},
    '20230711_OG_MSMS_764-55_NEG.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 38:5 (PE 18:1_20:4)', 'Sous-type': ''},
    '20230711_OG_MSMS_798-65_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 40:2 (PE 18:1_22:1)', 'Sous-type': 'PCC'},
    '20230711_OG_MSMS_804-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-42:6', 'Sous-type': ''},
    '20230711_OG_MSMS_864-65_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS 42:5', 'Sous-type': 'ADK'},
    '20230726_OG_MSMS_1006-65_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'Hex2Cer 38:2', 'Sous-type': ''},
    '20230726_OG_MSMS_714-55_NEG.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 34:2 (PE 16:0_18:2)', 'Sous-type': 'ADK'},
}

def load_data(directories, output_path):
    all_data_list = []
    all_mz_values = set()  # Création d'un ensemble pour stocker tous les m/z

    for directory in directories:
        file_list = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".mzML")]

        for mzml_file in file_list:
            exp = pyopenms.MSExperiment()
            pyopenms.MzMLFile().load(mzml_file, exp)

            chromatogram = exp.getChromatograms()[0]
            chrom_times, chrom_intensities = chromatogram.get_peaks()

            # Trouver les pics dans le chromatogramme
            peaks, _ = find_peaks(chrom_intensities, height=1)
            max_peak_index = np.argmax(chrom_intensities[peaks])
            max_peak_time = chrom_times[peaks[max_peak_index]]
            max_peak_intensity = chrom_intensities[peaks[max_peak_index]]

            # Affichage du chromatogramme avec le pic le plus intense en rouge
            plt.figure(figsize=(10, 5))
            plt.plot(chrom_times, chrom_intensities, label='Chromatogramme')
            plt.plot(max_peak_time, max_peak_intensity, 'ro', label='Pic le plus intense')
            plt.title(f'Chromatogramme (pic le plus intense à {max_peak_time:.2f} secondes)')
            plt.xlabel('Temps (s)')
            plt.ylabel('Intensité')
            plt.legend()
            plt.show()

            max_peak_spectrum = exp.getSpectra()[peaks[max_peak_index]]

            # Affichage du spectre du pic le plus intense avec le temps de rétention dans le titre
            mz_values, mass_intensities = max_peak_spectrum.get_peaks()
            plt.figure(figsize=(10, 5))
            plt.plot(mz_values, mass_intensities, label='Spectre de masse')
            plt.title(f'Spectre de masse (pic le plus intense à {max_peak_time:.2f} secondes)')
            plt.xlabel('m/z')
            plt.ylabel('Intensité')
            plt.legend()
            plt.show()

            all_mz_values.update(mz_values)  # Ajout des m/z de ce spectre à l'ensemble global

            # Extraire la date du nom de fichier
            date_str = os.path.basename(mzml_file).split('.')[0].split('_')[0]
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:]
            date = f"{day}/{month}/{year}"

            # Extraire l'm/z du nom de fichier et remplacer les tirets par des virgules
            mz_value = "_".join(os.path.basename(mzml_file).split('_')[3:-1]).replace("-", ",") if len(os.path.basename(mzml_file).split('_')) > 3 else os.path.basename(mzml_file).split('_')[-1]

            # Déterminer la polarité
            if "NEG" in os.path.basename(mzml_file):
                polarity = "Négatif"
            elif "POS" in os.path.basename(mzml_file):
                polarity = "Positif"
            else:
                polarity = ""

            # Déterminer le type MS
            type_ms = "MS2"

            # Déterminer le type
            if "OG" in os.path.basename(mzml_file):
                type = "Oeso gastrique"
            elif "GLIOMIC" in os.path.basename(mzml_file):
                type = "Gliome"
            else:
                type = ""

            # Extraire les annotations, sous-type, et tissus du dictionnaire de référence
            file_name = os.path.basename(mzml_file)
            reference_info = reference_dict.get(file_name, {"Tissus": "", "Annotations": "", "Sous-type": ""})
            tissus = reference_info.get("Tissus", "")
            annotations = reference_info.get("Annotations", "")
            sous_type = reference_info.get("Sous-type", "")

            intensity_dict = dict(zip(mz_values, mass_intensities))

            # Création d'une ligne de données
            row_data = {mz: intensity_dict.get(mz, 0) for mz in all_mz_values}
            row_data['File'] = os.path.basename(mzml_file)
            row_data['Date'] = date
            row_data['m/z'] = mz_value
            row_data['Polarité'] = polarity
            row_data['Type MS'] = type_ms
            row_data['Type'] = type
            row_data['Tissus'] = tissus
            row_data['Sous-type'] = sous_type
            row_data['Annotations'] = annotations
            row_data['Sum'] = sum(mass_intensities)
            all_data_list.append(pd.DataFrame(row_data, index=[0],))

    all_data = pd.concat(all_data_list, ignore_index=True)

    # Définir l'ordre des colonnes pour le DataFrame
    desired_columns = ['File', 'Date', 'm/z', 'Polarité', 'Type MS', 'Type', 'Tissus', 'Sous-type', 'Annotations', 'Sum'] + sorted(all_mz_values)
    all_data = all_data.reindex(columns=desired_columns)

    # Remplacer les valeurs NaN par 0
    all_data = all_data.fillna(0)

    # Enregistrer le DataFrame au format Parquet
    output_file = os.path.join(output_path, "Test_PCA.parquet")
    all_data.to_parquet(output_file, index=False)
    print(f"CSV file saved to: {output_file}")
    return all_data

# Chemin vers le bureau
output_path = r"C:\faycal2024\Faycal\mzml\OG"
directories = [r'C:\faycal2024\Faycal\mzml\OG\testmz']
all_data = load_data(directories, output_path)

def test_zeros(row):
    return all(value == 0 for value in row.values[10:])

# Utilisation de la fonction pour tester si une ligne ne contient que des zéros à partir de la colonne 11
for index, row in all_data.iterrows():
    if test_zeros(row):
        print("True")
    else:
        print("False")

# Sélectionner les noms de colonnes à partir de la 11ème colonne jusqu'à la dernière colonne
columns = all_data.columns[10:]

# Calculer les valeurs minimales et maximales pour ces noms de colonnes
min_column = min(columns)
max_column = max(columns)

# Afficher les résultats
print("Nom de colonne minimal :", min_column)
print("Nom de colonne maximal :", max_column)

# Retourner all_data
print(all_data)