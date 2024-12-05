import pyopenms
import os
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt

# Reference dictionary for annotations, sub-types, and tissues
reference_dict = {
    '20221222_GLIOMIC_MSMS_1028-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_601-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_627-55_POS.mzML': {'Tissus': '', 'Annotations': 'DG 36:6', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_651-55_POS.mzML': {'Tissus': '', 'Annotations': 'DG 38:1', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_680-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_691-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_728-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_736-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_750-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_756-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_763-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_768-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_780-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_782-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_790-55_POS.mzML': {'Tissus': '', 'Annotations': 'PS 36:1', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_804-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_810-65_POS.mzML': {'Tissus': '', 'Annotations': 'PC 38:4', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_812-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_826-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_828-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_832-65_POS.mzML': {'Tissus': '', 'Annotations': 'PS O-40:1', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_836-55_POS.mzML': {'Tissus': '', 'Annotations': 'PS 40:6', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_840-65_POS.mzML': {'Tissus': '', 'Annotations': 'PC 40:3', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_862-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_864-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_890-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20221222_GLIOMIC_MSMS_974-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_649-15_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_683-25_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_686-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_744-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_792-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_796-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_837-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_848-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_876-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_881-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_891-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230210_GLIOMIC_MSMS_931-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_1076-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_600-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_620-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_632-15_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_644-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_649-15_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_651-55_POS.mzML': {'Tissus': '', 'Annotations': 'DG 38:1', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_669-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_718-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_757-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_771-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_778-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_793-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_796-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_822-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_842-65_POS.mzML': {'Tissus': '', 'Annotations': 'PC 40:2', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_854-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_866-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_902-95_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_904-85_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_907-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230228_GLIOMIC_MSMS_923-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_600-25_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_612-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_619-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_627-55_POS.mzML': {'Tissus': '', 'Annotations': 'DG 36:6', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_629-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_632-15_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_644-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_655-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_661-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_701-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_713-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_718-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_730-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_794-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_806-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_850-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_878-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_904-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_912-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_918-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_958-85_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230303_GLIOMIC_MSMS_960-85_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_605-55_POS.mzML': {'Tissus': '', 'Annotations': 'DG 36:1', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_627-55_POS.mzML': {'Tissus': '', 'Annotations': 'DG 36:6', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_631-25_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_685-55_POS.mzML': {'Tissus': '', 'Annotations': 'PA O-36:3', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_749-55_POS.mzML': {'Tissus': '', 'Annotations': 'PA 40:6', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_758-65_POS.mzML': {'Tissus': '', 'Annotations': 'PC 34:2', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_760-65_POS.mzML': {'Tissus': '', 'Annotations': 'PS 34:2', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_764-55_POS.mzML': {'Tissus': '', 'Annotations': 'PE 38:6', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_766-55_POS.mzML': {'Tissus': '', 'Annotations': 'PE 38:5', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_782-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_788-65_POS.mzML': {'Tissus': '', 'Annotations': 'PE P-40:0', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_808-65_POS.mzML': {'Tissus': '', 'Annotations': 'PC 38:5', 'Sous-type': ''},
    '20230428_GLIOMIC_MSMS_907-65_POS.mzML': {'Tissus': '', 'Annotations': 'PI 0-40:1', 'Sous-type': ''},
    '20211026_OG_MSMS_720-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS P-32:0', 'Sous-type': ''},
    '20211026_OG_MSMS_754-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_758-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_758-55_POS2.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_796-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_810-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20211026_OG_MSMS_820-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_1081-85_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_1097-85_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_601-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_603-55_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'DG O-36:4', 'Sous-type': ''},
    '20230317_OG_MSMS_627-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_651-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_663-45_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PG 28:2', 'Sous-type': ''},
    '20230317_OG_MSMS_682-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_689-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_699-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'DG 42:5', 'Sous-type': 'ADK'},
    '20230317_OG_MSMS_700-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_702-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-34:2', 'Sous-type': 'ADK'},
    '20230317_OG_MSMS_706-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 34:7', 'Sous-type': 'PCC'},
    '20230317_OG_MSMS_709-25_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_716-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 34:2 (PE 18:1_16:1)', 'Sous-type': ''},
    '20230317_OG_MSMS_720-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS P-32:0', 'Sous-type': ''},
    '20230317_OG_MSMS_724-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 32:5', 'Sous-type': 'PCC'},
    '20230317_OG_MSMS_728-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-36:3', 'Sous-type': 'ADK'},
    '20230317_OG_MSMS_732-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS 32:2', 'Sous-type': ''},
    '20230317_OG_MSMS_734-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 36:7', 'Sous-type': ''},
    '20230317_OG_MSMS_738-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_740-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 36:4', 'Sous-type': ''},
    '20230317_OG_MSMS_742-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 36:3', 'Sous-type': ''},
    '20230317_OG_MSMS_744-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 36:2', 'Sous-type': ''},
    '20230317_OG_MSMS_746-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_762-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_768-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_772-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_774-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS O-36:2', 'Sous-type': 'ADK'},
    '20230317_OG_MSMS_782-55_POS.mzML': {'Tissus': '', 'Annotations': 'PC 36:4', 'Sous-type': ''},
    '20230317_OG_MSMS_794-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 40:5', 'Sous-type': 'ADK'},
    '20230317_OG_MSMS_805-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA 44:6', 'Sous-type': 'PCC'},
    '20230317_OG_MSMS_806-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 38:6', 'Sous-type': 'PCC'},
    '20230317_OG_MSMS_828-65_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'PE 42:2', 'Sous-type': ''},
    '20230317_OG_MSMS_850-75_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'PC P-42:2', 'Sous-type': ''},
    '20230317_OG_MSMS_855-75_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'PI 36:6', 'Sous-type': ''},
    '20230317_OG_MSMS_874-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_875-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_876-75_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'PS 42:0', 'Sous-type': ''},
    '20230317_OG_MSMS_881-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_881-75_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'TG 52:2', 'Sous-type': ''},
    '20230317_OG_MSMS_900-75_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'PS 44:10', 'Sous-type': ''},
    '20230317_OG_MSMS_902-75_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'PC 44:0', 'Sous-type': ''},
    '20230317_OG_MSMS_907-75_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_920-75_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_941-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PI 42:5', 'Sous-type': 'PCC'},
    '20230317_OG_MSMS_990-85_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230317_OG_MSMS_999-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_605-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'DG O-36:3', 'Sous-type': 'ADK'},
    '20230707_OG_MSMS_639-35_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_650-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'Cer 42:1', 'Sous-type': 'ADK'},
    '20230707_OG_MSMS_685-45_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA O-36:3', 'Sous-type': ''},
    '20230707_OG_MSMS_702-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-34:2', 'Sous-type': 'ADK'},
    '20230707_OG_MSMS_710-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'Cer 44:1, O4', 'Sous-type': ''},
    '20230707_OG_MSMS_712-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'Cer 44:0, O4 (t20:0_24:0 (2OH))', 'Sous-type': ''},
    '20230707_OG_MSMS_723-45_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PA 38:5', 'Sous-type': 'PCC'},
    '20230707_OG_MSMS_736-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_739-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_750-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PE O-38:6', 'Sous-type': ''},
    '20230707_OG_MSMS_756-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_758-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PC 34:2', 'Sous-type': ''},
    '20230707_OG_MSMS_766-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PC O-36:5', 'Sous-type': ''},
    '20230707_OG_MSMS_779-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_788-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 36:1', 'Sous-type': ''},
    '20230707_OG_MSMS_792-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE 40:6', 'Sous-type': 'ADK'},
    '20230707_OG_MSMS_796-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_808-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 38:5', 'Sous-type': ''},
    '20230707_OG_MSMS_812-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PC 38:3', 'Sous-type': ''},
    '20230707_OG_MSMS_814-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PE 42:9', 'Sous-type': ''},
    '20230707_OG_MSMS_828-65_POS.mzML': {'Tissus': 'PARTOUT', 'Annotations': 'PE 42:2', 'Sous-type': ''},
    '20230707_OG_MSMS_855-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PI 36:6', 'Sous-type': ''},
    '20230707_OG_MSMS_881-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_941-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PI 42:5', 'Sous-type': 'PCC'},
    '20230707_OG_MSMS_965-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_983-65_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230707_OG_MSMS_997-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PI 46:5', 'Sous-type': 'PCC'},
    '20230711_OG_MSMS_630-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': '', 'Sous-type': 'PCC'},
    '20230711_OG_MSMS_703-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'SM (d16:1_18:0)', 'Sous-type': ''},
    '20230711_OG_MSMS_778-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230711_OG_MSMS_780-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 36:5 (PC 22:4_14:1)', 'Sous-type': 'PCC'},
    '20230711_OG_MSMS_784-55_POS.mzML': {'Tissus': 'Sain', 'Annotations': 'PC 36:3', 'Sous-type': ''},
    '20230726_OG_MSMS_666-45_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS O-38:0', 'Sous-type': 'PCC'},
    '20230726_OG_MSMS_690-55_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230726_OG_MSMS_730-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-36:2', 'Sous-type': 'ADK'},
    '20230726_OG_MSMS_739-45_POS.mzML': {'Tissus': '', 'Annotations': '', 'Sous-type': ''},
    '20230726_OG_MSMS_774-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PS O-36:2', 'Sous-type': 'ADK'},
    '20230726_OG_MSMS_780-55_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PC 36:5 (PC 22:4_14:1)', 'Sous-type': 'PCC'},
    '20230726_OG_MSMS_810-65_POS.mzML': {'Tissus': 'Cancer', 'Annotations': 'PE O-42:4', 'Sous-type': 'ADK'},
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
    output_file = os.path.join(output_path, "data_ref_pos.parquet")
    all_data.to_parquet(output_file, index=False)
    print(f"Parquet file saved to: {output_file}")
    return all_data

# Chemin vers le bureau
output_path = r"C:\faycal2024\Faycal\mzml\OG"
directories = [r'C:\faycal2024\Faycal\mzml\gliomic\POS',r'C:\faycal2024\Faycal\mzml\OG\POS']
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