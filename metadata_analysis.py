'''
1. Stage 1 and Stage 2 Training and Test Data
            Train       Test
Stage 1     10712       1377
Stage 2     12089       3205

2. Stage 1 and Stage 2 Pneumothorax Distribution
            Pneumothorax    Non-Pneumothorax
Stage 1     2379(22.28%)    8296(77.71%)
Stage 2     2669(22.15%)    9378(77.84%)

3. Stage 1 and Stage 2 Gender Distribution
            Male            Female
Stage 1     5880(55.081%)   4795(44.918%)
Stage 2     6626(55.001)    5421(44.998%)

4. Stage 1 and Stage 2 Age Distribution 
Infancy (0-1), Childhood (1-11), Early Adolescence (12-18),Young Adults (19-39), Older Adults (40-65), Elderly (>65)
            Infancy         Childhood      Early Adolescence      Young Adults        Older Adults        Elderly
Stage 1     1(0.009%)       183(1.71%)     423(3.96%)             2777(26.01%)        5920(55.45%)        1371(12.84%)
Stage 2     1(0.008%)       216(1.79%)     484(4.01%)             3136(26.03%)        6678(55.43%)        1532(12.71%)

5. Stage 1 and Stage 2 View Position Distribution
            AP           PA          AP/PA Ratio
Stage 1     4181         6494        0.6438
Stage 2     4773         7274        0.6562
'''

import pandas as pd

# File Paths
stage1_metadata_path = "./dicom-metadata-stage_1.csv"
stage2_metadata_path = "./dicom-metadata-stage_2.csv"

# To get the Stage 1 and Stage 2 Pneumothorax Distribution
def get_pneumothorax_distribution(stage1_metadata_path, stage2_metadata_path):
    stage1_metadata = pd.read_csv(stage1_metadata_path)
    stage2_metadata = pd.read_csv(stage2_metadata_path)

    stage1_pneumothorax = stage1_metadata[stage1_metadata["Pneumothorax"]==1]
    stage1_non_pneumothorax = stage1_metadata[stage1_metadata["Pneumothorax"]==0]
    stage2_pneumothorax = stage2_metadata[stage2_metadata["Pneumothorax"]==1]
    stage2_non_pneumothorax = stage2_metadata[stage2_metadata["Pneumothorax"]==0]

    return stage1_pneumothorax, stage1_non_pneumothorax, stage2_pneumothorax, stage2_non_pneumothorax

stage1_P, stage1_NP, stage2_P, stage2_NP = get_pneumothorax_distribution(stage1_metadata_path, stage2_metadata_path)
print("Stage 1 Pneumothorax Count: ", len(stage1_P))
print("Stage 1 Non-Pneumothorax Count: ", len(stage1_NP))
print("Stage 2 Pneumothorax Count: ", len(stage2_P))
print("Stage 2 Non-Pneumothorax Count: ", len(stage2_NP))

# To get the Stage 1 and Stage 2 Gender Distribution
def gender_distribution(stage1_metadata_path, stage2_metadata_path):
    stage1_metadata = pd.read_csv(stage1_metadata_path)
    stage2_metadata = pd.read_csv(stage2_metadata_path)

    stage1_male = stage1_metadata[stage1_metadata["Sex"]=="M"]
    stage1_female = stage1_metadata[stage1_metadata["Sex"]=="F"]
    stage2_male = stage2_metadata[stage2_metadata["Sex"]=="M"]
    stage2_female = stage2_metadata[stage2_metadata["Sex"]=="F"]

    return stage1_male, stage1_female, stage2_male, stage2_female

stage1_M, stage1_F, stage2_M, stage2_F = gender_distribution(stage1_metadata_path, stage2_metadata_path)
print("Stage 1 Male Count: ", len(stage1_M))
print("Stage 1 Female Count: ", len(stage1_F))
print("Stage 2 Male Count: ", len(stage2_M))
print("Stage 2 Female Count: ", len(stage2_F))

# To get the Stage 1 and Stage 2 Age Distribution
def age_distribution(stage1_metadata_path, stage2_metadata_path):
    stage1_metadata = pd.read_csv(stage1_metadata_path)
    stage2_metadata = pd.read_csv(stage2_metadata_path)

    stage1_infancy = stage1_metadata[stage1_metadata["Age"]<=1]
    stage1_childhood = stage1_metadata[(stage1_metadata["Age"]>1) & (stage1_metadata["Age"]<=11)]
    stage1_early_adolescence = stage1_metadata[(stage1_metadata["Age"]>11) & (stage1_metadata["Age"]<=18)]
    stage1_young_adults = stage1_metadata[(stage1_metadata["Age"]>18) & (stage1_metadata["Age"]<=39)]
    stage1_older_adults = stage1_metadata[(stage1_metadata["Age"]>39) & (stage1_metadata["Age"]<=65)]
    stage1_elderly = stage1_metadata[stage1_metadata["Age"]>65]

    stage2_infancy = stage2_metadata[stage2_metadata["Age"]<=1]
    stage2_childhood = stage2_metadata[(stage2_metadata["Age"]>1) & (stage2_metadata["Age"]<=11)]
    stage2_early_adolescence = stage2_metadata[(stage2_metadata["Age"]>11) & (stage2_metadata["Age"]<=18)]
    stage2_young_adults = stage2_metadata[(stage2_metadata["Age"]>18) & (stage2_metadata["Age"]<=39)]
    stage2_older_adults = stage2_metadata[(stage2_metadata["Age"]>39) & (stage2_metadata["Age"]<=65)]
    stage2_elderly = stage2_metadata[stage2_metadata["Age"]>65]

    return stage1_infancy, stage1_childhood, stage1_early_adolescence, stage1_young_adults, stage1_older_adults, stage1_elderly, stage2_infancy, stage2_childhood, stage2_early_adolescence, stage2_young_adults, stage2_older_adults, stage2_elderly

stage1_I, stage1_C, stage1_EA, stage1_YA, stage1_OA, stage1_E, stage2_I, stage2_C, stage2_EA, stage2_YA, stage2_OA, stage2_E = age_distribution(stage1_metadata_path, stage2_metadata_path)
# Print Count and Percentage in that Stage
total_stage1 = len(stage1_I) + len(stage1_C) + len(stage1_EA) + len(stage1_YA) + len(stage1_OA) + len(stage1_E)
total_stage2 = len(stage2_I) + len(stage2_C) + len(stage2_EA) + len(stage2_YA) + len(stage2_OA) + len(stage2_E)

print("Stage 1 Infancy Count: ", len(stage1_I), "Percentage: ", len(stage1_I)/total_stage1*100)
print("Stage 1 Childhood Count: ", len(stage1_C), "Percentage: ", len(stage1_C)/total_stage1*100)
print("Stage 1 Early Adolescence Count: ", len(stage1_EA), "Percentage: ", len(stage1_EA)/total_stage1*100)
print("Stage 1 Young Adults Count: ", len(stage1_YA), "Percentage: ", len(stage1_YA)/total_stage1*100)
print("Stage 1 Older Adults Count: ", len(stage1_OA), "Percentage: ", len(stage1_OA)/total_stage1*100)
print("Stage 1 Elderly Count: ", len(stage1_E), "Percentage: ", len(stage1_E)/total_stage1*100)

print("Stage 2 Infancy Count: ", len(stage2_I), "Percentage: ", len(stage2_I)/total_stage2*100)
print("Stage 2 Childhood Count: ", len(stage2_C), "Percentage: ", len(stage2_C)/total_stage2*100)
print("Stage 2 Early Adolescence Count: ", len(stage2_EA), "Percentage: ", len(stage2_EA)/total_stage2*100)
print("Stage 2 Young Adults Count: ", len(stage2_YA), "Percentage: ", len(stage2_YA)/total_stage2*100)
print("Stage 2 Older Adults Count: ", len(stage2_OA), "Percentage: ", len(stage2_OA)/total_stage2*100)
print("Stage 2 Elderly Count: ", len(stage2_E), "Percentage: ", len(stage2_E)/total_stage2*100)

# To get the Stage 1 and Stage 2 View Position Distribution
def view_position_distribution(stage1_metadata_path, stage2_metadata_path):
    stage1_metadata = pd.read_csv(stage1_metadata_path)
    stage2_metadata = pd.read_csv(stage2_metadata_path)

    stage1_AP = stage1_metadata[stage1_metadata["View Position"]=="AP"]
    stage1_PA = stage1_metadata[stage1_metadata["View Position"]=="PA"]
    stage2_AP = stage2_metadata[stage2_metadata["View Position"]=="AP"]
    stage2_PA = stage2_metadata[stage2_metadata["View Position"]=="PA"]

    return stage1_AP, stage1_PA, stage2_AP, stage2_PA

stage1_AP, stage1_PA, stage2_AP, stage2_PA = view_position_distribution(stage1_metadata_path, stage2_metadata_path)
print("Stage 1 AP Count: ", len(stage1_AP))
print("Stage 1 PA Count: ", len(stage1_PA))
# AP/PA Ratio
print("Stage 1 AP/PA Ratio: ", len(stage1_AP)/len(stage1_PA))

print("Stage 2 AP Count: ", len(stage2_AP))
print("Stage 2 PA Count: ", len(stage2_PA))
# AP/PA Ratio
print("Stage 2 AP/PA Ratio: ", len(stage2_AP)/len(stage2_PA))
