import utils.option_a_step2_only
from utils.option_a_step2_only import classifier_a_step2 as classifier_a


#print(utils.option_a_step2_only.confus_matrix)
#print(utils.option_a_step2_only.report)
print(classifier_a(inputfile='assets/anomalous_dice/img_17584_cropped.jpg'))
print(classifier_a(inputfile='assets/anomalous_dice/img_17829_cropped.jpg'))
