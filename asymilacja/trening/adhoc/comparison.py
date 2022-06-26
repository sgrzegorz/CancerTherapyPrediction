import matplotlib.pyplot as plt
import pandas as pd

from asymilacja.model.utils_linsp import GS3

from data.klusek.EP1.config import threatment_start, threatment_end,threatment2_start

from asymilacja.paramteres.Vis7LinspKlusekShortLinearEachIter import plot_parameters

threatment_time = threatment_end - threatment_start
steps_backward = threatment_start
steps_forward = threatment2_start - threatment_start

df = pd.read_csv("data/klusek/EP1/stats0.csv")
df_true = df[(df['iteration'] >= 0) & (df['iteration']<=threatment2_start)]
P_true = list(df_true.prolif_cells)
t_true = list(df_true.iteration)
t_real = list(df_true.t)

l26 = '2/6 przedziału terapii'
l46 = '4/6 przedziału terapii'
l66 = 'pełen przedział terapii'
maximal_label = 'przedział terapii + okres po terapii'
przedzialu_label = [l26,l46,l66]
USE_REAL_TIME = True

EP1_GS3_adhoc ={'P0': 142730, 'C0': 9.997599278274212, 'gamma_p': 9.69571151252561e-05, 'K': 499999.9999991985, 'T_death': 33916095.87874119, 'eta': 0.19999610420891356, 'KDE': 0.0006645158511887903, 'lambda_p': 6.452133696263065e-05}
EP1_GS3_tuned ={'P0': 142730, 'C0': 9.9972278874274, 'gamma_p': 9.695720018477277e-05, 'K': 499999.99999999994, 'T_death': 49999999.999999925, 'eta': 0.19999354689824905, 'KDE': 0.0006645158511887903, 'lambda_p': 6.450749951157682e-05}


plot_parameters(GS3,EP1_GS3_adhoc,steps_forward,steps_backward,threatment_start,None,USE_REAL_TIME,t_real)
plot_parameters(GS3,EP1_GS3_tuned,steps_forward,steps_backward,threatment_start,None,USE_REAL_TIME,t_real)

plt.show()