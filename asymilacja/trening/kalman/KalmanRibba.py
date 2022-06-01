import numpy as np
from matplotlib import pyplot as plt
from numpy import array
import pandas as pd


import numpy, scipy, adao
print("Numpy:", numpy.__version__)
print("Scipy:", scipy.__version__)
print("Adao:", adao.version)
numpy.set_printoptions(precision=2,linewidth=5000,threshold=10000)



def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))

# groundTruthObservation = [53.24271645564879, 55.73408614284355, 57.813057709561186, 59.62972666793778,
#                           61.24840184295388,
#                           63.859216213539206, 65.7997799043894, 67.77926410429723, 69.28402751342766, 70.81265484922778,
#                           72.10088901384385, 71.29892580495805, 67.95073464005661, 61.331540140946, 57.63157970225158,
#                           50.12167244169479, 45.40360491222454, 41.49356685035813, 37.93253743983854, 36.1688656152416,
#                           35.02068670905743, 33.67645650462805, 32.581933361277734, 31.72173449803619,
#                           30.876810588310185,
#                           29.843469649983355, 28.78468810561838, 27.558211558432504, 26.884196350458918,
#                           26.12727674566478,
#                           25.65886011913788, 25.285974453184487, 24.9173266951422, 24.374992914197623,
#                           24.147445381985015,
#                           23.833506139271858, 23.73825937595852, 23.643266710167445, 23.63563916344681,
#                           23.628013239989844,
#                           23.619802542997036, 23.7863419394951, 23.778094619372666, 24.03227660799511,
#                           24.465129968143838,
#                           24.992959690942634, 25.437238636013635, 26.070659685386996, 26.715782904311283,
#                           27.370167860875657,
#                           28.732613710650547, 29.734825400376987, 31.597039003311526, 33.098010640853595,
#                           35.32361261509472,
#                           37.41004168571596, 39.57705260790178, 41.06724823357617, 43.37195516176883, 45.76131544560512,
#                           48.517302035576286]

df_true = pd.read_csv("data/klusek/EP1/stats0.csv")
from data.klusek.EP1.config import threatment_start, threatment_end, threatment2_start

threatment_time = threatment_end - threatment_start
df_true = df_true[(df_true['iteration'] >= threatment_start) & (df_true['iteration'] <= threatment2_start)]

P_true = list(df_true.prolif_cells)
N_true = list(df_true.dead_cells)
t_true = list(df_true.iteration)

df = df_true
df = df.iloc[::100, :]
# df = df_true[df_true.index < (6/6*threatment_time)+threatment_start]
if df.empty:
    raise ValueError("No data provided!")
P = list(df.prolif_cells)
N = list(df.dead_cells)
t = list(df.iteration)


noise =np.abs(np.random.normal(0,1,len(P))*1e3)
P = P+ noise
groundTruthObservation = P


from adao import adaoBuilder
Observations    = numpy.ravel(groundTruthObservation)
NbObs           = len(Observations)
TimeBetween2Obs = 1
TimeWindows     = NbObs * TimeBetween2Obs
NbOfEulerStep   = 5 # Number of substeps between 2 obs

initState = [142730,5.380861282944324]

base_lambda_p = 0.00010000000001493816
base_K = 295647.30000000016
base_gamma_p = 0.00019718594472837436
base_eta = 0.2999999999661837
base_KDE =  0.0004970985980341276



idealX = array([base_lambda_p, base_K, base_gamma_p, base_eta, base_KDE])


def rhs(z, t, lambda_p, K, gamma_p, eta, KDE):
    P, C = z

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]




def PlotOneSerie(t,__Observations, __Title=''):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10, 4)
    #
    plt.figure()
    plt.plot(t,__Observations,'k-')
    plt.title(__Title, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('Value')


Yobs = groundTruthObservation

TimeBetween2Obs = 1
NbOfEulerStep   = 5 # Number of substeps between 2 obs


def H( Z ):
    "Observation operator"
    # Z <=> [state, parameters]
    #
    __Z = numpy.ravel(Z)
    # Pstar = __Z[1]+__Z[2]+__Z[3]
    #
    return numpy.array(__Z[0])

def F( Z ):
    "Evolution operator"
    # Z = [state, parameters]
    #
    __Z = numpy.ravel(Z)
    #
    lambda_p, K, gamma_p, eta, KDE = __Z[2:]
    #
    dt = 0.1 * TimeBetween2Obs
    # Initial state
    z = numpy.ravel(__Z[0:2])
    state = None
    # state.append(z)
    for step in range(NbOfEulerStep):
        dt = TimeBetween2Obs * step/NbOfEulerStep
        # Use rhs to calculate dz/dt
        dzdt = numpy.ravel(rhs(z,None,lambda_p, K, gamma_p, eta, KDE))
        # dt/dt = (z(t+1) - z(t)) / dt ===> z(t+1) = z(t) + (dz/dt) * dt
        z = z + numpy.ravel(dzdt) * dt
        state = z
    #
    __Znpu = state.tolist()+__Z[2:].tolist()
    #
    return __Znpu

case = adaoBuilder.New('')
#
case.setObservationOperator(OneFunction        = H)
case.setObservationError   (ScalarSparseMatrix = 1.)
#
case.setEvolutionModel     (OneFunction        = F)
case.setEvolutionError     (ScalarSparseMatrix = 0.1**2)
#
case.setAlgorithmParameters(
    Algorithm="ExtendedKalmanFilter",
    Parameters={
        "StoreSupplementaryCalculations":[
            "Analysis",
            "APosterioriCovariance",
            "SimulatedObservationAtCurrentAnalysis",
            ],
        },
    )
#
# Loop to obtain an analysis at each observation arrival
#
XaStep = initState + idealX.tolist()
VaStep = numpy.identity(len(XaStep))
for i in range(1,len(Yobs)):
    case.setBackground         (Vector = XaStep)
    case.setBackgroundError    (Matrix = VaStep)
    case.setObservation        (Vector = Yobs[i])
    case.execute( nextStep = True )
    XaStep = case.get("Analysis")[-1]
    VaStep = case.get("APosterioriCovariance")[-1]
#
Xa = case.get("Analysis")[-1]
Pa = case.get("APosterioriCovariance")
Xs = [float(Pstar) for Pstar in case.get("SimulatedObservationAtCurrentAnalysis")]
#
print("")
print("  Optimal state and parameters\n",Xa)

names =["z", "t", "lambda_p", "K", "gamma_p", "eta", "KDE"]
params = {}
for xa,name in zip(Xa,names):
    # print(f'{name}={xa}',end="; ")
    params[name] = xa
print(params)

print("  Final a posteriori variance:",Pa[-1])
print("Simulated observations:\n",Xs)
print("")

print(len(t),len(Observations))
# PlotOneSerie(t,Observations,"Observations KalmanFilter")
print(len(t),len(Xs))

PlotOneSerie(t[:-1],Xs,"Simulated observations KalmanFilter")

plt.plot(t,P,color='red')
plt.show()


#



