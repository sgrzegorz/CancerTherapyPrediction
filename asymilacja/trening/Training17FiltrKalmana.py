import numpy as np
from matplotlib import pyplot as plt
from numpy import array
def unit_step_fun(x,threshold):
    return x*(1 / 2 + 1 / 2 *np.tanh(100 * (x-threshold)))

groundTruthObservation = [139887.9641729862, 118592.95534070632, 93143.49448497065, 60946.635835851695, 52736.022005834195, 44120.91842576219, 40833.18731037969, 49546.029773160015, 45349.49981449229, 47700.610364454, 55522.68045519205, 57034.434277481916, 56863.530305371794, 63386.82590251941, 67158.23048876172, 67967.01936054697, 71688.17098362498, 72384.73058040837, 75564.77701001195, 76028.3053909545, 80526.79792240774, 84514.09585629942, 83856.42756040981, 87953.16361363807, 84218.44278917545, 83676.55949518507, 84815.6711316833, 92867.85791662282, 97855.03670304752, 98283.7605806343, 96558.48188413934, 99511.14092424643, 99136.93729547695, 102000.25809532589, 99844.20807893926, 107903.29550345015, 107231.57694638398, 112835.51378925695, 111479.16211049238, 117376.51964853297, 119753.47922617271, 122092.53952312106, 121360.46480523307, 121548.5377673319, 127664.5031601658, 133145.76311546194, 134050.14210104893, 133922.89065739012, 136068.1350283073, 143172.23030868606, 140281.826602183, 148467.05595800898, 147293.3551082251, 151594.49990313768, 153980.87210151483, 153560.0313528456, 153073.84209471408, 164290.48700061612, 160737.59960060503, 160261.381881788, 169812.73908970982, 166367.761418243, 175171.5720463807, 178177.39419113038, 174403.74618961767, 186658.67594924875, 184813.73790576882, 189826.2193352948, 185830.74512709648, 194797.53321261352, 194935.47346821328, 198172.9912229869, 206713.90911907554, 204548.33694446457, 207849.59738212006, 216943.69544583833, 212822.38853342028, 223382.4689942598, 224258.71970282073, 224343.4551370558, 227934.94211982607, 238757.7762250831, 244984.69568972822, 248270.62206378765, 246610.9344109703, 255195.2416924986, 256435.79065776718, 266087.9235966054, 270951.2850310832, 274257.0386637566, 278792.2165725012, 285544.20574176614, 293229.0979652067, 296310.60313941224, 298832.5518096974, 305379.52920996706, 306471.35068820755, 318258.5891475188, 323169.8982424407, 328035.87533610564]

initState = [142730,1]

base_lambda_p = 6.80157379e-02
base_K = 0.3e6
base_gamma_p = 1e-3
base_eta = 0.05
base_KDE =  9.51318080e-2


idealX = array([base_lambda_p, base_K, base_gamma_p, base_eta, base_KDE])


# our tumor growth model
def rhs(z, t, lambda_p, K, gamma_p, eta, KDE):
    P, C = z

    dCdt =-KDE * C
    dPdt = lambda_p * P*(1-P/K)  - gamma_p * unit_step_fun(C,eta)  * P
    return [dPdt, dCdt]

# !pip install adao
import numpy, scipy, adao
print("Numpy:", numpy.__version__)
print("Scipy:", scipy.__version__)
print("Adao:", adao.version)
numpy.set_printoptions(precision=2,linewidth=5000,threshold=10000)
from adao import adaoBuilder

Observations    = numpy.ravel(groundTruthObservation)
NbObs           = len(Observations)
TimeBetween2Obs = 1
TimeWindows     = NbObs * TimeBetween2Obs
NbOfEulerStep   = 5



def PlotOneSerie(__Observations, __Title=''):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10, 4)
    #
    plt.figure()
    plt.plot(__Observations,'k-')
    plt.title(__Title, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('Value')


Yobs = groundTruthObservation

NbOfEulerStep   = 5


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
    Algorithm="KalmanFilter",
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
# print("  Optimal state and parameters\n",Xa)
# print("  Final a posteriori variance:",Pa[-1])
print("Simulated observations:\n",Xs)
print("")

PlotOneSerie(Observations,"Observations KalmanFilter")
PlotOneSerie(Xs,"Simulated observations KalmanFilter")
plt.show()
#

