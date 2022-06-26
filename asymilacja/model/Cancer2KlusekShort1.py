import numpy as np

def f(x,threshold):
    if x >= threshold: return x
    else: return 0

class CancerModel:
    def __init__(self,lambda_p,gamma_q,gamma_p,KDE,k_pq,K,eta):
        self.lambda_p = lambda_p # the rate constant of growth used in the logistic expression for the expansion of proliferative tissue. Tumor specific
        self.gamma_q =gamma_q # damages in quiescent tissue. Treatment specific
        self.gamma_p =gamma_p # damages in proliferative tissue. Treatment specific
        self.KDE = KDE # KDE is the rate constant for the decay of the PCV concentration in plasma, denoted C.
        self.k_pq = k_pq # the rate constant for transition from proliferation to quiescence. Tumor specific
        self.K =K # fixed maximal tumor size 100 mm
        self.eta = eta

    def model(self, X, t):
        [P,Q,C] = X

        lambda_p = self.lambda_p # the rate constant of growth used in the logistic expression for the expansion of proliferative tissue. Tumor specific
        gamma_q = self.gamma_q  # damages in quiescent tissue. Treatment specific
        gamma_p = self.gamma_p # damages in proliferative tissue. Treatment specific
        KDE = self.KDE  # KDE is the rate constant for the decay of the PCV concentration in plasma, denoted C.
        k_pq =self.k_pq  # the rate constant for transition from proliferation to quiescence. Tumor specific
        K = self.K
        eta = self.eta  # stopień żeby minimalna zawartosc lekarstwa nie wplywala na komorki

        dCdt = -KDE * C
        dPdt = lambda_p * P  - k_pq * P - gamma_p * f(C,eta) * KDE * P
        dQdt = k_pq * P - gamma_q * f(C,eta) * KDE* Q
        return [dPdt, dQdt,dCdt]

    def time_interval(self, start,end):
        return np.linspace(start,end,np.abs(start-end)+1)
