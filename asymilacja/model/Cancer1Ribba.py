import numpy as np


class CancerModel:
    def __init__(self,lambda_p,delta_qp,gamma_q,gamma_p,KDE,k_qpp,k_pq,K):
        self.lambda_p = lambda_p # the rate constant of growth used in the logistic expression for the expansion of proliferative tissue. Tumor specific
        self.delta_qp = delta_qp  # rate constant for elimination of the damaged quiescent tissue.
        self.gamma_q =gamma_q # damages in quiescent tissue. Treatment specific
        self.gamma_p =gamma_p # damages in proliferative tissue. Treatment specific
        self.KDE = KDE # KDE is the rate constant for the decay of the PCV concentration in plasma, denoted C.
        self.k_qpp =k_qpp # the rate constant for transfer from damaged quiescent tissue to proliferative tissue,
        self.k_pq = k_pq # the rate constant for transition from proliferation to quiescence. Tumor specific
        self.K =K # fixed maximal tumor size 100 mm

# P Proliferative tissue
# Q Undamaged quiescent tissue
# Q_p Damaged quiescent tissue
# C koncentracja (stężenie) wirtualnego lekarstwa w tkankach
    def model(self, X, t):
        [P,Q, Q_p,C] = X

        lambda_p = self.lambda_p # the rate constant of growth used in the logistic expression for the expansion of proliferative tissue. Tumor specific
        delta_qp =self.delta_qp   # rate constant for elimination of the damaged quiescent tissue.
        gamma_q = self.gamma_q  # damages in quiescent tissue. Treatment specific
        gamma_p = self.gamma_p # damages in proliferative tissue. Treatment specific
        KDE = self.KDE  # KDE is the rate constant for the decay of the PCV concentration in plasma, denoted C.
        k_qpp =self.k_qpp  # the rate constant for transfer from damaged quiescent tissue to proliferative tissue,
        k_pq =self.k_pq  # the rate constant for transition from proliferation to quiescence. Tumor specific
        K = self.K


        dCdt = -KDE * C
        dPdt = lambda_p * P*(1 - (P + Q + Q_p)/K) + k_qpp * Q_p - k_pq * P - gamma_p * C * KDE * P
        dQdt = k_pq * P - gamma_q * C * KDE * Q
        dQ_pdt = gamma_q * C *KDE * Q - k_qpp * Q_p - delta_qp * Q_p
        return [dPdt, dQdt, dQ_pdt,dCdt]

    def time_interval(self, start,end):
        return np.linspace(start,end,np.abs(start-end)+1)
