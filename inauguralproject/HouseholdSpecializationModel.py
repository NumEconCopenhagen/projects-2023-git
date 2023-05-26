
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import math

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0 
        par.theta = 0.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.argmin(HM, HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1+1e-8)/par.sigma)+par.alpha*HF**((par.sigma-1+1e-8)/par.sigma))**(par.sigma/(par.sigma-1+1e-8))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho + 1e-8)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # Define the ratio
        opt.HF_HM = HF[j]/HM[j]

        # Define the log function
        log_HF_HM = math.log(opt.HF / opt.HM)
        log_wF_wM = math.log(par.wF / par.wM)

        # f. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. define objective function
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])
        
        # b. Guesses
        LM_guess = 2
        HM_guess = 2
        LF_guess = 2
        HF_guess = 2

        # c. Result
        res = optimize.minimize(obj, x0= [LM_guess,HM_guess,LF_guess,HF_guess],method='nelder-mead')
        opt.LM_best = res.x[0]
        opt.HM_best = res.x[1]
        opt.LF_best = res.x[2]
        opt.HF_best = res.x[3]

        opt.HF_HM2 = float(opt.HF_best) / float(opt.HM_best)

        # Define the log function
        opt.log_HF_HM2 = math.log(opt.HF_HM2)
        opt.log_wF_wM2 = math.log(par.wF / par.wM)

        # f. print
        return opt.log_HF_HM2, opt.log_wF_wM2, opt, opt.LM_best, opt.HM_best, opt.LF_best, opt.HF_best 

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

    # loop in wF
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF
            q4 = self.solve()
            sol.HF_vec[i] = q4[2].HF_best
            sol.HM_vec[i] = q4[2].HM_best
            sol.LF_vec[i] = q4[2].LF_best
            sol.LM_vec[i] = q4[2].LM_best
        
        return sol.HF_vec, sol.HM_vec, sol.LF_vec, sol.LM_vec

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0, sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol.beta0, sol.beta1
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol

        # define objective function to minimize
        def objective(x):
            alpha, sigma = x
            par.alpha = alpha
            par.sigma = sigma
            self.solve_wF_vec()
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2+(par.beta1_target - sol.beta1)**2
        
        # Guess
        guess = [0.5, 1]

        # call solver
        solution = optimize.minimize(objective, guess, method='Nelder-Mead')
        alp, sig = solution.x

        return alp, sig

        pass

    def estimateS1(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        
        par = self.par
        sol = self.sol

        # define objective function to minimize
        def objective(x):
            alpha, sigma = x
            par.alpha = 0.5
            par.sigma = sigma
            self.solve_wF_vec()
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2+(par.beta1_target - sol.beta1)**2
        
        # initial guess
        guess = [0.5, 1]

        # call solver
        solution = optimize.minimize(objective, guess, method='Nelder-Mead')

        alpha, sigma = solution.x

        return alpha, sigma


    def estimateS2(self, sigma=None):
        """ estimate alpha and sigma """
        def objective(x, self):
            par = self.par
            sol=self.sol
            par.theta = x[0] #Incorporates the theta value into the par object
            par.sigma = x[1]
            self.solve_wF_vec()
            self.run_regression()
            return (0.4-sol.beta0)**2+(-0.1-sol.beta1)**2
        guess = [(0.5,1)]
        result = optimize.minimize(objective, guess, args = (self,), method = 'Nelder-Mead')

        return result
    
    pass 

class HouseholdSpecializationModelClassExtension:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0 
        par.theta = 0.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.argmin(HM, HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1+1e-8)/par.sigma)+par.alpha*HF**((par.sigma-1+1e-8)/par.sigma))**(par.sigma/(par.sigma-1+1e-8))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho + 1e-8)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*((TM+HF*par.theta)**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # Define the ratio
        opt.HF_HM = HF[j]/HM[j]

        # Define the log function
        log_HF_HM = math.log(opt.HF / opt.HM)
        log_wF_wM = math.log(par.wF / par.wM)

        # f. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. define objective function
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])
        
        # b. Guesses
        LM_guess = 2
        HM_guess = 2
        LF_guess = 2
        HF_guess = 2

        # c. Result
        res = optimize.minimize(obj, x0= [LM_guess,HM_guess,LF_guess,HF_guess],method='nelder-mead')
        opt.LM_best = res.x[0]
        opt.HM_best = res.x[1]
        opt.LF_best = res.x[2]
        opt.HF_best = res.x[3]

        opt.HF_HM2 = float(opt.HF_best) / float(opt.HM_best)

        # Define the log function
        opt.log_HF_HM2 = math.log(opt.HF_HM2)
        opt.log_wF_wM2 = math.log(par.wF / par.wM)

        # f. print
        return opt.log_HF_HM2, opt.log_wF_wM2, opt, opt.LM_best, opt.HM_best, opt.LF_best, opt.HF_best 

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

    # loop in wF
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF
            q4 = self.solve()
            sol.HF_vec[i] = q4[2].HF_best
            sol.HM_vec[i] = q4[2].HM_best
            sol.LF_vec[i] = q4[2].LF_best
            sol.LM_vec[i] = q4[2].LM_best
        
        return sol.HF_vec, sol.HM_vec, sol.LF_vec, sol.LM_vec

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0, sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol.beta0, sol.beta1
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol

        # define objective function to minimize
        def objective(x):
            alpha, sigma = x
            par.alpha = alpha
            par.sigma = sigma
            self.solve_wF_vec()
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2+(par.beta1_target - sol.beta1)**2
        
        # Guess
        guess = [0.5, 1]

        # call solver
        solution = optimize.minimize(objective, guess, method='Nelder-Mead')
        alp, sig = solution.x

        return alp, sig

        pass

    def estimateS1(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        
        par = self.par
        sol = self.sol

        # define objective function to minimize
        def objective(x):
            alpha, sigma = x
            par.alpha = 0.5
            par.sigma = sigma
            self.solve_wF_vec()
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2+(par.beta1_target - sol.beta1)**2
        
        # initial guess
        guess = [0.5, 1]

        # call solver
        solution = optimize.minimize(objective, guess, method='Nelder-Mead')

        alpha, sigma = solution.x

        return alpha, sigma


    def estimateS2(self, sigma=None):
        """ estimate alpha and sigma """
        def objective(x, self):
         par = self.par
         sol = self.sol
         par.theta = x[0]
         par.sigma = x[1]
         self.solve_wF_vec()
         self.run_regression()
         desired_beta0 = 0.4  # Update with desired beta0 value
         desired_beta1 = -0.1  # Update with desired beta1 value
         return (desired_beta0 - sol.beta0) ** 2 + (desired_beta1 - sol.beta1) ** 2

        guess = [(1.0,1)]
        result = optimize.minimize(objective, guess, args = (self), method = 'L-BFGS-B')

        return result
    
    pass 
    
