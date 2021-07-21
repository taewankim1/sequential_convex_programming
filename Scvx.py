
# coding: utf-8

# In[4]:
from __future__ import division
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


# In[5]:
import cost
import model
import IPython

from Scaling import compute_scaling


# In[7]:

class Scvx:
    def __init__(self,name,horizon,maxIter,Model,Cost,Const,type_discretization='zoh',w_vc=1e4,w_tr=1e-3,tol_vc=1e-10,tol_tr=1e-3,tol_bc=1e-3):
        self.name = name
        self.model = Model
        self.const = Const
        self.cost = Cost
        self.N = horizon
        
        # cost optimization
        self.verbosity = True
        self.w_vc = w_vc
        self.w_tr = w_tr
        self.tol_fun = 1e-6
        self.tol_tr = tol_tr
        self.tol_vc = tol_vc
        self.tol_bc = tol_bc
        self.maxIter = maxIter
        self.last_head = True
        self.type_discretization = type_discretization   
        self.initialize()
        
    def initialize(self) :
        
        self.dV = np.zeros((1,2))
        self.x0 = np.zeros(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.u = np.ones((self.N+1,self.model.iu))
        self.vc = np.ones((self.N,self.model.ix)) * 1e-1
        self.tr = np.ones((self.N+1))

        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N+1,self.model.iu))
        self.vcnew = np.zeros((self.N,self.model.ix))
        self.Alpha = np.power(10,np.linspace(0,-3,11))

        self.A = np.zeros((self.N,self.model.ix,self.model.ix))
        self.B = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bm = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bp = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.s = np.zeros((self.N,self.model.ix))
        self.z = np.zeros((self.N,self.model.ix))

        self.c = 0
        self.cvc = 0
        self.ctr = 0
        self.cnew = 0
        self.cvcnew = 0
        self.ctrnew = 0

        self.cx = np.zeros((self.N+1,self.model.ix))
        self.cu = np.zeros((self.N,self.model.iu))
        self.cxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
        self.cxu = np.zeros((self.N,self.model.ix,self.model.iu))
        self.cuu = np.zeros((self.N,self.model.iu,self.model.iu))
    


    def forward_full(self,x0,u) :
        N = self.N
        ix = self.model.ix
        iu = self.model.iu

        def dfdt(t,x,um,up) :
            if self.type_discretization == "zoh" :
                u = um
            elif self.type_discretization == "foh" :
                alpha = (self.model.delT - t) / self.model.delT
                beta = t / self.model.delT
                u = alpha * um + beta * up
            return np.squeeze(self.model.forward(x,u,discrete=False))

        xnew = np.zeros((N+1,ix))
        xnew[0] = x0
        cnew = np.zeros(N+1)

        for i in range(N) :
            sol = solve_ivp(dfdt,(0,self.model.delT),xnew[i],args=(u[i],u[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            xnew[i+1] = sol.y[:,-1]
            cnew[i] = self.cost.estimate_cost(xnew[i],u[i])
        cnew[N] = self.cost.estimate_cost(xnew[N],np.zeros(self.model.iu))

        return xnew,u,cnew
        
    def cvxopt(self):
        # TODO - we can get rid of most of loops here

        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        Sx,iSx,sx,Su,iSu,su = compute_scaling(self.x,self.u)


        x_cvx = cvx.Variable((N+1,ix))
        u_cvx = cvx.Variable((N+1,iu))


        vc = cvx.Variable((N,ix))
        # vc.value = np.zeros((N,ix))

        tr = cvx.Variable((N+1),nonneg=True)

        # delx = x_cvx - self.x
        # delu = u_cvx - self.u

        constraints = []
        # initial & final boundary condition
        constraints.append(Sx@x_cvx[0] + sx == self.xi)
        constraints += self.const.bc_final(Sx@x_cvx[-1]+sx,self.xf)

        # trust region
        for i in range(N+1) :
            constraints.append(cvx.quad_form(x_cvx[i] - iSx@(self.x[i]-sx),np.eye(ix)) + \
             cvx.quad_form(u_cvx[i]-iSu@(self.u[i]-su),np.eye(iu)) <= tr[i] )

        # inequality contraints
        for i in range(0,N+1) :
            h = self.const.forward(Sx@x_cvx[i]+sx,Su@u_cvx[i]+su,self.x[i],self.u[i])
            constraints += h
        # IPython.embed()

        # model constraints
        for i in range(0,N) :
            if self.type_discretization == 'zoh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.B[i]@(Su@u_cvx[i]+su)+self.s[i]+self.z[i]+vc[i])
            elif self.type_discretization == 'foh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx) + self.Bm[i]@(Su@u_cvx[i]+su)+self.Bp[i]@(Su@u_cvx[i+1]+su)+self.s[i]+self.z[i]+vc[i]) 

        # cost
        objective = []
        objective_vc = []
        objective_tr = []
        objective_test = []
        for i in range(0,N) :
            objective.append(self.cost.estimate_cost_cvx(Sx@x_cvx[i]+sx,Su@u_cvx[i]+su))
            objective_vc.append(self.w_vc * cvx.norm(vc[i],1))
        objective.append(self.cost.estimate_cost_cvx(Sx@x_cvx[N]+sx,Su@u_cvx[N]+su))
        objective_tr.append(self.w_tr * cvx.norm(tr,2))
        # objective_tr.append(0*self.w_tr * cvx.norm(tr,2))

        l = cvx.sum(objective)
        l_vc = cvx.sum(objective_vc)
        l_tr = cvx.sum(objective_tr)

        l_all = l + l_vc + l_tr
        prob = cvx.Problem(cvx.Minimize(l_all), constraints)
        prob.solve(verbose=False,solver=cvx.ECOS)
        if prob.status == cvx.OPTIMAL_INACCURATE :
            print("WARNING: inaccurate solution")

        x_bar = np.zeros_like(self.x)
        u_bar = np.zeros_like(self.u)
        for i in range(N+1) :
            x_bar[i] = Sx@x_cvx[i].value + sx
            u_bar[i] = Su@u_cvx[i].value + su

        return prob.status,l.value,l_vc.value,l_tr.value,x_bar,u_bar,vc.value,tr.value
                   
        
    def update(self,x0,u0,xi=None,xf=None):
        # initial trajectory
        self.x0 = x0
        
        # initial input
        self.u = u0

        # initial condition
        if xi is None :
            self.xi = x0[0]
        else :
            self.xi = xi

        # final condition
        if xf is None :
            self.xf = x0[-1]
        else :
            self.xf = xf
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # timer setting
        # trace for iteration
        # timer, counters, constraints
        # timer begin!!
        
        # generate initial trajectory
        diverge = False
        stop = False

        self.x = self.x0
        self.c = np.sum(self.cost.estimate_cost(self.x[:N,:],self.u[:N]))
        self.c += self.cost.estimate_cost(self.x[N,:],np.zeros(iu))
        self.cvc = 0
        self.ctr = 0


        # iterations starts!!
        flgChange = True
        total_num_iter = 0
        for iteration in range(self.maxIter) :
            # differentiate dynamics and cost
            if flgChange == True:
                start = time.time()
                if self.type_discretization == 'zoh' :
                    self.A,self.B,self.s,self.z = self.model.diff_discrete_zoh(self.x[0:N,:],self.u[0:N,:])
                elif self.type_discretization == 'foh' :
                    self.A,self.Bm,self.Bp,self.s,self.z = self.model.diff_discrete_foh(self.x[0:N,:],self.u)

                # IPython.embed()
                # remove small element
                eps_machine = np.finfo(float).eps
                self.A[np.abs(self.A) < eps_machine] = 0
                self.B[np.abs(self.B) < eps_machine] = 0
                self.Bm[np.abs(self.Bm) < eps_machine] = 0
                self.Bp[np.abs(self.Bp) < eps_machine] = 0

                flgChange = False
                pass

            time_derivs = (time.time() - start)

            # step2. cvxopt
            try :
                prob_status,l,l_vc,l_tr,self.xbar,self.ubar,self.vcnew, self.trnew = self.cvxopt()
            except cvx.SolverError :
                print("FAIL : Solver fail")
                total_num_iter = -2
                break
            # IPython.embed()

            # step3. line-search to find new control sequence, trajectory, cost
            flag_cvx = False
            flag_boundary = False
            if prob_status == cvx.OPTIMAL or prob_status == cvx.OPTIMAL_INACCURATE :
                flag_cvx = True
                start = time.time()
                # self.xnew,self.unew,self.cnew = self.forward_piecewise(self.xbar,self.ubar)
                self.xnew,self.unew,self.cnew = self.forward_full(self.x0[0,:],self.ubar)
                self.cnew = np.sum(self.cnew)
                self.cvcnew = np.sum(self.w_vc*np.linalg.norm(self.vcnew,1,1))
                self.ctrnew = np.sum(self.w_tr*np.linalg.norm(self.trnew,2))

                dcost = self.c + self.cvc + self.ctr - self.cnew - self.cvcnew - self.ctrnew
                expected = self.c + self.cvc + self.ctr - l - l_vc - l_tr
                # check the boundary condtion
                bc_error_norm = np.linalg.norm(self.xnew[-1,self.const.idx_bc_f]-self.xf[self.const.idx_bc_f],2)
                if  bc_error_norm >= self.tol_bc :
                    # print("{:12.3g} Boundary conditions are not satisified: just accept this step".format(bc_error_norm))
                    flag_boundary = False
                else :
                    flag_boundary = True
                if expected < 0 and iteration > 0 :
                    print("non-positive expected reduction")
                time_forward = time.time() - start
            else :
                print("CVXOPT Failed: should not occur")
                dcost = 0
                expected = 0

            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                print("iteration   total_cost  cost        ||vc||     ||tr||       reduction   expected    w_tr        bounary")
            if flag_cvx == True:
                if self.verbosity == True:
                    print("%-12d%-12.3g%-12.3g%-12.3g%-12.3g%-12.3g%-12.3g%-12.1f%-1d(%2.3g)" % ( iteration,self.c+self.cvc+self.ctr,
                                                                                        self.c,self.cvc,self.ctr,
                                                                                        dcost,expected,self.w_tr,flag_boundary,bc_error_norm))

                # accept changes
                self.x = self.xbar
                self.u = self.ubar
                self.vc = self.vcnew
                self.tr = self.trnew
                self.c = l #self.cnew #l
                self.cvc = l_vc #self.cvcnew #l_vc
                self.ctr = l_tr #self.ctrnew #l_tr
                flgChange = True

                # if iteration > 1 and \
                if iteration > 1 and flag_boundary == True and  \
                                np.linalg.norm(self.trnew,2) < self.tol_tr and np.max(np.linalg.norm(self.vcnew,1,1)) < self.tol_vc :
                    if self.verbosity == True:
                        print("SUCCEESS: virtual control and trust region < tol")
                        total_num_iter = iteration
                    break
                

            else :
                # reduce trust region
                # if self.w_tr <= 10 :
                    # print("TERMINATED: trust region < 10",self.w_tr)
                total_num_iter = -1
                break
                print("increase the w_tr")
                self.w_tr = self.w_tr / 10
                # print status
                if self.verbosity == True :
                    print("%-12d%-12s%-12s%-12s%-12.3g%-12.3g%-12.1f" % ( iteration,'NOSTEP','NOSTEP','NOSTEP',dcost,expected,self.w_tr))

            if total_num_iter == self.maxIter - 1 :
                print("FAIL : reached to max iteration")
                total_num_iter = -1
            
            self.xppg,self.uppg,self.cppg = self.forward_full(self.x0[0,:],self.ubar)

        return self.xnew, self.unew, self.xbar, self.ubar, total_num_iter
        


        
        
        
        
        
        
        
        
        
        
        
        


