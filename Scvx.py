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

import cost
import model
import IPython

from Scaling import compute_scaling


class Scvx:
    def __init__(self,name,horizon,maxIter,Model,Cost,Const,type_discretization='zoh',
                        w_c=1,w_vc=1e4,w_tr=1e-3,tol_vc=1e-10,tol_tr=1e-3,tol_bc=1e-3,
                        flag_policyopt=False,verbosity=True):
        self.name = name
        self.model = Model
        self.const = Const
        self.cost = Cost
        self.N = horizon
        
        # cost optimization
        self.verbosity = verbosity
        self.w_c = w_c
        self.w_vc = w_vc
        self.w_tr = w_tr
        # self.tol_fun = 1e-6
        self.tol_tr = tol_tr
        self.tol_vc = tol_vc
        self.tol_bc = tol_bc
        self.maxIter = maxIter
        self.last_head = True
        self.type_discretization = type_discretization   
        self.flag_policyopt = flag_policyopt
        self.initialize()

    def initialize(self) :
        
        self.dV = np.zeros((1,2))
        self.x0 = np.zeros(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.u = np.ones((self.N+1,self.model.iu))
        self.xbar = np.zeros((self.N+1,self.model.ix))
        self.ubar = np.ones((self.N+1,self.model.iu))
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

    def get_model(self) :
        return self.A,self.B,self.s,self.z,self.vc

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
        cnew[N] = self.cost.estimate_final_cost(xnew[N],u[N])

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

        # initial & final boundary condition
        constraints = []
        constraints.append(Sx@x_cvx[0] + sx == self.xi)
        constraints += self.const.bc_final(Sx@x_cvx[-1]+sx,self.xf)

        # state and input contraints
        for i in range(0,N+1) :
            h = self.const.forward(Sx@x_cvx[i]+sx,Su@u_cvx[i]+su,self.x[i],self.u[i],i==N)
            constraints += h

        # model constraints
        for i in range(0,N) :
            if self.type_discretization == 'zoh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.B[i]@(Su@u_cvx[i]+su)+self.s[i]+self.z[i]+vc[i])
            elif self.type_discretization == 'foh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.Bm[i]@(Su@u_cvx[i]+su)
                                                                            +self.Bp[i]@(Su@u_cvx[i+1]+su)
                                                                            +self.s[i]+self.z[i]+vc[i])

        # cost
        objective = []
        objective_vc = []
        objective_tr = []
        for i in range(0,N+1) :
            if i < N :
                objective_vc.append(self.w_vc * cvx.norm(vc[i],1))
            objective.append(self.w_c * self.cost.estimate_cost_cvx(Sx@x_cvx[i]+
                                    sx,Su@u_cvx[i]+su,i))
            if self.flag_policyopt is True :
                objective_tr.append( self.w_tr * (cvx.quad_form(u_cvx[i]-iSu@(self.u_const[i]-su),np.eye(iu))) )
            else :
                objective_tr.append( self.w_tr * (cvx.quad_form(x_cvx[i] -
                                    iSx@(self.x[i]-sx),np.eye(ix)) + cvx.quad_form(u_cvx[i]-iSu@(self.u[i]-su),np.eye(iu))) )

        l = cvx.sum(objective)
        l_vc = cvx.sum(objective_vc)
        l_tr = cvx.sum(objective_tr)

        l_all = l + l_vc + l_tr
        prob = cvx.Problem(cvx.Minimize(l_all), constraints)

        error = False
        try :
            prob.solve(verbose=False,solver=cvx.ECOS,warm_start=True)
        except cvx.SolverError:
            print("FAIL: SolverError")
            error = True

        if prob.status == cvx.OPTIMAL_INACCURATE :
            print("WARNING: inaccurate solution")

        try :
            x_bar = np.zeros_like(self.x)
            u_bar = np.zeros_like(self.u)
            for i in range(N+1) :
                x_bar[i] = Sx@x_cvx[i].value + sx
                u_bar[i] = Su@u_cvx[i].value + su
        except ValueError :
            print("FAIL: ValueError")
            error = True
        except TypeError :
            print("FAIL: TypeError")
            error = True
        # print("x_min {:f} x_max {:f} u_min {:f} u _max{:f}".format(np.min(x_cvx.value),
        #                                                         np.max(x_cvx.value),
        #                                                         np.min(u_cvx.value),
        #                                                         np.max(u_cvx.value)))

        return prob.status,l.value,l_vc.value,l_tr.value,x_bar,u_bar,vc.value,error
                   
        
    def run(self,x0,u0,xi,xf,u_const=None):
        # initial trajectory
        self.x0 = x0
        
        # initial input
        self.u0 = u0
        self.u = u0
        if u_const is None :
            self.u_const = u0
        else :
            self.u_const = u_const

        # initial condition
        self.xi = xi

        # final condition
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
        self.c += self.cost.estimate_final_cost(self.x[N,:],self.u[N,:])
        self.cvc = 0
        self.ctr = 0

        # iterations starts!!
        flgChange = True
        total_num_iter = 0
        flag_boundary = False
        for iteration in range(self.maxIter) :

            # differentiate dynamics and cost
            if flgChange == True:
                start = time.time()
                if self.type_discretization == 'zoh' :
                    self.A,self.B,self.s,self.z = self.model.diff_discrete_zoh(self.x[0:N,:],self.u[0:N,:])
                elif self.type_discretization == 'foh' :
                    self.A,self.Bm,self.Bp,self.s,self.z = self.model.diff_discrete_foh(self.x[0:N,:],self.u)

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
            # try :
            prob_status,l,l_vc,l_tr,self.xbar,self.ubar,self.vcnew,error = self.cvxopt()
            if error == True :
                total_num_iter = 1e5
                break



            # step3. line-search to find new control sequence, trajectory, cost
            flag_cvx = False
            if prob_status == cvx.OPTIMAL or prob_status == cvx.OPTIMAL_INACCURATE :
                flag_cvx = True
                start = time.time()
                self.xnew,self.unew,self.cnew = self.forward_full(self.x0[0,:],self.ubar)
                self.cnew = np.sum(self.cnew)

                dcost = self.c + self.cvc + self.ctr - self.cnew - l_vc - l_tr
                expected = self.c + self.cvc + self.ctr - l - l_vc - l_tr
                # check the boundary condtion
                bc_error_norm = np.linalg.norm(self.xnew[-1,self.const.idx_bc_f]-self.xf[self.const.idx_bc_f],2)
                if  bc_error_norm >= self.tol_bc :
                    # print("{:12.3g} Boundary conditions are not satisified: just accept this step".format(bc_error_norm))
                    flag_boundary = False
                else :
                    flag_boundary = True
                if expected < 0 and iteration > 0 and self.verbosity is True:
                    print("non-positive expected reduction")
                time_forward = time.time() - start
            else :
                print("CVXOPT Failed: should not occur")
                total_num_iter = 1e5
                dcost = 0
                expected = 0
                break

            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                print("iteration   total_cost  cost        ||vc||     ||tr||       reduction   expected    w_tr        bounary")
            # accept changes
            self.x = self.xbar
            self.u = self.ubar
            self.vc = self.vcnew
            self.c = l 
            self.cvc = l_vc 
            self.ctr = l_tr
            flgChange = True

            if self.verbosity == True:
                print("%-12d%-12.6f%-12.6f%-12.3g%-12.3g%-12.3g%-12.3g%-12.6f%-1d(%2.3g)" % ( iteration+1,self.c+self.cvc+self.ctr,
                                                                                    self.c,self.cvc/self.w_vc,self.ctr/self.w_tr,
                                                                                    dcost,expected,self.w_tr,flag_boundary,bc_error_norm))
            if flag_boundary == True and  \
                            self.ctr/self.w_tr < self.tol_tr and self.cvc/self.w_vc < self.tol_vc :
                if self.verbosity == True:
                    print("SUCCEESS: virtual control and trust region < tol")
                    total_num_iter = iteration+1
                break
            if iteration == self.maxIter - 1 :
                print("NOT ENOUGH : reached to max iteration")
                total_num_iter = iteration+1

        return self.xnew,self.unew,self.xbar,self.ubar,total_num_iter,flag_boundary,l,l_vc,l_tr
        


        
        
        
        
        
        
        
        
        
        
        
        


