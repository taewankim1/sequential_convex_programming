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
from Scvx import Scvx

from Scaling import TrajectoryScaling

class Scvx_tf_free(Scvx):
    def __init__(self,name,horizon,tf,maxIter,Model,Cost,Const,Scaling=None,type_discretization='zoh',
                        w_c=1,w_vc=1e4,w_tr=1e-3,tol_vc=1e-10,tol_tr=1e-3,tol_bc=1e-3,
                        flag_policyopt=False,verbosity=True):
        self.name = name
        self.model = Model
        self.const = Const
        self.cost = Cost
        self.N = horizon
        self.tf = tf
        self.delT = tf/horizon
        if Scaling is None :
            self.Scaling = TrajectoryScaling() 
            self.flag_update_scale = True
        else :
            self.Scaling = Scaling
            self.flag_update_scale = False
        
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

    def forward_full(self,x0,u) :
        N = self.N
        ix = self.model.ix
        iu = self.model.iu

        def dfdt(t,x,um,up) :
            if self.type_discretization == "zoh" :
                u = um
            elif self.type_discretization == "foh" :
                alpha = (self.delT - t) / self.delT
                beta = t / self.delT
                u = alpha * um + beta * up
            return np.squeeze(self.model.forward(x,u,discrete=False))

        xnew = np.zeros((N+1,ix))
        xnew[0] = x0

        for i in range(N) :
            # sol = solve_ivp(dfdt,(0,self.delT),xnew[i],args=(u[i],u[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            sol = solve_ivp(dfdt,(0,self.delT),xnew[i],args=(u[i],u[i+1]))
            xnew[i+1] = sol.y[:,-1]

        return xnew,np.copy(u)

    def cvxopt(self):
        # TODO - we can get rid of most of loops here

        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        if self.flag_update_scale is True :
            self.Scaling.update_scaling_from_traj(self.x,self.u)
        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()
        S_sigma = self.Scaling.S_sigma

        x_cvx = cvx.Variable((N+1,ix))
        u_cvx = cvx.Variable((N+1,iu))
        vc = cvx.Variable((N,ix))
        sigma = cvx.Variable(nonneg=True)

        # initial & final boundary condition
        constraints = []
        constraints.append(Sx@x_cvx[0]+sx  == self.xi)
        constraints.append(Sx@x_cvx[-1]+sx  == self.xf)

        # state and input contraints
        for i in range(0,N+1) :
            h = self.const.forward(Sx@x_cvx[i]+sx,Su@u_cvx[i]+su,self.x[i],self.u[i],i==N)
            constraints += h

        # model constraints
        for i in range(0,N) :
            if self.type_discretization == 'zoh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.B[i]@(Su@u_cvx[i]+su)
                                                                            +sigma*S_sigma*self.s[i]
                                                                            +self.z[i]
                                                                            +vc[i])
            elif self.type_discretization == 'foh' :
                constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.Bm[i]@(Su@u_cvx[i]+su)
                                                                            +self.Bp[i]@(Su@u_cvx[i+1]+su)
                                                                            +sigma*S_sigma*self.s[i]
                                                                            +self.z[i]
                                                                            # +self.x_prop_n[i]-self.x_prop[i]
                                                                            +vc[i] 
                                                                            )

        # cost
        w_rate = 1e4
        objective = []
        objective_vc = []
        objective_tr = []
        # objective_rate = []
        objective.append(self.w_c * self.cost.estimate_cost_cvx(sigma*S_sigma))
        for i in range(0,N+1) :
            if i < N :
                objective_vc.append(self.w_vc * cvx.norm(vc[i],1))
                # objective_rate.append(w_rate * cvx.quad_form(u_cvx[i+1]-u_cvx[i],np.diag([1,0,0])))
            # objective_tr.append( self.w_tr * (cvx.quad_form(u_cvx[i]-iSu@(self.u[i]-su),np.eye(iu))) )
            objective_tr.append( self.w_tr * (cvx.quad_form(x_cvx[i] -
                                    iSx@(self.x[i]-sx),np.eye(ix)) + cvx.quad_form(u_cvx[i]-iSu@(self.u[i]-su),np.eye(iu))) )
        objective_tr.append(1 * (sigma-self.tf/S_sigma)**2)

        l = cvx.sum(objective)
        l_vc = cvx.sum(objective_vc)
        l_tr = cvx.sum(objective_tr)
        # l_rate = cvx.sum(objective_rate)

        l_all = l + l_vc + l_tr
        prob = cvx.Problem(cvx.Minimize(l_all), constraints)

        error = False
        prob.solve(verbose=True,solver=cvx.MOSEK)
        # prob.solve(verbose=False,solver=cvx.CPLEX)
        # prob.solve(verbose=False,solver=cvx.ECOS)

        if prob.status == cvx.OPTIMAL_INACCURATE :
            print("WARNING: inaccurate solution")

        try :
            x_bar = np.zeros_like(self.x)
            u_bar = np.zeros_like(self.u)
            for i in range(N+1) :
                x_bar[i] = Sx@x_cvx[i].value + sx
                u_bar[i] = Su@u_cvx[i].value + su
            sigma_bar = sigma.value * S_sigma
        except ValueError :
            print(prob.status,"FAIL: ValueError")
            error = True
        except TypeError :
            print(prob.status,"FAIL: TypeError")
            error = True
        # print("x_min {:f} x_max {:f} u_min {:f} u _max{:f}".format(np.min(x_cvx.value),
        #                                                         np.max(x_cvx.value),
        #                                                         np.min(u_cvx.value),
        #                                                         np.max(u_cvx.value)))
        # sigma_bar = self.tf
        return prob.status,l.value,l_vc.value,l_tr.value,x_bar,u_bar,sigma_bar,vc.value,error
                   
        
    def run(self,x0,u0,xi,xf,u_const=None):
        # initial trajectory
        self.x0 = x0

        # save trajectory
        x_traj = []
        u_traj = []
        
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
        self.c = self.tf
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
                    self.A,self.B,self.s,self.z,self.x_prop_n = self.model.diff_discrete_zoh(self.x[0:N,:],self.u[0:N,:],self.delT,self.tf)
                    self.x_prop = np.squeeze(self.A@np.expand_dims(self.x[0:N,:],2) +
                                    self.B@np.expand_dims(self.u[0:N,:],2) + 
                                    np.expand_dims(self.tf*self.s+self.z,2))
                elif self.type_discretization == 'foh' :
                    self.A,self.Bm,self.Bp,self.s,self.z,self.x_prop_n = self.model.diff_discrete_foh_serial(self.x[0:N,:],self.u,self.delT,self.tf)
                    # self.A,self.Bm,self.Bp,self.s,self.z,self.x_prop_n = self.model.diff_discrete_foh_test(self.x[0:N,:],self.u,1/self.N,self.tf)
                    # self.A,self.Bm,self.Bp,self.s,self.z,self.x_prop_n = self.model.diff_discrete_foh(self.x[0:N,:],self.u,self.delT,self.tf)
                    # A_,Bm_,Bp_,s_,z_,x_prop_n_ = self.model.diff_discrete_foh(self.x[0:N,:],self.u,self.delT,self.tf)
                    # print("A",np.max(np.abs(self.A-A_)))
                    # print("Bm",np.max(np.abs(self.Bm-Bm_)))
                    # print("Bp",np.max(np.abs(self.Bp-Bp_)))
                    # print("s",np.max(np.abs(self.s-s_)))
                    # print("z",np.max(np.abs(self.z-z_)))
                    # print("x_prop_n",np.max(np.abs(self.x_prop_n-x_prop_n_)))
                    # self.x_prop = np.squeeze(self.A@np.expand_dims(self.x[0:N,:],2) +
                    #                 self.Bm@np.expand_dims(self.u[0:N,:],2) + 
                    #                 self.Bp@np.expand_dims(self.u[1:N+1,:],2) + 
                    #                 np.expand_dims(self.tf*self.s+self.z,2))
                # print("prop_n - prop",np.sum(self.x_prop_n-self.x_prop))
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
            prob_status,l,l_vc,l_tr,self.xbar,self.ubar,self.sigma,self.vcnew,error = self.cvxopt()
            if error == True :
                total_num_iter = 1e5
                break

            # step3. line-search to find new control sequence, trajectory, cost
            flag_cvx = False
            if prob_status == cvx.OPTIMAL or prob_status == cvx.OPTIMAL_INACCURATE :
                flag_cvx = True
                start = time.time()
                self.xnew,self.unew = self.forward_full(self.x0[0,:],self.ubar)

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
                expected = 0
                break

            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                print("iteration   total_cost        cost        ||vc||     ||tr||       reduction   w_tr        bounary")
            # accept changes
            self.x = self.xbar
            self.u = self.ubar
            self.tf = self.sigma
            self.delT = self.tf/self.N
            self.vc = self.vcnew
            self.c = l 
            self.cvc = l_vc 
            self.ctr = l_tr
            flgChange = True
            x_traj.append(self.x)
            u_traj.append(self.u)

            if self.verbosity == True:
                print("%-12d%-18.3f%-12.3f%-12.3g%-12.3g%-12.3g%-12.3f%-1d(%2.3g)" % ( iteration+1,self.c+self.cvc+self.ctr,
                                                                                    self.c,self.cvc/self.w_vc,self.ctr/self.w_tr,
                                                                                    expected,self.w_tr,flag_boundary,bc_error_norm))
            if flag_boundary == True and  \
                            self.ctr/self.w_tr < self.tol_tr and self.cvc/self.w_vc < self.tol_vc :
                if self.verbosity == True:
                    print("SUCCEESS: virtual control and trust region < tol")
                    total_num_iter = iteration+1
                break
            if iteration == self.maxIter - 1 :
                print("NOT ENOUGH : reached to max iteration")
                total_num_iter = iteration+1

        return self.xnew,self.unew,self.xbar,self.ubar,self.sigma,total_num_iter,flag_boundary,l,l_vc,l_tr,x_traj,u_traj
        


        
        
        
        
        
        
        
        
        
        
        
        


