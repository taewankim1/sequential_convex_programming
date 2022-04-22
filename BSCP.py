from __future__ import division
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
from SCP import SCP
from Scaling import TrajectoryScaling

class BSCP(SCP):
    def __init__(self,name,horizon,tf,maxIter,Model,Cost,Const,Scaling=None,type_discretization='zoh',
                        w_c=1,w_bf=1e4,w_tr=1e-3,tol_bf=1e-10,tol_tr=1e-3,tol_bc=1e-3,
                        flag_policyopt=False,verbosity=True):
        self.name = name
        self.model = Model
        self.const = Const
        self.cost = Cost
        self.N = horizon
        self.tf = tf
        if Scaling is None :
            self.Scaling = TrajectoryScaling() 
            self.flag_update_scale = True
        else :
            self.Scaling = Scaling
            self.flag_update_scale = False
        
        # cost optimization
        self.verbosity = verbosity
        self.w_c = w_c
        self.w_bf = w_bf
        self.w_tr = w_tr
        self.tol_tr = tol_tr
        self.tol_bf = tol_bf
        self.tol_bc = tol_bc
        self.maxIter = maxIter
        self.last_head = True
        self.type_discretization = type_discretization   
        self.flag_policyopt = flag_policyopt
        self.initialize()

    def forward_full(self,x0,u,T,iteration) :
        N = self.N
        ix = self.model.ix
        iu = self.model.iu

        def dfdt(t,x,um,up,delT) :
            if self.type_discretization == "zoh" :
                u = um
            elif self.type_discretization == "foh" :
                alpha = (delT - t) / delT
                beta = t / delT
                u = alpha * um + beta * up
            return np.squeeze(self.model.forward(x,u))

        xnew = np.zeros((N+1,ix))
        xnew[0] = x0

        for i in range(N) :
            # sol = solve_ivp(dfdt,(0,self.delT),xnew[i],args=(u[i],u[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            if iteration < 10 :
                sol = solve_ivp(dfdt,(0,T[i]),xnew[i],args=(u[i],u[i+1],T[i]))
            else :
                sol = solve_ivp(dfdt,(0,T[i]),xnew[i],args=(u[i],u[i+1],T[i]),rtol=1e-6,atol=1e-10)
            xnew[i+1] = sol.y[:,-1]

        return xnew,np.copy(u)

    def cvxopt(self,x,u,T,x_prop):
        # TODO - we can get rid of loops here

        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        ih = self.const.ih
        N = self.N

        if self.flag_update_scale is True :
            self.Scaling.update_scaling_from_traj(self.x,self.u)
        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()
        S_sigma = self.Scaling.S_sigma

        # cvx variabels
        dx_cvx = cvx.Variable((N+1,ix))
        du_cvx = cvx.Variable((N+1,iu))
        dT_cvx = cvx.Variable(N)

        # scaling
        dx,du,dT = [],[],[]
        for i in range(N+1) :
            dx.append(Sx@dx_cvx[i]+sx)
            du.append(Su@du_cvx[i]+su)
            # dx.append(dx_cvx[i])
            # du.append(du_cvx[i])
            if i < N :
                dT.append(S_sigma*dT_cvx[i])

        # bf = cvx.Variable((N+1,ih))
        bf_b = cvx.Variable((2,ix))
        rho = cvx.Variable((N+1))

        # initial & final boundary condition
        constraints = []
        constraints.append(dx[0]+x[0]+bf_b[0]  == self.xi)
        constraints.append(dx[-1]+x[-1]+bf_b[-1]  == self.xf)


        for i in range(0,N+1) :
            # state and input contraints
            # h = self.const.forward_buffer(x[i]+dx[i],u[i]+du[i],bf[i])
            h = self.const.forward(x[i]+dx[i],u[i]+du[i])
            constraints += h
            if i < N :
                # dynamics
                constraints.append(dx[i+1] == self.A[i]@dx[i]+self.Bm[i]@du[i]
                                                +self.Bp[i]@du[i+1]
                                                +self.s[i]*dT[i]
                                                +x_prop[i]-x[i+1]
                                                )
                # trust region
                constraints.append(dT[i]+T[i]>=500/N)
                constraints.append(dT[i]+T[i]<=800/N)
                constraints.append(dT[i]<= 50/N)
                constraints.append(dT[i]>= -50/N)

            constraints.append(cvx.norm(dx_cvx[i]) + cvx.norm(du_cvx[i])<=rho[i])
            # constraints.append(cvx.norm(dx[i]) + cvx.norm(du[i])<=rho[i])


        # cost
        objective = []
        objective_tr = []
        objective_buffer = []

        # final time TODO : should change it to general case
        objective.append(self.w_c * (cvx.sum(dT)+np.sum(T)))
        # for i in range(0,N+1) :
        #     objective_buffer.append(self.w_bf*(cvx.quad_form(bf[i],np.eye(ih))))
        #     objective_tr.append(self.w_tr*(cvx.quad_form(dx_cvx[i],np.eye(ix))+
        #                           cvx.quad_form(du_cvx[i],np.diag([1,1,1]))))
        # objective_buffer.append(self.w_bf_b*(cvx.quad_form(bf_b[0],np.eye(ix))))
        # objective_buffer.append(self.w_bf_b*(cvx.quad_form(bf_b[1],np.eye(ix))))

        # objective_buffer.append(self.w_bf*(cvx.norm(cvx.vec(bf_b)) + cvx.norm(cvx.vec(bf))     ))
        objective_buffer.append(self.w_bf*(cvx.norm(cvx.vec(bf_b))))
        objective_tr.append(self.w_tr * cvx.norm(rho))


        l = cvx.sum(objective)
        l_bf = cvx.sum(objective_buffer)
        l_tr = cvx.sum(objective_tr)

        l_all = l + l_bf + l_tr
        prob = cvx.Problem(cvx.Minimize(l_all), constraints)

        error = False
        # prob.solve(verbose=False,solver=cvx.MOSEK)
        # prob.solve(verbose=False,solver=cvx.CPLEX)
        # prob.solve(verbose=False,solver=cvx.OSQP)#,eps_abs=1e-3,eps_rel=1e-3)
        # prob.solve(verbose=False,solver=cvx.ECOS)
        prob.solve(verbose=False,solver=cvx.GUROBI)

        if prob.status == cvx.OPTIMAL_INACCURATE :
            print("WARNING: inaccurate solution")

        try :
            xbar = np.zeros_like(x)
            ubar = np.zeros_like(u)
            Tbar = np.zeros_like(T)
            for i in range(N+1) :
                xbar[i] = x[i] + dx[i].value
                ubar[i] = u[i] + du[i].value
                if i < N :
                    Tbar[i] = T[i] + dT[i].value
        except ValueError :
            print(prob.status,"FAIL: ValueError")
            # print(dT_cvx.value)
            error = True
        except TypeError :
            print(prob.status,"FAIL: TypeError")
            error = True
        # print(l_tr.value/self.w_tr)
        return prob.status,l.value,l_bf.value,l_tr.value,xbar,ubar,Tbar,error
                   
        
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

        # initial del time
        self.T = np.ones(self.N) * self.tf/self.N

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
        self.cbf = 0
        self.ctr = 0

        # iterations starts!!
        flgChange = True
        total_num_iter = 0
        flag_boundary = False
        for iteration in range(self.maxIter) :
            # Discretization & Linearization
            if flgChange == True:
                start = time.time()
                self.A,self.Bm,self.Bp,self.s,self.z,self.x_prop = self.model.diff_discrete_foh_var_vectorized(self.x[0:N,:],self.u,self.T)
                # self.A,self.Bm,self.Bp,self.s,self.z,self.x_prop = self.model.diff_discrete_foh_var(self.x[0:N,:],self.u,self.T)
                self.print_eigenvalue(self.A)
                # # remove small element
                # eps_machine = np.finfo(float).eps
                # self.A[np.abs(self.A) < eps_machine] = 0
                # self.Bm[np.abs(self.Bm) < eps_machine] = 0
                # self.Bp[np.abs(self.Bp) < eps_machine] = 0
                flgChange = False
                pass
            time_derivs = (time.time() - start)
            # step2. cvxopt
            # try :
            prob_status,l,l_bf,l_tr,self.xbar,self.ubar,self.Tbar,error = self.cvxopt(self.x,self.u,self.T,self.x_prop)
            if error == True :
                total_num_iter = 1e5
                break

            # step3. line-search to find new control sequence, trajectory, cost
            flag_cvx = False
            if prob_status == cvx.OPTIMAL or prob_status == cvx.OPTIMAL_INACCURATE :
                flag_cvx = True
                start = time.time()
                self.xnew,self.unew = self.forward_full(self.x0[0,:],self.ubar,self.Tbar,iteration)

                expected = self.c + self.cbf + self.ctr - l - l_bf - l_tr
                # check the boundary condtion
                bc_error_norm = np.max(np.abs(self.xnew-self.xbar))
                # bc_error_norm = np.linalg.norm(self.xnew[-1,self.const.idx_bc_f]-self.xf[self.const.idx_bc_f],2)

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
                print("iteration   total_cost        cost        ||bf||     ||tr||       reduction   w_tr        bounary")
            # accept changes
            self.x = self.xbar
            self.u = self.ubar
            self.T = self.Tbar
            self.tf = np.sum(self.Tbar)
            self.c = l 
            self.cbf = l_bf 
            self.ctr = l_tr
            flgChange = True
            x_traj.append(self.x)
            u_traj.append(self.u)

            if self.verbosity == True:
                print("%-12d%-18.3f%-12.5f%-12.3g%-12.3g%-12.3g%-12.6f%-1d(%2.3g)" % ( iteration+1,self.c+self.cbf+self.ctr,
                                                                                    self.c,self.cbf/self.w_bf,self.ctr/self.w_tr,
                                                                                    expected,self.w_tr,flag_boundary,bc_error_norm))
            if flag_boundary == True and  \
                            self.ctr/self.w_tr < self.tol_tr and self.cbf/self.w_bf < self.tol_bf :
                if self.verbosity == True:
                    print("SUCCEESS: virtual control and trust region < tol")
                    total_num_iter = iteration+1
                break
            if iteration == self.maxIter - 1 :
                print("NOT ENOUGH : reached to max iteration")
                total_num_iter = iteration+1

        return self.xnew,self.unew,self.xbar,self.ubar,self.Tbar,total_num_iter,flag_boundary,l,l_bf,l_tr,x_traj,u_traj
        


        
        
        
        
        
        
        
        
        
        
        
        


