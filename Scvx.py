
# coding: utf-8

# In[4]:
from __future__ import division
import matplotlib.pyplot as plt
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


# In[7]:

class Scvx:
    def __init__(self,name,horizon,maxIter,Model,Cost,Const):
        self.name = name
        self.model = Model
        self.const = Const
        self.cost = Cost
        self.N = horizon
        
        # cost optimization
        self.verbosity = True
        self.lambda_nu = 1e3
        self.tr_radius = 10
        self.rho0 = 0
        self.rho1 = 0.25
        self.rho2 = 0.9
        self.tr_alpha = 2
        self.tolFun = 1e-6
        # self.tolGrad = 1e-4
        self.maxIter = maxIter
        self.last_head = True
        
        self.initialize()
        
    def initialize(self) :
        
        self.dV = np.zeros((1,2))
        self.x0 = np.zeros(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.u = np.ones((self.N,self.model.iu))
        self.nu = np.ones((self.N,self.model.ix)) * 1e-3
        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N,self.model.iu))
        self.nunew = np.zeros((self.N,self.model.ix))
        self.Alpha = np.power(10,np.linspace(0,-3,11))
        self.fx = np.zeros((self.N,self.model.ix,self.model.ix))
        self.fu = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.hx = np.zeros((self.N+1,self.const.ih,self.const.ix))
        self.hu = np.zeros((self.N,self.const.ih,self.const.iu))  
        self.c = np.zeros(self.N+1)
        self.cnu = np.ones(self.N)
        self.cnew = np.zeros(self.N+1)
        self.cnunew = np.zeros(self.N)
        self.cx = np.zeros((self.N+1,self.model.ix))
        self.cu = np.zeros((self.N,self.model.iu))
        self.cxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
        self.cxu = np.zeros((self.N,self.model.ix,self.model.iu))
        self.cuu = np.zeros((self.N,self.model.iu,self.model.iu))
    
    def forward(self,x0,u,delu,alpha):
        # TODO - change integral method to odefun
        # horizon
        N = self.N
        
        # variable setting
        xnew = np.zeros((N+1,self.model.ix))
        unew = np.zeros((N,self.model.iu))
        cnew = np.zeros(N+1)
        
        # initial state
        xnew[0,:] = x0
        
        # roll-out
        for i in range(N):
            unew[i,:] = u[i,:] + alpha * delu[i,:]
            xnew[i+1,:] = self.model.forward(xnew[i,:],unew[i,:],i)
            cnew[i] = self.cost.estimate_cost(xnew[i,:],unew[i,:])
            
        cnew[N] = self.cost.estimate_cost(xnew[N,:],np.zeros(self.model.iu))
        return xnew,unew,cnew
        
    def cvxopt(self):
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        delx = cvx.Variable((N+1,ix))
        delu = cvx.Variable((N,iu))
        nu = cvx.Variable((N,ix))
        
        x_new = self.x + delx
        u_new = self.u + delu 

        constraints = []
        # boundary condition
        constraints.append(x_new[0,:] == self.x0[0])
        constraints.append(x_new[-1,:] == self.x0[-1])

        # trust region
        constraints.append(cvx.norm(delu,"inf") <= self.tr_radius)

        # inequality contraints
        h = self.const.forward(self.x,np.vstack((self.u,np.zeros(2))))
        # IPython.embed()
        for i in range(0,N) :
            constraints.append(h[i] + self.hx[i]@delx[i] + self.hu[i]@delu[i] <= 0)
        constraints.append(h[N] + self.hx[N]@delx[N] <= 0)


        objective = []
        objective_vc = []

        for i in range(0,N) :
            constraints.append(delx[i+1,:] == self.fx[i,:,:]@delx[i,:] + self.fu[i,:,:]@delu[i,:] + nu[i,:] )
            objective.append(self.cost.estimate_cost_cvx(x_new[i],u_new[i]))
            objective_vc.append(self.lambda_nu * cvx.norm(nu[i,:],1))
            # objective_quad.append(cx[i]@delx[i] + cu[i]@delu[i] + 
            #                     0.5*cvx.quad_form(delx[i],cxx[i]) + 
            #                     0.5*cvx.quad_form(delu[i],cuu[i]))# + delx[i]@cxu[i]@delu[i])
        objective.append(self.cost.estimate_cost_cvx(x_new[N],np.zeros(iu)))

        l = cvx.sum(objective)
        l_vc = cvx.sum(objective_vc)

        objective_all = l + l_vc
        prob = cvx.Problem(cvx.Minimize(objective_all), constraints)
        prob.solve(solver=cvx.MOSEK)

        return prob.status,l.value,l_vc.value, delu.value, nu.value
                   
        
    def update(self,x0,u0):
        # current position
        self.x0 = x0
        
        # initial input
        self.u = u0
        
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
        self.c[:N] = self.cost.estimate_cost(self.x[:N,:],self.u)
        self.c[N] = self.cost.estimate_cost(self.x[N,:],np.zeros(iu))
        # for j in range(np.size(self.Alpha,axis=0)):   
        #     for i in range(self.N):
        #         self.x[i+1,:] = self.model.forward(self.x[i,:],self.Alpha[j]*self.u[i,:],i)       
        #         self.c[i] = self.cost.estimate_cost(self.x[i,:],self.Alpha[j]*self.u[i,:])
        #         self.cnu[i] = self.lambda_nu*np.linalg.norm(self.nu[i,:],1)
        #         if  np.max( self.x[i+1,:] ) > 1e8 :                
        #             diverge = True
        #             print("initial trajectory is already diverge")

        #     self.c[self.N] = self.cost.estimate_cost(self.x[self.N,:],np.zeros(self.model.iu))
        #     if diverge == False:
        #         break
        # iterations starts!!
        flgChange = True
        for iteration in range(self.maxIter) :
            # differentiate dynamics and cost
            if flgChange == True:
                start = time.time()
                self.fx, self.fu = self.model.diff(self.x[0:N,:],self.u)
                self.hx[0:N], self.hu = self.const.diff_numeric(self.x[0:N,:],self.u)
                self.hx[N],_ = self.const.diff_numeric(self.x[N:,:],np.zeros((1,iu)))
                # c_x_u = self.cost.diff_cost(self.x[0:N,:],self.u)
                # c_xx_uu = self.cost.hess_cost(self.x[0:N,:],self.u)
                # c_xx_uu = 0.5 * ( np.transpose(c_xx_uu,(0,2,1)) + c_xx_uu )
                # self.cx[0:N,:] = c_x_u[:,0:self.model.ix]
                # self.cu[0:N,:] = c_x_u[:,self.model.ix:self.model.ix+self.model.iu]
                # self.cxx[0:N,:,:] = c_xx_uu[:,0:ix,0:ix]
                # self.cxu[0:N,:,:] = c_xx_uu[:,0:ix,ix:(ix+iu)]
                # self.cuu[0:N,:,:] = c_xx_uu[:,ix:(ix+iu),ix:(ix+iu)]
                # c_x_u = self.cost.diff_cost(self.x[N:,:],np.zeros((1,iu)))
                # c_xx_uu = self.cost.hess_cost(self.x[N:,:],np.zeros((1,iu)))
                # c_xx_uu = 0.5 * ( c_xx_uu + c_xx_uu.T)
                # self.cx[N,:] = c_x_u[0:self.model.ix]
                # self.cxx[N,:,:] = c_xx_uu[0:ix,0:ix]
                flgChange = False
                pass

            time_derivs = (time.time() - start)

            # step2. cvxopt
            prob_status,l,l_vc,delu,self.nunew = self.cvxopt()

            # step3. line-search to find new control sequence, trajectory, cost
            fwdPassDone = False
            if prob_status == 'optimal' :
                start = time.time()
                # non-linear forward
                self.xnew,self.unew,self.cnew = self.forward(self.x0[0,:],self.u,delu,1)
                self.cnunew = self.lambda_nu*np.linalg.norm(self.nunew,1,1)
                dcost = np.sum(self.c) + np.sum(self.cnu) - np.sum( self.cnew ) - np.sum(self.cnunew)
                expected = np.sum(self.c) + np.sum(self.cnu) - l - l_vc
                rho = dcost / expected
                if expected < 0 :
                    print("non-positive expected reduction: should not occur")
                    rho = np.sign(dcost)
                if rho > 0 :
                    fwdPassDone = True
                #     break          
                if fwdPassDone == False :
                    alpha_temp = 1e8 # % signals failure of forward pass
                    pass
                time_forward = time.time() - start
            else :
                print("CVXOPT Failed: should not occur")
                dcost = 0
                expected = 0
                break
            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                print("iteration   cost        cost_vc   reduction    expected    radius_tr")
            if fwdPassDone == True:
                if self.verbosity == True:
                    print("%-12d%-12.3g%-12.3g%-12.3g%-12.3g%-12.1f" % ( iteration,np.sum(self.c),np.sum(self.cnu),dcost,expected,self.tr_radius))

                # accept changes
                self.x = self.xnew
                self.u = self.unew
                self.nu = self.nunew
                self.c = self.cnew
                self.cnu = self.cnunew
                flgChange = True

                # update trust region
                if rho < self.rho1 :
                    self.tr_radius = self.tr_radius / self.tr_alpha

                elif self.rho2 < rho :
                    if self.tr_radius < 1e5 :
                        self.tr_radius = self.tr_radius * self.tr_alpha


                # terminate?
                if expected < self.tolFun :
                    if self.verbosity == True:
                        print("SUCCEESS: cost change < tolFun",dcost)
                    break

            else :
                # reduce trust region
                self.tr_radius = self.tr_radius / self.tr_alpha
                # print status
                if self.verbosity == True :
                    print("%-12d%-12s%-12s%-12.3g%-12.3g%-12.1f" % ( iteration,'NO STEP','NO STEP',dcost,expected,self.tr_radius))
                    # print("%-12d%-12s%-12.3g" %
                        # ( iteration,'NO STEP', dcost))

                # if self.verbosity == True:
                #     print("Failed: cvxopt failed",dcost)


        return self.x, self.u
        


        
        
        
        
        
        
        
        
        
        
        
        


